import gymnasium as gym
import pandas as pd
import tensorflow as tf
import numpy as np
import seaborn as sns
import ale_py
import shimmy


class ExperienceReplayBuffer:
    def __init__(self, max_size: int, environment_name: str, parallel_game_unrolls: int, observation_preprocessing_function: callable, unroll_steps: int):
        self.max_size = max_size
        self.environment_name = environment_name
        self.parellel_game_unrolls = parallel_game_unrolls
        self.observation_preprocessing_function = observation_preprocessing_function
        self.envs = gym.vector.make(environment_name, num_envs=self.parellel_game_unrolls)
        self.num_possible_actions = self.envs.single_action_space.n
        self.current_states, _ = self.envs.reset()
        self.data = []
        self.unroll_steps = unroll_steps

    def fill_with_samples(self, dqn_network, epsilon: float):
        state_list = []
        actions_list = []
        rewards_list = []
        subsequent_states_list = []
        terminateds_list = []

        for i in range(self.unroll_steps):
            actions = self.sample_epsilon_greedy(dqn_network, epsilon)
            next_states, rewards, terminateds, _, _ = self.envs.step(actions)
            state_list.append(self.current_states)
            actions_list.append(actions)
            rewards_list.append(rewards)
            subsequent_states_list.append(next_states)
            terminateds_list.append(terminateds)

            self.current_states = next_states

        def data_generator():
            for states_batch, actions_batch, rewards_batch, subsequent_states_batch, terminateds_batch in zip(state_list, actions_list, rewards_list, subsequent_states_list, terminateds_list):
                for game_idx in range(self.parellel_game_unrolls):
                    state = states_batch[game_idx, :, :, :]
                    action = actions_batch[game_idx]
                    reward = rewards_batch[game_idx]
                    subsequent_state = subsequent_states_batch[game_idx, :, :, :]
                    terminated = terminateds_batch[game_idx]
                    yield (state, action, reward, subsequent_state, terminated)

        dataset_tensor_specs = (
            tf.TensorSpec(shape=(210, 160, 3), dtype=tf.uint8), 
            tf.TensorSpec(shape=(), dtype=tf.int32), 
            tf.TensorSpec(shape=(), dtype=tf.float32), 
            tf.TensorSpec(shape=(210, 160, 3), dtype=tf.uint8), 
            tf.TensorSpec(shape=(), dtype=tf.bool)
        )
        new_samples_dataset = tf.data.Dataset.from_generator(data_generator, output_signature=dataset_tensor_specs)

        new_samples_dataset = new_samples_dataset.map(
            lambda state, action, reward, subsequent_state, terminated: 
                (self.observation_preprocessing_function(state), action, reward, self.observation_preprocessing_function(subsequent_state), terminated))
        new_samples_dataset.cache().shuffle(buffer_size=self.unroll_steps * self.parellel_game_unrolls)

        # make sure cache is applied
        for elem in new_samples_dataset:
            continue

        self.data.append(new_samples_dataset)
        datapoints_in_data = len(self.data) * self.unroll_steps * self.parellel_game_unrolls
        if datapoints_in_data > self.max_size:
            self.data.pop(0)


    def create_dataset(self):
        erp_dataset = tf.data.Dataset.sample_from_datasets(self.data, weights=[1.0 / len(self.data)] * len(self.data), stop_on_empty_dataset=False)
        return erp_dataset


    def sample_epsilon_greedy(self, dqn_network, epsilon: float):
        observations = self.observation_preprocessing_function(self.current_states)
        q_values = dqn_network(observations) # shape: (parallel_game_unrolls, 4)
        greedy_actions = tf.argmax(q_values, axis=1) # shape: (parallel_game_unrolls,)
        random_actions = tf.random.uniform(shape=(self.parellel_game_unrolls,), minval=0, maxval=self.num_possible_actions, dtype=tf.int64)
        epsilon_sampling = tf.random.uniform(shape=(self.parellel_game_unrolls,), minval=0, maxval=1) > epsilon
        actions = tf.where(epsilon_sampling, greedy_actions, random_actions)
        return actions

def observation_preprocessing_function(observation):
    observation = tf.image.resize(observation, size=(84, 84))
    observation = tf.cast(observation, dtype=tf.float32) / 128.0 - 1.0
    return observation
    

def create_dqn_network(num_actions: int):
    input_layer = tf.keras.layers.Input(shape=(84, 84, 3), dtype=tf.float32)
    x = tf.keras.layers.Conv2D(filters=16, kernel_size=3, activation='relu', padding='same')(input_layer)
    x = tf.keras.layers.Conv2D(filters=16, kernel_size=3, activation='relu', padding='same')(x) + x # residual connection
    x = tf.keras.layers.Conv2D(filters=16, kernel_size=3, activation='relu', padding='same')(x) + x # residual connection
    x = tf.keras.layers.MaxPool2D(pool_size=2)(x)
    x = tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', padding='same')(x)
    x = tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', padding='same')(x) + x # residual connection
    x = tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', padding='same')(x) + x # residual connection
    x = tf.keras.layers.MaxPool2D(pool_size=2)(x)
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu', padding='same')(x)
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu', padding='same')(x) + x # residual connection
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu', padding='same')(x) + x # residual connection
    x = tf.keras.layers.MaxPool2D(pool_size=2)(x)
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu', padding='same')(x) + x # residual connection
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu', padding='same')(x) + x # residual connection
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu', padding='same')(x) + x # residual connection
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(units=64, activation='linear')(x) + x # residual connection
    x = tf.keras.layers.Dense(units=num_actions, activation='linear')(x)
    model = tf.keras.Model(inputs=input_layer, outputs=x)
    return model

def train_dqn(train_dqn_network, target_network, dataset, optimizer, gamma: float, num_training_steps: int, batch_size: int=256):
    dataset = dataset.batch(batch_size).prefetch(4)
    @tf.function
    def training_step(q_target, observations, actions):
        with tf.GradientTape() as tape:
            q_predictions_all_actions = train_dqn_network(observations) # shape: (batch_size, num_actions)
            q_predictions = tf.gather(q_predictions_all_actions, actions, batch_dims=1)
            loss = tf.reduce_mean(tf.square(q_predictions - q_target))
        gradients = tape.gradient(loss, train_dqn_network.trainable_variables)
        optimizer.apply_gradients(zip(gradients, train_dqn_network.trainable_variables))
        return loss
    
    losses = []
    q_values = []
    for i, state_transition in enumerate(dataset):
        state, action, reward, subsequent_state, terminated = state_transition
        q_vals = target_network(subsequent_state)
        q_values.append(q_vals.numpy())
        max_q_values = tf.reduce_max(q_vals, axis=1)
        use_subsequent_state = tf.where(terminated, tf.zeros_like(max_q_values, dtype=tf.float32), tf.ones_like(max_q_values, dtype=tf.float32))
        q_target = reward + (gamma * max_q_values * use_subsequent_state)
        loss = training_step(q_target, observations=state, actions=action)
        losses.append(loss)
        if i >= num_training_steps:
            break
    return np.mean(losses), np.mean(q_values)

def test_q_network(test_dqn_network, environment_name: str, num_parallel_tests: int, gamma: float):
    envs = gym.vector.make(environment_name, num_parallel_tests)
    states, _ = envs.reset()
    done = False
    timestep = 0
    episodes_finished = np.zeros(num_parallel_tests, dtype=np.bool)
    returns = np.zeros(num_parallel_tests)

    while not done:
        q_values = test_dqn_network(observation_preprocessing_function(states))
        actions = tf.argmax(q_values, axis=1)
        states, rewards, terminateds, _, _ = envs.step(actions)
        episodes_finished = np.logical_or(episodes_finished, terminateds)
        returns += ((gamma**timestep)*rewards)*(np.logical_not(episodes_finished).astype(np.float32))
        timestep += 1
        done = np.all(episodes_finished)
    return np.mean(returns)

def polyak_averaging_weights(source_network, target_network, polyak_averaging_factor: float):
    source_network_weights = source_network.get_weights()
    target_network_weights = target_network.get_weights()
    averaged_weights = []
    for source_network_weight, target_network_weight in zip(source_network_weights, target_network_weights):
        fraction_kept_weights = polyak_averaging_factor * target_network_weight
        fraction_updated_weights = (1 - polyak_averaging_factor) * source_network_weight
        average_weight = fraction_kept_weights + fraction_updated_weights
        averaged_weights.append(average_weight)
    target_network.set_weights(averaged_weights)



def dqn():
    ENVIRONMENT_NAME = 'ALE/Breakout-v5'
    NUMBER_ACTIONS = gym.make(ENVIRONMENT_NAME).action_space.n
    ERP_SIZE = 100000
    PARALLEL_GAME_UNROLLS = 64
    UNROLL_STEPS = 4
    EPSILON = 0.2
    GAMMA = 0.98
    NUM_TRAINING_STEPS_PER_ITER = 4
    NUM_TRAINING_ITERS = 50000
    TEST_EVERY_N_STEPS = 50
    PREFILL_STEPS = 100
    POLYAK_AVERAGING_FACTOR = 0.99
    erp = ExperienceReplayBuffer(max_size=ERP_SIZE, environment_name=ENVIRONMENT_NAME, parallel_game_unrolls=PARALLEL_GAME_UNROLLS, 
                                 unroll_steps=UNROLL_STEPS, observation_preprocessing_function=observation_preprocessing_function)
    # This is the DQN we train
    dqn_agent = create_dqn_network(num_actions=NUMBER_ACTIONS)
    # This is the target network to calc the target Q values
    target_network = create_dqn_network(num_actions=NUMBER_ACTIONS)
    dqn_agent.summary()
    dqn_agent(tf.random.uniform(shape=(1, 84, 84, 3)))
    polyak_averaging_weights(dqn_agent, target_network, polyak_averaging_factor=0.)

    dqn_optimizer = tf.keras.optimizers.Adam()
    return_tracker = []
    dqn_prediction_error = []
    average_return_values = []

    # prefill the replay buffer
    prefill_exploration_epsilon = 1.
    for _ in range(PREFILL_STEPS):
        erp.fill_with_samples(dqn_agent, prefill_exploration_epsilon)

    for step in range(NUM_TRAINING_ITERS):
        # Step 1: Put some s, a, r, s' trainsition into the replay buffer
        erp.fill_with_samples(dqn_agent, EPSILON)
        dataset = erp.create_dataset()
        # Step 2: Train some samples from the replay buffer
        average_loss, average_q_values = train_dqn(dqn_agent, target_network, dataset, dqn_optimizer, gamma=GAMMA, num_training_steps=NUM_TRAINING_STEPS_PER_ITER)
        polyak_averaging_weights(dqn_agent, target_network, polyak_averaging_factor=POLYAK_AVERAGING_FACTOR)
        # Step 3: Test the agent
        if step % TEST_EVERY_N_STEPS == 0:
            average_return = test_q_network(dqn_agent, ENVIRONMENT_NAME, num_parallel_tests=10, gamma=GAMMA)
            return_tracker.append(average_return)
            dqn_prediction_error.append(average_loss)
            average_return_values.append(average_q_values)
            print(f'TESTING: Average return: {average_return}, Average loss: {average_loss}, Average Q-value-estimation: {average_q_values}')

    results_dict = {'return_tracker': return_tracker, 'dqn_prediction_error': dqn_prediction_error, 'average_q_values': average_q_values}
    results_df = pd.DataFrame(results_dict)
    sns.lineplot(data=results_df, x=results_df.index, y='return_tracker')

if __name__ == '__main__':
    dqn()