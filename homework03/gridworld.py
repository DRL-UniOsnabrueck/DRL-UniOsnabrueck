import time

import numpy as np
from enum import Enum
import matplotlib.pyplot as plt


class Actions(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


class Memory():
    def __init__(self):
        self.reset()

    def reset(self):
        self.store = []

    def add(self, state, action, reward):
        self.store.append((state, action, reward))

    def dump(self):
        return self.store


class Q_Table():
    def __init__(self, cols, rows, actions):
        self.cols = cols
        self.rows = rows
        self.actions = actions
        self.q = np.zeros((self.rows, self.cols, len(self.actions)))

    def get(self, state, action):
        return self.q[state[0], state[1], action.value]

    def set(self, state, action, value):
        self.q[state[0], state[1], action.value] = value

    def get_max_action(self, state):
        # Get the indices of the max values
        max_indices = np.where(np.array(self.q[state[0], state[1]]) == np.max(
            self.q[state[0], state[1]]))[0]
        # Randomly pick one of the max indices
        random_max_index = np.random.choice(max_indices)
        return random_max_index


class Returns():
    def __init__(self, cols, rows, actions):
        self.returns = [
            [[[] for _ in range(len(actions))] for _ in range(cols)] for _ in range(rows)]

    def append(self, state, action, g):
        self.returns[state[0]][state[1]][action.value].append(g)

    def average(self, state, action):
        action_returns = self.returns[state[0]][state[1]][action.value]
        return np.mean(action_returns) if action_returns else 0


class GridWorld:
    def __init__(self, cols=4, rows=4, alpha=0.5, epsilon=0.2, gamma=0.9, verbose=False, starting_state=None):
        self.cols = cols
        self.rows = rows
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        self.actions = list(Actions)
        self.goal_state = (self.rows - 1, self.cols - 1)
        self.starting_state = self.create_random_position(
            exclude=[self.goal_state]) if starting_state is None else starting_state
        self.reset()
        self.walls = [self.create_random_position(
            exclude=[self.state, self.goal_state])]
        self.ices = [self.create_random_position(
            exclude=[self.state, self.goal_state] + self.walls)]
        self.verbose = verbose

    def reset(self):
        self.episode_reward = 0
        self.q_table = Q_Table(self.cols, self.rows, self.actions)
        self.returns = Returns(self.cols, self.rows, self.actions)
        self.episode_reward_history = []
        self.episode_durations = []
        self.q_values_history = []
        self.state = self.starting_state

    def set_initial_state(self):
        self.state = self.create_random_position(
            exclude=[self.goal_state] + self.walls + self.ices)
        self.episode_reward = 0

    def create_random_position(self, exclude=None):
        if exclude is None:
            exclude = []

        while True:
            x, y = np.random.randint(
                0, self.cols), np.random.randint(0, self.rows)
            pos = (x, y)
            if pos not in exclude:
                return pos

    def pick_action(self, state):
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.choice(list(Actions))

        return Actions(self.q_table.get_max_action(state))

    def up(self):
        return np.max((self.state[0] - 1, 0)), self.state[1]

    def down(self):
        return np.min((self.state[0] + 1, self.rows - 1)), self.state[1]

    def left(self):
        return self.state[0], np.max((self.state[1] - 1, 0))

    def right(self):
        return self.state[0], np.min((self.state[1] + 1, self.cols - 1))

    def is_final_state(self):
        return self.state == self.goal_state

    def move(self, method='sarsa'):
        action = self.pick_action(self.state)

        if action == Actions.UP:
            next_state = self.up()
        elif action == Actions.DOWN:
            next_state = self.down()
        elif action == Actions.LEFT:
            next_state = self.left()
        elif action == Actions.RIGHT:
            next_state = self.right()

        if next_state in self.walls:
            reward = -10
        elif next_state in self.ices:
            reward = -5
        elif next_state == self.goal_state:
            reward = 100
        else:
            reward = -1

        if self.verbose:
            print(f'Current state: {self.state}')
            print(f'Episode reward so far: {self.episode_reward}')
            print(f'Action: {action}')
            print(f'Next state: {next_state}')
            print(f'Reward: {reward}\n')

        self.episode_reward += reward
        if method == 'mc':
            self.memory.add(self.state, action, reward)
        if method == 'sarsa':
            self.update_q_value(self.state, action, next_state, reward)
        self.state = next_state

    def rollout(self, method):
        self.set_initial_state()
        while not self.is_final_state():
            self.move(method)

    def calculate_returns(self):
        g = 0
        processed_pairs = set()
        for sar in reversed(self.memory.store):
            step, action, reward = sar
            g = self.gamma * g + reward
            pair = (step, action)

            if pair not in processed_pairs:
                self.returns.append(step, action, g)
                processed_pairs.add(pair)

    def update_q_value(self, state, action, next_state, reward):
        q = self.q_table.get(state, action)
        next_action = self.pick_action(next_state)
        q_prime = self.q_table.get(next_state, next_action)
        new_q_val = q + self.alpha * (reward + self.gamma * q_prime - q)
        self.q_table.set(state, action, new_q_val)

    def update_q_table(self):
        for row in range(self.rows):
            for col in range(self.cols):
                state = (row, col)
                for action in self.actions:
                    avg_return = self.returns.average(state, action)
                    self.q_table.set(state, action, avg_return)

    def train(self, episodes, method='sarsa'):
        if method == 'mc':
            self.memory = Memory()

        for episode in range(episodes):
            if self.verbose:
                print(f'Episode {episode + 1}')
            start_time = time.time()
            self.rollout(method)
            self.episode_reward_history.append(self.episode_reward)
            if method == 'mc':
                self.calculate_returns()
                self.update_q_table()
                self.memory.reset()
            end_time = time.time()
            duration = end_time - start_time
            self.episode_durations.append(duration)
            self.q_values_history.append(self.q_table.get(self.starting_state, Actions.DOWN))


def plot_average_return_per_episode(episode_reward_history, name='SARSA'):
    episodes = len(episode_reward_history)
    average_return_per_episode = [
        np.mean(episode_reward_history[:i + 1]) for i in range(episodes)]
    plt.plot(range(1, episodes + 1), average_return_per_episode)
    plt.xlabel('Episodes')
    plt.ylabel('Average Return per Episode')
    plt.title(f'Average Return per Episode vs Episodes ({name})')
    plt.show()


def plot_average_return_per_wallclock_time(total_rewards_history, episode_durations, name='SARSA'):
    episodes = len(total_rewards_history)
    average_return_per_episode = [
        np.mean(total_rewards_history[:i + 1]) for i in range(episodes)]
    wall_clock_time = np.cumsum(episode_durations)
    plt.plot(wall_clock_time, average_return_per_episode)
    plt.xlabel('Wall-clock Time (seconds)')
    plt.ylabel('Average Return per Episode')
    plt.title(f'Average Return per Episode vs Wall-clock Time ({name})')
    plt.show()


def main(plot=False):
    num_episodes = 1000
    gridworld = GridWorld(verbose=False, starting_state=(0, 0))
    gridworld.train(num_episodes, method='mc')
    if plot:
        plot_average_return_per_episode(gridworld.episode_reward_history, name='Monte Carlo')
        plot_average_return_per_wallclock_time(
            gridworld.episode_reward_history, gridworld.episode_durations, name='Monte Carlo')
    q_values_mc = gridworld.q_values_history.copy()

    gridworld.reset()
    gridworld.train(num_episodes, method='sarsa')
    if plot:
        plot_average_return_per_episode(gridworld.episode_reward_history, name='SARSA')
        plot_average_return_per_wallclock_time(
            gridworld.episode_reward_history, gridworld.episode_durations, name='SARSA')
    q_values_sarsa = gridworld.q_values_history.copy()

    return q_values_mc, q_values_sarsa


if __name__ == '__main__':
    main(plot=True)
