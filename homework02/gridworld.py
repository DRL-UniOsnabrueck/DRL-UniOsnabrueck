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

    def get_max(self, state):
        return np.max(self.q[state[0], state[1]])

    def get_max_action(self, state):
        # Get the indices of the max values
        max_indices = np.where(np.array(self.q[state[0], state[1]]) == np.max(
            self.q[state[0], state[1]]))[0]
        # Randomly pick one of the max indices
        random_max_index = np.random.choice(max_indices)
        return random_max_index

    def get_max_action_value(self, state):
        return np.max(self.q[state[0], state[1]])


class Returns():
    def __init__(self, cols, rows, actions):
        self.cols = cols
        self.rows = rows
        self.actions = actions
        self.returns = [
            [[[] for _ in range(len(actions))] for _ in range(cols)] for _ in range(rows)]

    def append(self, state, action, g):
        self.returns[state[0]][state[1]][action.value].append(g)

    def average(self, state, action):
        num_returns = len(self.returns[state[0]][state[1]][action.value])
        if num_returns > 0:
            return sum(self.returns[state[0]][state[1]][action.value]) / num_returns
        else:
            return 0


class GridWorld:
    def __init__(self, cols=4, rows=4, epsilon=0.2, gamma=0.9):
        self.cols = cols
        self.rows = rows
        self.epsilon = epsilon
        self.gamma = gamma
        self.goal_state = (self.rows - 1, self.cols - 1)
        self.actions = list(Actions)
        self.wall = self.create_random_wall(1)
        self.ice = self.create_random_ice(1)
        self.set_initial_state()
        self.memory = Memory()
        self.q_table = Q_Table(self.cols, self.rows, self.actions)
        self.returns = Returns(self.cols, self.rows, self.actions)
        self.episode_reward_history = []
        self.episode_durations = []

    def set_initial_state(self):
        self.state = self.get_random_start_state()
        self.episode_reward = 0

    def get_random_start_state(self):
        return np.random.randint(0, self.rows), np.random.randint(0, self.cols)

    def create_random_wall(self, length):
        wall = []
        while len(wall) < length:
            x = np.random.randint(0, self.cols)
            y = np.random.randint(0, self.rows)
            if (x, y) == self.goal_state:
                continue
            wall.append((x, y))
        return wall

    def create_random_ice(self, length):
        ice = []
        for _ in range(length):
            x = np.random.randint(0, self.cols)
            y = np.random.randint(0, self.rows)
            ice.append((x, y))
        return ice

    def pick_action(self):
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.choice(list(Actions))

        return Actions(self.q_table.get_max_action(self.state))

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

    def move(self, verbose=False):
        action = self.pick_action()
        next_state = self.get_next_state(action)

        if next_state in self.wall:
            reward = -10
        elif next_state in self.ice:
            reward = -5
        elif next_state == self.goal_state:
            reward = 100
        else:
            reward = -1

        if verbose:
            print(f'Current state: {self.state}')
            print(f'Episode reward so far: {self.episode_reward}')
            print(f'Action: {action}')
            print(f'Next state: {next_state}')
            print(f'Reward: {reward}\n')

        self.episode_reward += reward
        self.memory.add(self.state, action, reward)
        self.state = next_state

    def get_next_state(self, action):
        if action == Actions.UP:
            return self.up()
        elif action == Actions.DOWN:
            return self.down()
        elif action == Actions.LEFT:
            return self.left()
        elif action == Actions.RIGHT:
            return self.right()

    def rollout(self):
        self.set_initial_state()
        while not self.is_final_state():
            self.move()

    def calculate_returns(self):
        g = 0
        processed_pairs = set()
        for i in range(len(self.memory.store)):
            step, action, reward = self.memory.store[i]
            g = self.gamma * g + reward
            pair = (step, action)

            if pair not in processed_pairs:
                self.returns.append(step, action, g)
                processed_pairs.add(pair)

    def update_q_table(self):
        for row in range(self.rows):
            for col in range(self.cols):
                state = (row, col)
                for action in self.actions:
                    avg_return = self.returns.average(state, action)
                    self.q_table.set(state, action, avg_return)

    def train(self, episodes, verbose=False):
        for episode in range(episodes):
            if verbose:
                print(f'Episode {episode + 1}')
            start_time = time.time()
            self.rollout()
            self.episode_reward_history.append(self.episode_reward)
            self.calculate_returns()
            self.update_q_table()
            self.memory.reset()
            end_time = time.time()
            duration = end_time - start_time
            self.episode_durations.append(duration)


def plot_average_return_per_episode(episode_reward_history):
    episodes = len(episode_reward_history)
    average_return_per_episode = [np.mean(episode_reward_history[:i+1]) for i in range(episodes)]
    plt.plot(range(1, episodes+1), average_return_per_episode)
    plt.xlabel('Episodes')
    plt.ylabel('Average Return per Episode')
    plt.title('Average Return per Episode vs Episodes')
    plt.show()

def plot_average_return_per_wallclock_time(total_rewards_history, episode_durations):
    episodes = len(total_rewards_history)
    average_return_per_episode = [np.mean(total_rewards_history[:i+1]) for i in range(episodes)]
    wall_clock_time = np.cumsum(episode_durations)
    plt.plot(wall_clock_time, average_return_per_episode)
    plt.xlabel('Wall-clock Time (seconds)')
    plt.ylabel('Average Return per Episode')
    plt.title('Average Return per Episode vs Wall-clock Time')
    plt.show()


num_episodes = 100
gridworld = GridWorld()
gridworld.train(num_episodes, verbose=False)
plot_average_return_per_episode(gridworld.episode_reward_history)
plot_average_return_per_wallclock_time(gridworld.episode_reward_history, gridworld.episode_durations)