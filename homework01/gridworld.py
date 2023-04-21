import numpy as np
from enum import Enum


class Actions(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


class GridWorld:
    def __init__(self, cols, rows):
        self.cols = cols
        self.rows = rows
        self.actions = list(Actions)
        self.state = (0, 0)
        self.goal_state = (self.rows - 1, self.cols - 1)
        self.reward = 0
        self.wall = self.create_random_wall(1)
        self.ice = self.create_random_ice(1)
        self.policy = [0.25, 0.25, 0.25, 0.25]
        self.evaluate()

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
        for i in range(length):
            x = np.random.randint(0, self.cols)
            y = np.random.randint(0, self.rows)
            ice.append((x, y))
        return ice

    def pick_action(self):
        return np.random.choice(list(Actions), p=self.policy)

    def up(self):
        if self.state[0] == 0:
            return self.state
        else:
            return self.state[0] - 1, self.state[1]

    def down(self):
        if self.state[0] == self.rows - 1:
            return self.state
        else:
            return self.state[0] + 1, self.state[1]

    def left(self):
        if self.state[1] == 0:
            return self.state
        else:
            return self.state[0], self.state[1] - 1

    def right(self):
        if self.state[1] == self.cols - 1:
            return self.state
        else:
            return self.state[0], self.state[1] + 1

    def is_final_state(self):
        if self.state == self.goal_state:
            return True
        else:
            return False

    def move(self):
        action = self.pick_action()
        next_state = self.get_next_state(action)

        if next_state in self.wall:
            self.reward -= 10
        elif next_state in self.ice:
            self.reward -= 5
            self.state = next_state
        elif next_state == self.goal_state:
            self.reward += 100
            self.state = next_state
        else:
            self.reward -= 1
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

    def update_policy(self):
        distances = self.calculate_manhattan_distances_for_next_states()
        self.policy = self.map_distances_to_probabilities(distances)

    def calculate_manhattan_distances_for_next_states(self):
        next_states = [self.get_next_state(action) for action in self.actions]
        return [self.manhattan_distance(next_state) for next_state in next_states]

    def manhattan_distance(self, state):
        return abs(state[0] - self.goal_state[0]) + abs(state[1] - self.goal_state[1])

    def map_distances_to_probabilities(self, distances):
        distances = np.array(distances)
        probabilities = np.zeros_like(distances)

        # If there is a winning state, set its probability to 1 (there should be only one such state)
        if (distances == 0).any():
            probabilities[distances == 0] = 1
            return probabilities

        # Convert to inverse distances (since we want closer ones to have higher probability)
        distances = 1 / distances

        # Normalize inverse distances so they sum up to 1
        probabilities = distances / np.sum(distances)

        return probabilities

    def evaluate(self):
        while not self.is_final_state():
            print(f'Current state: {self.state}')
            print(f'Current policy: {self.policy}')
            print(f'Current reward: {self.reward}')
            self.move()
            self.update_policy()

        print(self.reward)


gridWorld = GridWorld(5, 5)
