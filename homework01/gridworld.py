import numpy as np

class GridWorld:
    def __init__(self, cols, rows):
        self.cols = cols
        self.rows = rows
        self.moves = ['up', 'down', 'left', 'right']
        self.state = (0, 0)
        self.goal_state = (self.rows - 1, self.cols - 1)
        self.reward = 0
        self.wall = self.create_random_wall(1)
        self.ice = self.create_ice(1)
        self.evaluate()

    def create_random_wall(self, length):
        wall = []
        for i in range(length):
            x = np.random.randint(0, self.cols)
            y = np.random.randint(0, self.rows)
            wall.append((x, y))
        return wall

    def create_ice(self, length):
        ice = []
        for i in range(length):
            x = np.random.randint(0, self.cols)
            y = np.random.randint(0, self.rows)
            ice.append((x, y))
        return ice

    def choosing_an_action(self):
        i = np.random.choice([0, 1, 2, 3], p=[0.25, 0.25, 0.25, 0.25])
        return self.moves[i]

    def up(self):
        next_state = (self.state[0] - 1, self.state[1])
        if self.state[0] == 0:
            return self.state
        elif next_state in self.wall:
            return self.state
        elif next_state in self.ice:
            self.reward -= 2
            return next_state
        else:
            self.reward -= 1
            return next_state


    def down(self):
        next_state = (self.state[0] + 1, self.state[1])
        if self.state[0] == self.rows - 1:
            return self.state
        elif next_state in self.wall:
            return self.state
        elif next_state in self.ice:
            self.reward -= 2
            return next_state
        else:
            self.reward -= 1
            return next_state

    def left(self):
        next_state = (self.state[0], self.state[1] - 1)
        if self.state[1] == 0:
            return self.state
        elif next_state in self.wall:
            return self.state
        elif next_state in self.ice:
            self.reward -= 2
            return next_state
        else:
            self.reward -= 1
            return next_state
        
    def right(self):
        next_state = (self.state[0], self.state[1] + 1)
        if self.state[1] == self.cols - 1:
            return self.state
        elif next_state in self.wall:
            return self.state
        elif next_state in self.ice:
            self.reward -= 2
            return next_state
        else:
            self.reward -= 1
            return next_state

    def is_final_state(self):
        if self.state == self.goal_state:
            return True
        else:
            return False

    def move(self):
        action = self.choosing_an_action()
        if action == 'up':
            return self.up()
        elif action == 'down':
            return self.down()
        elif action == 'left':
            return self.left()
        elif action == 'right':
            return self.right()

    def evaluate(self):
        while not self.is_final_state():
            self.state = self.move()

        print(self.reward)



gridWorld = GridWorld(5, 5)

