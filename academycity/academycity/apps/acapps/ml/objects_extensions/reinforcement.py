
from ..basic_ml_objects import BaseDataProcessing, BasePotentialAlgo

import json
import numpy as np

#
# https://towardsdatascience.com/hands-on-introduction-to-reinforcement-learning-in-python-da07f7aaca88
# https://medium.com/@ngao7/reinforcement-learning-q-learner-with-detailed-example-and-code-implementation-f7578976473c
#
# https://towardsdatascience.com/reinforcement-learning-with-python-part-1-creating-the-environment-dad6e0237d2d
# https://towardsdatascience.com/deep-reinforcement-learning-with-python-part-2-creating-training-the-rl-agent-using-deep-q-d8216e59cf31
# https://towardsdatascience.com/deep-reinforcement-learning-with-python-part-3-using-tensorboard-to-analyse-trained-models-606c214c14c7
#
ACTIONS = {'U': (-1, 0), 'D': (1, 0), 'L': (0, -1), 'R': (0, 1)}


class Maze(object):
    def __init__(self):
        # start with defining your maze
        self.dimension = 10
        self.maze = np.zeros((self.dimension, self.dimension))
        self.maze[0, 0] = 2
        self.maze[1, 1] = 1
        self.maze[5, :-1] = 1
        self.maze[1:4, 5] = 1
        self.maze[2, 5:] = 1
        self.maze[3, 2] = 1
        self.maze[3, 3] = 1
        self.maze[1, 2] = 1
        self.maze[2, 2] = 1
        self.maze[2, 3] = 1
        self.maze[2, 0] = 1
        self.maze[6, :3] = 1
        self.maze[7, 4:] = 1
        self.maze[8, 7:] = 1
        self.maze[9, :6] = 1

        self.robot_position = (0, 0)  # current robot position
        self.steps = 0  # contains num steps robot took
        self.allowed_states = None  # for now, this is none
        self.construct_allowed_states()  # not implemented yet
        self.path = [(0, 0)]

    def print(self):
        print(self.maze)
        return self.maze
        # print(self.allowed_states)

    def is_allowed_move(self, state, action):
        y, x = state
        y += ACTIONS[action][0]
        x += ACTIONS[action][1]
        # moving off the board
        if y < 0 or x < 0 or y > (self.dimension-1) or x > (self.dimension-1):
            return False
        # moving into start position or empty space
        if self.maze[y, x] == 0 or self.maze[y, x] == 2:
            return True
        else:
            return False

    def construct_allowed_states(self):
        allowed_states = {}
        for y, row in enumerate(self.maze):
            for x, col in enumerate(row):
                # iterate through all valid spaces
                if self.maze[(y, x)] != 1:
                    allowed_states[(y, x)] = []
                    for action in ACTIONS:
                        if self.is_allowed_move((y, x), action):
                            allowed_states[(y, x)].append(action)
        self.allowed_states = allowed_states

    def update_maze(self, action):
        y, x = self.robot_position
        self.maze[y, x] = 0  # set the current position to empty
        y += ACTIONS[action][0]
        x += ACTIONS[action][1]
        self.robot_position = (y, x)
        self.maze[y, x] = 2
        self.steps += 1
        self.path.append((y, x))
        return self.get_state_and_reward()

    def is_game_over(self):
        if self.robot_position == (self.dimension-1, self.dimension-1):
            return True
        return False

    def give_reward(self):
        if self.robot_position == (self.dimension, self.dimension):
            return 0
        else:
            return -1

    def get_state_and_reward(self):
        return self.robot_position, self.give_reward()


class Agent(object):
    def __init__(self, states, alpha=0.15, random_factor=0.2, gama=0.5, decay_rate=10e-4):
        self.alpha = alpha
        self.random_factor = random_factor
        self.gama = gama
        self.decay_rate = decay_rate

        # start the rewards table
        self.G = {}
        self.init_reward(states)

        # print("AAA G", self.G)

    def init_reward(self, states):
        for i, row in enumerate(states):
            for j, col in enumerate(row):
                self.G[(j,i)] = np.random.uniform(high=0.02, low=0.01)

    def learn(self):
        self.random_factor = self.random_factor * (1-self.decay_rate) # decrease random_factor

    def choose_action(self, state, allowed_moves):
        next_move = None
        n = np.random.random()
        if n < self.random_factor:
            next_move = np.random.choice(allowed_moves)
        else:
            max_g = -10e15 # some really small random number
            for action in allowed_moves:
                # print(state, " Actions ", ACTIONS[action])
                # for x in zip(state, ACTIONS[action]):
                #     print("x", x)
                new_state = tuple([sum(x) for x in zip(state, ACTIONS[action])])
                if self.G[new_state] >= max_g:
                    next_move = action
                    max_g = self.G[new_state]
        return next_move

    def update_q_table(self, state, state_n, reward):
        self.G[state] = self.G[state] + self.alpha * (reward + self.gama * self.G[state_n] - self.G[state])


class RIAlgo(object):
    def __init__(self, dic):
        # print("90567-888-000 RIAlgo\n", dic, '\n', '-'*50)
        try:
            super(RIAlgo, self).__init__()
        except Exception as ex:
            print("Error 9057-888-1 RIAlgo:\n"+str(ex), "\n", '-'*50)
        # print("MLAlgo\n", self.app)
        # print("90004-020 RIAlgo\n", dic, '\n', '-'*50)
        self.app = dic["app"]


class RIDataProcessing(BaseDataProcessing, BasePotentialAlgo, RIAlgo):
    def __init__(self, dic):
        # print("90567-888-1 RIDataProcessing\n", dic, '\n', '-' * 50)
        super().__init__(dic)
        # print("9057-888-  RIDataProcessing ", self.app)

    def reinforcement(self, dic):
        print("90555-500: \n", "="*50, "\n", dic, "\n", "="*50)
        maze = Maze()
        maze.print()
        agent = Agent(maze.maze, alpha=0.1, random_factor=0.25, decay_rate=10e-4)
        move_history = []
        best_path = []
        best_steps = 10000
        for i in range(5000):
            if i % 1000 == 0:
                print("i=", i)

            while not maze.is_game_over():
                state_, _ = maze.get_state_and_reward()  # get the current state
                action = agent.choose_action(state_, maze.allowed_states[state_])  # choose an action (explore or exploit)
                state_n, reward = maze.update_maze(action)  # update the maze according to the action
                agent.update_q_table(state_, state_n, reward)
                if maze.steps > 1000:
                    # end the agent if it takes too long to find the goal
                    maze.robot_position = (maze.dimension-1, maze.dimension-1)

            agent.learn()  # robot should learn after every episode
            move_history.append(maze.steps)  # get a history of number of steps taken to plot later
            if maze.steps <= best_steps:
                best_steps = maze.steps
                best_path = maze.path

            maze = Maze()  # reinitialize the maze

        print(move_history[-1])
        print(move_history)
        print(best_steps)
        print(best_path)
        print(agent.random_factor)

        d = {}
        for j in range(maze.maze.shape[0]):
            if j not in d:
                d[j]=[]
            for i in range(maze.maze.shape[1]):
                d[j].append(maze.maze[j, i])
        print(d)

        result = {"status": "ok", "output": {"maze": d, "path": best_path}}
        return result

    def reinforcement_2(self, dic):
        print("90555-500: \n", "="*50, "\n", dic, "\n", "="*50)

        result = {"status": "ok"}
        return result

