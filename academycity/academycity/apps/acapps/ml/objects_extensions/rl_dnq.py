from ..basic_ml_objects import BaseDataProcessing, BasePotentialAlgo

from ....core.utils import log_debug, clear_log_debug

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import random
import gym
import numpy as np
from collections import deque
from keras.models import Model, load_model
from keras.layers import Input, Dense
from keras.optimizers import Adam, RMSprop


class DQNAgent:
    def __init__(self):
        self.env = gym.make('CartPole-v1')
        # by default, CartPole-v1 has max episode steps = 500
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n
        self.memory = deque(maxlen=2000)

        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.001
        self.epsilon_decay = 0.999
        self.batch_size = 64
        self.train_start = 1000

        # create main model
        self.model = self.create_model()

    def create_model(self):
        X_input = Input((self.state_size,))
        action_space = self.action_size

        # 'Dense' is the basic form of a neural network layer
        # Input Layer of state size(4) and Hidden Layer with 51  nodes
        X = Dense(512, input_shape=(self.state_size,), activation="relu", kernel_initializer='he_uniform')(X_input)

        # Hidden layer with 256 nodes
        X = Dense(256, activation="relu", kernel_initializer='he_uniform')(X)

        # Hidden layer with 64 nodes
        X = Dense(64, activation="relu", kernel_initializer='he_uniform')(X)

        # Output Layer with # of actions:   nodes (left, right)
        X = Dense(action_space, activation="linear", kernel_initializer='he_uniform')(X)

        model = Model(inputs=X_input, outputs=X, name='CartPole')
        model.compile(loss="mse", optimizer=RMSprop(lr=0.00025, rho=0.95, epsilon=0.01), metrics=["accuracy"])

        model.summary()
        return model

    def update_replay_memory(self, transition):
        self.memory.append(transition)
        if len(self.memory) > self.train_start:
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

    def act(self, state):
        if np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            return np.argmax(self.model.predict(state))

    def replay(self):
        if len(self.memory) < self.train_start:
            return
        # Randomly sample minibatch from the memory
        minibatch = random.sample(self.memory, min(len(self.memory), self.batch_size))
        state = np.zeros((self.batch_size, self.state_size))
        next_state = np.zeros((self.batch_size, self.state_size))
        action, reward, done = [], [], []

        # do this before prediction
        # for speedup, this could be done on the tensor level
        # but easier to understand using a loop
        for i in range(self.batch_size):
            state[i] = minibatch[i][0]
            action.append(minibatch[i][1])
            reward.append(minibatch[i][2])
            next_state[i] = minibatch[i][3]
            done.append(minibatch[i][4])

        # do batch prediction to save speed
        target = self.model.predict(state)
        target_next = self.model.predict(next_state)

        for i in range(self.batch_size):
            # correction on the Q value for the action used
            if done[i]:
                target[i][action[i]] = reward[i]
            else:
                # Standard - DQN
                # DQN chooses the max Q value among next actions
                # selection and evaluation of action is on the target Q Network
                # Q_max = max_a' Q_target(s', a')
                target[i][action[i]] = reward[i] + self.gamma * (np.amax(target_next[i]))

        # Train the Neural Network with batches
        self.model.fit(state, target, batch_size=self.batch_size, verbose=0)

    def load(self, name):
        self.model = load_model(name)

    def save(self, name):
        self.model.save(name)

    def train(self, file_name, episodes):
        clear_log_debug()
        for e in range(episodes):
            log_debug("Episode A: " + str(e))
            state = self.env.reset()
            state = np.reshape(state, [1, self.state_size])
            done = False
            i = 0
            log_debug("Episode AA: " + str(e))
            while not done:
                # self.env.render()
                action = self.act(state)
                next_state, reward, done, _ = self.env.step(action)
                next_state = np.reshape(next_state, [1, self.state_size])
                if not done or i == self.env._max_episode_steps - 1:
                    reward = reward
                else:
                    reward = -100
                self.update_replay_memory((state, action, reward, next_state, done))

                state = next_state
                i += 1
                if done:
                    print("episode: {}/{}, score: {}, e: {:.2}".format(e, episodes, i, self.epsilon))
                    if i == 500:
                        print("Saving trained model as cartpole-dqn.keras")
                        log_debug("file_name: " + file_name)
                        log_debug("Episode Z1: ")
                        self.save(file_name)
                        log_debug("Episode Z2: ")
                self.replay()
            log_debug("Episode B: " + str(e))
        log_debug("End Train...")

    def test(self, file_name, episodes):
        clear_log_debug()
        self.load(file_name)
        all_states = []
        for e in range(episodes):
            log_debug("Episode A: " + str(e))
            state = self.env.reset()
            state = np.reshape(state, [1, self.state_size])
            episode_states = []
            for time in range(500):
                action = np.argmax(self.model.predict(state))
                next_state, reward, done, _ = self.env.step(action)
                episode_states.append(next_state.tolist())
                state = np.reshape(next_state, [1, self.state_size])
                if done:
                    break
            all_states.append(episode_states)
            log_debug("Episode B: " + str(e))
        log_debug("End Test...")
        return all_states


class DNQAlgo(object):
    def __init__(self, dic):  # to_data_path, target_field
        # print("90567-8-000 DNQAlgo\n", dic, '\n', '-'*50)
        try:
            super(DNQAlgo, self).__init__()
        except Exception as ex:
            print("Error 9057-010 DNQAlgo:\n"+str(ex), "\n", '-'*50)
        # print("MLAlgo\n", self.app)
        # print("90004-020 DNQAlgo\n", dic, '\n', '-'*50)
        self.app = dic["app"]


class DNQDataProcessing(BaseDataProcessing, BasePotentialAlgo, DNQAlgo):
    def __init__(self, dic):
        # print("90567-010 DNQDataProcessing\n", dic, '\n', '-' * 50)
        super().__init__(dic)
        # print("9005 DNQDataProcessing ", self.app)

    def train(self, dic):
        print("90155-dqn: \n", "="*50, "\n", dic, "\n", "="*50)

        # n_timesteps = 60  # Lookback window of 60 days
        # n_features = 5  # e.g., OHLC and volume
        #
        # # Example of running the model (for simplicity, using random data)
        # X = np.random.rand(1000, n_timesteps, n_features)  # Simulated stock data
        # y = np.random.randint(0, 3, size=(1000,))  # Random actions (buy/sell/hold)
        # print(X, "\n\n", X.shape, "\n\n", y)
        #
        #
        # return

        episodes = int(dic["episodes"])
        # ---

        agent = DQNAgent()
        save_to_file = os.path.join(self.TO_OTHER, "cartpole-dqn.keras")
        agent.train(save_to_file, episodes)

        result = {"status": "ok dqn"}
        return result

    def test(self, dic):
        print("90155-dqn: \n", "="*50, "\n", dic, "\n", "="*50)
        episodes = int(dic["episodes"])
        # ---

        agent = DQNAgent()
        read_from_file = os.path.join(self.TO_OTHER, "cartpole-dqn.keras")
        results = agent.test(read_from_file, episodes)
        for i in results:
            print(i)

        result = {"status": "ok dqn", "results": results}
        return result

