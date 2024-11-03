from ..basic_ml_objects import BaseDataProcessing, BasePotentialAlgo
from ....core.templatetags.core_tags import model_name

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

import yfinance as yf
from datetime import datetime, timedelta


import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque

import yfinance as yf
import pandas as pd
from tensorflow.keras import layers

import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import json
from sklearn.metrics import mean_absolute_percentage_error
from abc import ABC, abstractmethod

# # Define a simple LSTM model for trading
# def create_rrl_model(input_shape):
#     model = Sequential()
#     model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
#     model.add(Dropout(0.2))
#     model.add(LSTM(50, return_sequences=False))
#     model.add(Dropout(0.2))
#     model.add(Dense(25, activation='relu'))
#     model.add(Dense(3, activation='softmax'))  # 3 actions: buy, sell, hold
#     return model
#
# class GridWorldEnv:
#     def __init__(self, grid_size=5):
#         self.grid_size = grid_size
#         self.reset()
#
#     def reset(self):
#         self.agent_pos = [random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1)]
#         self.goal_pos = [random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1)]
#         return self.get_observation()
#
#     def step(self, action):
#         # Actions: 0 = left, 1 = right,   = up, 3 = down
#         if action == 0:  # left
#             self.agent_pos[1] = max(0, self.agent_pos[1] - 1)
#         elif action == 1:  # right
#             self.agent_pos[1] = min(self.grid_size - 1, self.agent_pos[1] + 1)
#         elif action == 2:  # up
#             self.agent_pos[0] = max(0, self.agent_pos[0] - 1)
#         elif action == 3:  # down
#             self.agent_pos[0] = min(self.grid_size - 1, self.agent_pos[0] + 1)
#
#         # Reward and termination condition
#         done = self.agent_pos == self.goal_pos
#         reward = 1 if done else -0.1
#         return self.get_observation(), reward, done
#
#     def get_observation(self):
#         # Agent sees a 3x3 grid around it (clipped at boundaries)
#         obs = np.zeros((3, 3))
#         x, y = self.agent_pos
#         for i in range(-1, 2):
#             for j in range(-1, 2):
#                 xi, yj = x + i, y + j
#                 if 0 <= xi < self.grid_size and 0 <= yj < self.grid_size:
#                     obs[i + 1, j + 1] = 1 if [xi, yj] == self.goal_pos else 0
#         return obs
#
#
# class RNN_DQN(nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim):
#         super(RNN_DQN, self).__init__()
#         self.fc1 = nn.Linear(input_dim, hidden_dim)
#         self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
#         self.fc  = nn.Linear(hidden_dim, output_dim)
#         self.hidden_dim = hidden_dim
#
#     def forward(self, x, hidden):
#         x = F.relu(self.fc1(x))
#         x, hidden = self.lstm(x.unsqueeze(1), hidden)  # Add sequence dimension
#         x = F.relu(x)
#         x = self.fc2(x.squeeze(1))
#         return x, hidden
#
#     def init_hidden(self, batch_size):
#         # LSTM hidden state initialization
#         return (torch.zeros(1, batch_size, self.hidden_dim),
#                 torch.zeros(1, batch_size, self.hidden_dim))
#
#
# class ReplayBuffer:
#     def __init__(self, capacity):
#         self.buffer = deque(maxlen=capacity)
#
#     def push(self, sequence):
#         self.buffer.append(sequence)
#
#     def sample(self, batch_size):
#         batch = random.sample(self.buffer, batch_size)
#         return batch
#
#     def __len__(self):
#         return len(self.buffer)
#
#
# class DQNAgent:
#     def __init__(self, input_dim, hidden_dim, output_dim, replay_capacity=1000,
#                  gamma=0.99, epsilon=0.1):
#         self.model = RNN_DQN(input_dim, hidden_dim, output_dim)
#         self.target_model = RNN_DQN(input_dim, hidden_dim, output_dim)
#         self.target_model.load_state_dict(self.model.state_dict())
#         self.target_model.eval()
#
#         self.optimizer = optim.Adam(self.model.parameters())
#         self.replay_buffer = ReplayBuffer(replay_capacity)
#         self.gamma = gamma
#         self.epsilon = epsilon
#         self.batch_size = 32
#         self.hidden_dim = hidden_dim
#
#     def select_action(self, state, hidden):
#         if random.random() > self.epsilon:
#             with torch.no_grad():
#                 q_values, hidden = self.model(state.unsqueeze(0), hidden)
#                 return q_values.argmax().item(), hidden
#         else:
#             r = random.randint(0, 3)
#             return r, hidden
#
#     def update(self):
#         if len(self.replay_buffer) < self.batch_size:
#             return
#
#         batch = self.replay_buffer.sample(self.batch_size)
#
#         loss = 0
#         for sequence in batch:
#             states, actions, rewards, next_states, dones = zip(*sequence)
#
#             hidden = self.model.init_hidden(1)  # Initialize LSTM hidden state
#             q_values, hidden = self.model(states[0].unsqueeze(0), hidden)
#             q_value = q_values.gather(1, actions[0].unsqueeze(0))
#
#             next_hidden = self.target_model.init_hidden(1)
#             next_q_values, next_hidden = self.target_model(next_states[0].unsqueeze(0), next_hidden)
#             next_q_value = next_q_values.max(1)[0].detach()
#             expected_q_value = rewards[0] + self.gamma * next_q_value * (1 - dones[0].int())
#             loss += F.mse_loss(q_value, expected_q_value.unsqueeze(0))
#         self.optimizer.zero_grad()
#         loss.backward()
#         self.optimizer.step()
#
#     def update_target(self):
#         self.target_model.load_state_dict(self.model.state_dict())



# --------------------------------------------
# Trading environment using Yahoo Finance data

class TradingEnv(gym.Env):
    def __init__(self, df):
        super(TradingEnv, self).__init__()

        self.df = df
        self.current_step = 0

        self.action_space = gym.spaces.Discrete(3)  # 0: Hold, 1: Buy, 2: Sell
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)

        self.balance = 1000
        self.shares_held = 0
        self.current_price = 0
        self.net_worth = 1000

    def reset(self):
        self.balance = 1000
        self.shares_held = 0
        self.net_worth = 1000
        self.current_step = 0
        return self._next_observation(), {}

    def _next_observation(self):
        self.current_price = self.df.iloc[self.current_step]['Close']
        obs = np.array([
            self.df.iloc[self.current_step]['Open'],
            self.df.iloc[self.current_step]['Low'],
            self.df.iloc[self.current_step]['High'],
            self.current_price,
            self.df.iloc[self.current_step]['Volume'],
            self.df.iloc[self.current_step]['High'] - self.df.iloc[self.current_step]['Low']
        ])
        return obs

    def step(self, action):
        prev_net_worth = self.net_worth
        if action == 1:  # Buy
            if self.balance > self.current_price:
                self.shares_held += 1
                self.balance -= self.current_price
        elif action == 2:  # Sell
            if self.shares_held > 0:
                self.shares_held -= 1
                self.balance += self.current_price

        self.current_step += 1
        self.net_worth = self.balance + self.shares_held * self.current_price

        reward = self.net_worth - prev_net_worth
        done = self.current_step >= len(self.df) - 1

        return self._next_observation(), reward, done, {}

    def render(self):
        profit = self.net_worth - 1000
        print(f'Step: {self.current_step}')
        print(f'Balance: {self.balance}')
        print(f'Shares held: {self.shares_held}')
        print(f'Net worth: {self.net_worth}')
        print(f'Profit: {profit}')


class AbstractModels(ABC):
    def __init__(self):
        # print("Abstract")
        pass

    @abstractmethod
    def normalize_data(self, **data):
        pass


class Models(AbstractModels):
    def __init__(self, dic):
        super(Models, self).__init__()
        # print("Models 99-99-100", dic)
        self.files_path = dic["files_path"]

    # Function to pull stock data using yfinance
    @staticmethod
    def get_stock_data(ticker, start, end):
        stock_data = yf.download(ticker, start=start, end=end)
        return stock_data[['Open', 'High', 'Low', 'Close', 'Volume']]

    def normalize_data(self, **data):
        print("nd")

    def normalize_data_min_max(self, **data):
            df_ = data["df"]
            scaler_file = data["scaler_file"]
            is_get = data["is_get"]
            if is_get:
                with open(scaler_file, 'rb') as f:
                    scaler = pickle.load(f)
                df = scaler.transform(df_)
            else:
                scaler = MinMaxScaler(feature_range=(0, 1))
                df = scaler.fit_transform(df_)
                with open(scaler_file, 'wb') as f:
                    pickle.dump(scaler, f)
            df = pd.DataFrame(df, columns=df_.columns)
            d_ = {"scaler": scaler, "df":df}
            return d_

    @staticmethod
    def create_sequences(**kwargs):
        data = kwargs["data"]
        timesteps = kwargs["timesteps"]
        X = []
        y = []
        for i in range(len(data) - timesteps):
            X.append(data[i:i + timesteps])
            y.append(data.iloc[i + timesteps]['Close'])  # Predict the 'Close' price
        return np.array(X), np.array(y)

    def get_model(self, dic):
        print("get_model\n", dic)
        model_name_ = dic["model_name"]
        model_index = dic["model_index"]
        ticker = dic["ticker"]
        model_file = self.files_path + "/" + ticker + "_" + model_name_ + "_" + str(model_index) + ".pkl"
        # print(model_file)
        # ---------------
        s_model = "Models." + model_name_ + "(dic)"
        # print(s_model)

        if model_file and os.path.exists(model_file):
            print(f"Loading model from {model_file}")
            model = load_model(model_file)
        else:
            print("Creating new model")
            # self.model = create_lstm_model((1, 6), env.action_space.n)
            s = "Models." + model_name_ + "(dic)"
            print(s)
            model = eval(s_model)
        target_model = eval(s_model)
        target_model.set_weights(model.get_weights())
        # -----------------
        return model, target_model, model_file

    @staticmethod
    def create_lstm_model(dic):
        input_shape = dic["input_shape"]
        output_shape = dic["output_shape"]
        try:
            dropout_rate = dic["dropout_rate"]
        except Exception as ex:
            dropout_rate = 0.2
        try:
            numer_nodes = dic["numer_nodes"]
        except Exception as ex:
            numer_nodes = 64

        model = Sequential([
            layers.LSTM(numer_nodes, return_sequences=True, input_shape=input_shape),
            layers.Dropout(dropout_rate),
            layers.LSTM(numer_nodes, return_sequences=True, input_shape=input_shape),
            layers.Dropout(dropout_rate),
            #
            layers.Dense(32, activation='relu'),
            # layers.Dense(output_shape, activation='softmax')
            layers.Dense(output_shape, activation='linear')
        ])
        model.compile(optimizer='adam', loss='mse')
        return model

    @staticmethod
    def create_cnn_lstm_model(dic):
        input_shape = dic["input_shape"]
        output_shape = dic["output_shape"]
        try:
            dropout_rate = dic["dropout_rate"]
        except Exception as ex:
            dropout_rate = 0.2
        try:
            numer_nodes = dic["numer_nodes"]
        except Exception as ex:
            numer_nodes = 64

        model = Sequential([
            # CNN layers for feature extraction
            layers.Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
            layers.MaxPooling1D(pool_size=2),
            layers.Conv1D(filters=128, kernel_size=3, activation='relu'),
            layers.MaxPooling1D(pool_size=2),
            layers.Dropout(dropout_rate),

            # LSTM layers for sequential data processing
            layers.LSTM(64, return_sequences=True),
            layers.LSTM(64),
            layers.Dropout(dropout_rate),

            # Dense layers for decision making
            layers.Dense(32, activation='relu'),
            layers.Dropout(dropout_rate),

            # Output layer
            layers.Dense(output_shape, activation='softmax')
        ])

        model.compile(optimizer='adam', loss='mse')
        return model


# Trading Agent with LSTM
class TradingAgent:
    def __init__(self, dic):
        self.model = dic["model"]
        self.target_model = dic["target_model"]
        self.env = dic["env"]
        self.target_update_freq = dic["target_update_freq"]
        self.alpha = dic["alpha"]
        self.gama = dic["gama"]
        self.target_update_freq = dic["target_update_freq"]
        # Notice: epsilon is not use due to the choose function

    def choose_action(self, state):
        state = state.reshape((1, 1, 6))  # Reshape for LSTM input
        target_f = self.model.predict(state, verbose=0)
        # print(target_f)
        # probabilities = tf.nn.softmax(target_f[0][0])
        t_ = target_f[0][0]
        # print("t_  ", t_)
        s = sum(t_)
        if s == 0:
            t_ = [1/len(t_) for _ in t_]
        else:
            mi = min(t_)
            ma = max(t_)
            t_ = [(_-mi)/(ma-mi) for _ in t_]
        s = sum(t_)
        probabilities = [x / s for x in t_]
        # print("BBB", t_, probabilities)

        # print("AAA\n", probabilities, "\n", self.env.action_space.n)
        action = np.random.choice(self.env.action_space.n, p=probabilities)
        # print("action=", action)
        return action, target_f

    def train(self, episodes, file_name, scaler):
        for episode in range(episodes):
            print("E", episode)
            state, _= self.env.reset()
            print("F", state)
            done = False
            total_reward = 0
            while not done:
                action, target_f = self.choose_action(state)
                # print("A", target_f)
                next_state, reward, done, _ = self.env.step(action)
                pr_ = self.target_model.predict(next_state.reshape((1, 1, 6)), verbose=0)
                target = reward + (self.gama * np.max(pr_))
                target_f[0][0][action] += self.alpha*(target-target_f[0][0][action])
                # print("B1 target=", target_f[0], target)
                self.model.fit(state.reshape((1, 1, 6)), target_f, epochs=1, verbose=0)
                state = next_state
                total_reward += reward

            if (episode + 1) % self.target_update_freq == 0:
                self.target_model.set_weights(self.model.get_weights())

            if (episode + 1) %   == 0:
                # print("save", episode, file_name)
                self.model.save(file_name)

            print(f'Episode {episode + 1}/{episodes}, Total Reward: {total_reward}')
            # print(next_state)

    def test(self):
        state, _= self.env.reset()
        print(state)
        # print(self.model.get_weights())
        done = False
        total_reward = 0
        while not done:
            action, target_f = self.choose_action(state)
            next_state, reward, done, _ = self.env.step(action)
            # print("h", next_state)
            # print("kk", self.model.predict(next_state.reshape((1, 1, 6))))

            pr_ = self.target_model.predict(next_state.reshape((1, 1, 6)), verbose=0)
            target = reward + (self.gama * np.max(pr_))
            # print("i", target)
            target_f[0][0][action] += self.alpha*(target - target_f[0][0][action])
            self.model.fit(state.reshape((1, 1, 6)), target_f, epochs=1, verbose=0)
            state = next_state
            total_reward += reward
        print("Total Reward", str(total_reward))
        return total_reward

# ---------------------------------------------------------------
class RRLAlgo(object):
    def __init__(self, dic):
        # print("90567-8-000 Algo\n", dic, '\n', '-'*50)
        try:
            super(RRLAlgo, self).__init__()
        except Exception as ex:
            print("Error 9057-010 Algo:\n"+str(ex), "\n", '-'*50)
        # print("MLAlgo\n", self.app)
        # print("90004-020 Algo\n", dic, '\n', '-'*50)
        self.app = dic["app"]

# https://chatgpt.com/c/66e0947c-6714-800c-9probabilitiesef3-3aa45026ed5a
class RRLDataProcessing(BaseDataProcessing, BasePotentialAlgo, RRLAlgo):
    def __init__(self, dic):
        print("90567-010 DataProcessing\n", dic, '\n', '-' * 50)
        super().__init__(dic)
        # print("9005 DataProcessing ", self.app)
        self.MODELS_PATH = os.path.join(self.TO_OTHER, "models")
        os.makedirs(self.MODELS_PATH, exist_ok=True)
        # print(self.MODELS_PATH)
        models_dic = {"files_path": self.MODELS_PATH}
        self.models = Models(models_dic)
        # --------
        self.days_of_investment = int(dic["days_of_investment"])
        self.now_date = datetime.now()
        training_years = int(dic["training_years"])
        self.end_date_1 = self.now_date - timedelta(days=self.days_of_investment-1)
        self.end_date = self.now_date - timedelta(days=self.days_of_investment)
        self.start_date = self.end_date - timedelta(days=training_years*365)
        # -----
        print("self.now_date", self.now_date, "\nself.end_date", self.end_date,
              "\nself.end_date_1", self.end_date_1, "\nself.start_date", self.start_date)
        # -----
        self.SCALER_PATH = os.path.join(self.TO_OTHER, "scalers")
        os.makedirs(self.SCALER_PATH, exist_ok=True)
        self.scaler = None

    def train(self, dic):
        print("\n90199-RLL train: \n", "="*50, "\n", dic, "\n", "="*50)
        # Load stock data using Yahoo Finance
        ticker = dic["ticker"]
        episodes = int(dic["episodes"])
        # ---------------
        print("train: ", ticker, self.start_date, self.end_date)
        df_ = Models.get_stock_data(ticker, self.start_date, self.end_date)
        scaler_file = self.SCALER_PATH + "/" + ticker + ".pkl"
        dic_ = self.models.normalize_data_min_max(df = df_, scaler_file = scaler_file, is_get=False)
        scaler = dic_["scaler"]
        df = dic_["df"]
        print("A\n", df)

        return

        # ---------------
        env = TradingEnv(df)
        # -----------
        model_index = 1
        model_dic = {"model_name": "create_lstm_model",
                     "ticker": ticker,
                     "model_index": model_index,
                     "input_shape": (1, 6),
                     "output_shape": env.action_space.n,
                     "dropout_rate": 0.2,
                     "number_nodes": 64}
        model, target_model, model_file = self.models.get_model(model_dic)
        # -------------------------
        dic = {"env":env, "model": model, "target_model": target_model,
               "gama": 0.99, "alpha": 0.2,"target_update_freq": 1}
        agent = TradingAgent(dic)
        agent.train(episodes=episodes, file_name=model_file, scaler=scaler)
        # ---------------
        print("Training is done")
        result = {"status": "ok"}
        return result

    def test(self, dic):
        print("90222-RLL: \n", "="*50, "\n", dic, "\n", "="*50)
        ticker = dic["ticker"]

        # ---------------
        print("test: ", ticker, self.now_date, self.end_date_1)
        df_ = Models.get_stock_data(ticker, self.end_date_1, self.now_date)
        scaler_file = self.SCALER_PATH + "/" + ticker + ".pkl"
        dic_ = self.models.normalize_data_min_max(df = df_, scaler_file = scaler_file, is_get=True)
        scaler = dic_["scaler"]
        df = dic_["df"]
        print("B\n", df)

        env = TradingEnv(df)
        # -----------
        model_index = 1
        model_dic = {"model_name": "create_lstm_model",
                     "ticker": ticker,
                     "model_index": model_index,
                     "input_shape": (1, 6),
                     "output_shape": env.action_space.n,
                     "dropout_rate": 0.2,
                     "number_nodes": 64}
        model, target_model, model_file = self.models.get_model(model_dic)
        # -------------------------
        dic = {"env":env, "model": model, "target_model": target_model,
               "gama": 0.99, "alpha": 0.2,"target_update_freq": 1}
        agent = TradingAgent(dic)
        total_reward = agent.test()
        # ----------
        result = {"status": "ok", "total_reward": total_reward}
        return result

    def temp_fun(self, dic):
        print("90333-RLL: \n", "="*50, "\n", dic, "\n", "="*50)
        ticker = dic["ticker"]
        data = Models.get_stock_data(ticker, self.start_date, self.end_date)
        data = (data - data.mean()) / data.std()
        print(data)
        # Create sequences
        timesteps = 10
        X, y = Models.create_sequences(data=data, timesteps=timesteps)
        X = X.reshape((X.shape[0], timesteps, X.shape[2], 1))  # (samples, time steps, features, channels)
        print(X, y, X.shape, y.shape)

        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        input_shape = (timesteps, X.shape[2], 1)
        model = Sequential()
        model.add(layers.Conv2D(filters=64, kernel_size=(3, 2), activation='relu', input_shape=input_shape))

        model.add(layers.MaxPooling2D(pool_size=(2, 1)))
        model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
        model.add(layers.MaxPooling2D(pool_size=(2, 1)))

        # Reshape for LSTM layer
        model.add(layers.Reshape((model.output_shape[1], model.output_shape[2] * model.output_shape[3])))

        # LSTM layer for sequence learning
        model.add(layers.LSTM(50, activation='tanh', return_sequences=False))

        # Dense layer for output
        model.add(layers.Dense(50, activation='relu'))
        model.add(layers.Dense(1))  # Output

        # Compile the model
        model.compile(optimizer='adam', loss='mean_squared_error')

        # Train the model
        history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2)

        train_loss = history.history['loss']
        val_loss = history.history['val_loss']
        # --------------------------------------
        # Evaluate the model on the test set
        test_loss = model.evaluate(X_test, y_test)
        print(f'Test Loss: {test_loss}')
        y_pred = model.predict(X_test)

        mape = mean_absolute_percentage_error(y_test, y_pred)
        print(f'Mean Absolute Percentage Error (MAPE): {mape * 100:.2f}%')

        result = {"y_test": json.dumps(y_test.tolist()),
                  "y_pred": json.dumps(y_pred.squeeze().tolist()),
                  "test_loss": test_loss,
                  "mape": mape,
                  "train_loss":train_loss, "val_loss":val_loss}

        # print(result)

        # try:
        #     # Plot training history
        #     plt.plot(history.history['loss'], label='Training Loss')
        #     plt.plot(history.history['val_loss'], label='Validation Loss')
        #     plt.title('Training and Validation Loss')
        #     plt.xlabel('Epochs')
        #     plt.ylabel('Loss')
        #     plt.legend()
        #     plt.show()
        #     plt.pause(0.1)
        #
        #     # Predictions
        #     y_pred = model.predict(X_test)
        #
        #     # Plot the predictions vs actual values
        #     plt.figure(figsize=(12, 6))
        #     plt.plot(y_test, label='Actual', color='blue')
        #     plt.plot(y_pred, label='Predicted', color='red', alpha=0.7)
        #     plt.title('Predictions vs Actual Values')
        #     plt.xlabel('Sample Index')
        #     plt.ylabel('Sine Value')
        #     plt.legend()
        #     plt.show()
        #     plt.pause(0.1)
        # except Exception as ex:
        #     print("Error", ex)
        # ----------
        result = {"status": "ok", "result": result}
        return result

    def train2(self, dic):
        print("90199-RLL: \n", "="*50, "\n", dic, "\n", "="*50)
        episodes = int(dic["episodes"])

        env = GridWorldEnv(grid_size=5)
        agent = DQNAgent(input_dim=9, hidden_dim=128, output_dim=4)
        target_update_freq = 10

        for episode in range(episodes):
            state = torch.tensor(env.reset(), dtype=torch.float32).view(-1)
            hidden = agent.model.init_hidden(1)
            done = False
            sequence = []

            while not done:
                action, hidden = agent.select_action(state, hidden)
                next_state, reward, done = env.step(action)
                next_state = torch.tensor(next_state, dtype=torch.float32).view(-1)

                # print((state, torch.tensor([action]), torch.tensor([reward]), next_state, torch.tensor([done])))
                sequence.append((state, torch.tensor([action]), torch.tensor([reward]), next_state, torch.tensor([done])))
                if len(sequence) >= 10:  # Store only last 10 steps
                    agent.replay_buffer.push(sequence)
                    sequence = []
                state = next_state

            agent.update()

            if episode % target_update_freq == 0:
                agent.update_target()

        result = {"status": "ok"}
        return result

    def train1(self, dic):
        print("90155-dqn: \n", "="*50, "\n", dic, "\n", "="*50)

        # ticker_ = "GOOG" # dic["ticker"]
        # end_date = datetime.now()
        # start_date = end_date - timedelta(days=5000)
        # df = yf.download(ticker_, start_date, end_date)
        # # print("A\n", df)
        # df = df.drop('Close', 1).rename(columns={"Adj Close": "Close"})
        # # print("B\n", df)
        # df = df.reset_index()
        # print(df)

        # =====
        # Example data dimensions
        n_timesteps = 60  # Lookback window of 60 days
        n_features = 5  # e.g., OHLC and volume

        # Create the model
        model = create_rrl_model((n_timesteps, n_features))

        # Compile the model
        model.compile(optimizer='adam', loss='categorical_crossentropy')

        # Example of running the model (for simplicity, using random data)
        X = np.random.rand(1000, n_timesteps, n_features)  # Simulated stock data
        y = np.random.randint(0, 3, size=(1000,))  # Random actions (buy/sell/hold)

        # Train the model (replace with actual market data and actions)
        model.fit(X, tf.keras.utils.to_categorical(y), epochs=10, batch_size=64)

        result = {"status": "ok"}
        return result


