import os
from contextlib import redirect_stdout
import math
import numpy as np
import pandas as pd
import yfinance as yf
from collections import deque
from itertools import islice
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Conv2D, Dropout, Flatten, Reshape, MaxPooling2D
from tensorflow.keras.optimizers import Adam
# -----
from ..basic_ml_objects import BaseDataProcessing, BasePotentialAlgo
# -----
from datetime import datetime, timedelta
import pickle


class Position:
    def __init__(self, balance=100000):
        print("Position")
        self.balance = balance  # Initial balance
        self.holdings = 0
        self.position_value = 0
        self.net_worth = self.balance  # Initial net worth is the balance

    def update_position(self, price, action):
        if action == 0:  # Hold
            pass
        elif action == 1:  # Buy
            shares_to_buy = self.balance // price
            self.holdings += shares_to_buy
            self.balance -= shares_to_buy * price
        elif action == 2:  # Sell
            self.balance += self.holdings * price
            self.holdings = 0
        # print("V", price, action)
        self.position_value = self.holdings * price
        self.net_worth = self.balance + self.position_value
        # print("D")

    def get_total_value(self, price):
        return self.balance + self.holdings * price

class DataProcessing():
    def __init__(self, ticker, scalars_files_path, days_of_investment=30, years_of_data=10.0):
        self.ticker = ticker
        self.scaler_file = scalars_files_path + "/" + ticker + ".pkl"
        # ----------
        self.now_date = datetime.now()
        self.start_train_date = self.now_date - timedelta(days=math.ceil(years_of_data*365))
        self.end_train_date = self.now_date - timedelta(days=days_of_investment-1)
        self.start_test_date = self.end_train_date
        self.end_test_date = self.now_date
        # -----------
        self.stock_data = None
        # -----------

    def load_train_data(self):
        stock_data = yf.download(self.ticker, start=self.start_train_date, end=self.end_train_date)
        self.stock_data = stock_data[['Open', 'High', 'Low', 'Close', 'Volume']]
        scaler = MinMaxScaler(feature_range=(0, 1))
        df = scaler.fit_transform(self.stock_data)
        with open(self.scaler_file, 'wb') as f:
            pickle.dump(scaler, f)
        data = pd.DataFrame(df, index=self.stock_data.index, columns=self.stock_data.columns)
        return data, scaler

    def load_test_data(self):
        stock_data = yf.download(self.ticker, start=self.start_test_date, end=self.end_test_date)
        self.stock_data = stock_data[['Open', 'High', 'Low', 'Close', 'Volume']]
        with open(self.scaler_file, 'rb') as f:
            scaler = pickle.load(f)
        df = scaler.transform(self.stock_data)
        data = pd.DataFrame(df, index=self.stock_data.index, columns=self.stock_data.columns)
        return data, scaler

class Models:
    def __init__(self, models_files_path):
        print("Models")
        self.models_files_path = models_files_path

    @staticmethod
    def create_cnn_lstm_model(dic):
        """
        CNN + LSTM hybrid model for stock data (e.g., Close, Open, High, Low, Volume).
        The model starts with a Conv2D layer followed by an LSTM for sequential processing.

        Args:
            input_shape (tuple): The shape of the input data (time_steps, num_features).
            output_shape (int): The number of actions to predict (output).

        Returns:
            model: The compiled CNN-LSTM model.
        """
        time_steps = dic["input_shape"][0]
        num_features = dic["input_shape"][1]
        output_shape = dic["output_shape"]

        model = Sequential()

        # 1. Reshape input data to (time_steps, num_features, 1) to match the Conv2D input format.
        model.add(Reshape((time_steps, num_features, 1), input_shape=(time_steps, num_features)))
        print(f"Reshape -> Input: {(None, time_steps, num_features)} -> Output: {(None, time_steps, num_features, 1)}")

        # 2. Conv2D layer with 3  filters and (2, 2) kernel.
        model.add(Conv2D(filters=32, kernel_size=(2, 2), activation='relu', padding='valid'))
        print(
            f"Conv2D -> Input: {(None, time_steps, num_features, 1)} -> Output: {(None, time_steps - 1, num_features - 1, 32)}")

        # 3. Dropout layer after Conv2D to prevent overfitting
        model.add(Dropout(0.3))  # Drop 30% of the neurons randomly
        print(f"Dropout after Conv2D")

        # 4. Flatten the output from the Conv2D layer.
        model.add(Flatten())
        print(
            f"Flatten -> Input: {(None, time_steps - 1, num_features - 1, 32)} -> Output: {(None, (time_steps - 1) * (num_features - 1) * 32)}")

        # 5. Reshape the flattened output to (time_steps-1, -1) to prepare it for LSTM.
        model.add(Reshape((time_steps - 1, -1)))
        print(
            f"Reshape -> Input: {(None, (time_steps - 1) * (num_features - 1) * 32)} -> Output: {(None, time_steps - 1, (num_features - 1) * 32)}")

        # 6. Add an LSTM layer with 50 units.
        model.add(LSTM(50, return_sequences=False))
        print(f"LSTM -> Input: {(None, time_steps - 1, (num_features - 1) * 32)} -> Output: {(None, 50)}")

        # 7. Dropout layer after LSTM to prevent overfitting
        model.add(Dropout(0.3))  # Drop 30% of the neurons randomly
        print(f"Dropout after LSTM")

        # 8. Add a Dense layer for action output (output_shape = 3 for Buy/Sell/Hold).
        model.add(Dense(output_shape, activation='linear'))
        print(f"Dense -> Input: {(None, 50)} -> Output: {(None, output_shape)}")

        # Compile the model
        model.compile(optimizer='adam', loss='mse')

        return model

        #
        # """
        # CNN + LSTM hybrid model for stock data (e.g., Close, Open, High, Low, Volume).
        # The model starts with a single Conv2D layer followed by an LSTM for sequential processing.
        #
        # Args:
        #     input_shape (tuple): The shape of the input data (time_steps, num_features).
        #     output_shape (int): The number of actions to predict (output).
        #
        # Returns:
        #     model: The compiled CNN-LSTM model.
        # """
        #
        # model = Sequential()
        #
        # # 1. Reshape input data to (time_steps, num_features, 1) to match the Conv2D input format.
        # #    Input shape: (None, time_steps, num_features)
        # #    Output shape: (None, time_steps, num_features, 1) (since Conv2D expects a channel dimension)
        # model.add(Reshape((time_steps, num_features, 1), input_shape=(time_steps, num_features)))
        # print(f"Reshape -> Input: {(None, time_steps, num_features)} -> Output: {(None, time_steps, num_features, 1)}")
        #
        # # 2. Conv2D layer with 3  filters and (2, 2) kernel.
        # #    Input shape: (None, time_steps, num_features, 1)
        # #    Output shape: (None, time_steps, num_features, 32) (after applying 3  filters)
        # model.add(Conv2D(filters=32, kernel_size=(2, 2), activation='relu', padding='same'))
        # print(
        #     f"Conv2D -> Input: {(None, time_steps, num_features, 1)} -> Output: {(None, time_steps, num_features, 32)}")
        #
        # # 3. Flatten the output from the Conv2D layer.
        # #    Input shape: (None, time_steps, num_features, 32)
        # #    Output shape: (None, time_steps * num_features * 32)
        # model.add(Flatten())
        # print(
        #     f"Flatten -> Input: {(None, time_steps, num_features, 32)} -> Output: {(None, time_steps * num_features * 32)}")
        #
        # # 4. Reshape the flattened output to (time_steps, -1) to prepare it for LSTM.
        # #    Input shape: (None, time_steps * num_features * 32)
        # #    Output shape: (None, time_steps, -1) where -1 is automatically computed (in this case, 32*num_features)
        # model.add(Reshape((time_steps, -1)))
        # print(
        #     f"Reshape -> Input: {(None, time_steps * num_features * 32)} -> Output: {(None, time_steps, num_features * 32)}")
        #
        # # 5. Add an LSTM layer with 50 units.
        # #    Input shape: (None, time_steps, num_features * 32)
        # #    Output shape: (None, 50) (since `return_sequences=False`, only the final output is returned)
        # model.add(LSTM(50, return_sequences=False))
        # print(f"LSTM -> Input: {(None, time_steps, num_features * 32)} -> Output: {(None, 50)}")
        #
        # # 6. Add a Dense layer for action output (output_shape = 3 for Buy/Sell/Hold).
        # #    Input shape: (None, 50)
        # #    Output shape: (None, output_shape) (e.g., (None, 3))
        # model.add(Dense(output_shape, activation='linear'))
        # print(f"Dense -> Input: {(None, 50)} -> Output: {(None, output_shape)}")
        #
        # # Compile the model
        # model.compile(optimizer='adam', loss='mse')
        #
        # return model

    @staticmethod
    def create_simple_lstm_model(dic):
        input_shape = dic["input_shape"]
        output_shape = dic["output_shape"]
        model = Sequential()
        model.add(LSTM(50, input_shape=(input_shape, 1), return_sequences=True))
        model.add(LSTM(50))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(output_shape, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))
        return model

    def get_model(self, dic):
        # print(dic)
        model_name_ = dic["model_name"]
        model_index = dic["model_index"]
        ticker = dic["ticker"]

        model_file = os.path.join(self.models_files_path, f"{ticker}_{model_name_}_{model_index}.pkl")
        # print(model_file)
        print("A model")
        s_model = f"Models.create_{model_name_}_model(dic)"
        # print(s_model)
        if model_file and os.path.exists(model_file):
            model = load_model(model_file)
        else:
            model = eval(s_model)
        print("B target_model")
        target_model = eval(s_model)

        target_model.set_weights(model.get_weights())
        return model, target_model, model_file

    @staticmethod
    def save_model(model, filename):
        with open(os.devnull, 'w') as f, redirect_stdout(f):
            model.save(filename)

    @staticmethod
    def load_model(filename):
        return load_model(filename)

class TradingEnv:
    def __init__(self, data, position, window_size):
        print("TradingEnv")
        self.data = data
        self.window_size = window_size
        self.position = position
        self.current_step = 0
        self.done = False
        self.last_net_worth = self.position.net_worth  # Track last net worth

    def reset(self):
        self.current_step = self.window_size
        self.done = False
        self.last_net_worth = self.position.net_worth
        return self._get_observation()

    def _get_observation(self):
        obs = self.data.iloc[self.current_step - self.window_size:self.current_step]
        return np.expand_dims(obs, axis=0), obs.index[-1].date()

    def step(self, action):
        # print("A step")
        # print("A ste\n", self.data.iloc[self.current_step-1])

        current_price = self.data.iloc[self.current_step-1][3]  # Using the Close price
        # print("current_price", current_price)
        self.position.update_position(current_price, action)

        # Calculate reward based on the change in net worth
        reward = self.position.net_worth - self.last_net_worth
        self.last_net_worth = self.position.net_worth  # Update the last net worth
        self.current_step += 1
        # print(self.current_step, self.data.shape)
        if self.current_step >= len(self.data) - 1:
            self.done = True
        obs = self._get_observation()
        return obs[0], obs[1], reward, self.done

class TradingAgent:
    def __init__(self, state_size, features_size, action_size, memory_size, discount_factor, exploration_max, exploration_min,
                 exploration_decay, model_name, model_index, ticker, models_instance):
        self.state_size = state_size
        self.features_size = features_size
        self.action_size = action_size  # Now set to 3 for Hold/Buy/Sell
        self.memory = deque(maxlen=memory_size)
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_max
        self.exploration_min = exploration_min
        self.exploration_decay = exploration_decay
        self.models_instance = models_instance
        self.model_index = model_index
        self.ticker = ticker

        # Initialize models
        model_params = {
            "model_name": model_name,
            "model_index": self.model_index,
            "ticker": self.ticker,
            "input_shape": (state_size, features_size),
            "output_shape": action_size
        }
        self.model, self.target_model, self.model_file = models_instance.get_model(model_params)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.exploration_rate:
            return np.random.choice(self.action_size)
        q_values = self.model.predict(state, verbose=0)
        return np.argmax(q_values[0])

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        # print("AA")
        batch = list(islice(self.memory, len(self.memory) - batch_size, len(self.memory)))
        # print(batch)
        for state, action, reward, next_state, done in batch:
            try:
                target = reward
                if not done:
                    target = reward + self.discount_factor * np.amax(self.target_model.predict(next_state, verbose=0))
                target_f = self.model.predict(state, verbose=0)
                target_f[0][action] = target
                self.model.fit(state, target_f, epochs=1, verbose=0)
            except Exception as ex:
                print("Error 20-20", ex)
        try:
            # Update the target model after the batch
            self.target_model.set_weights(self.model.get_weights())

            # Save the model to a file
            # self.models_instance.save_model(self.model, self.model_file)

            if self.exploration_rate > self.exploration_min:
                self.exploration_rate *= self.exploration_decay
            # print("End relay")
        except Exception as ex:
            print("Error 20-30", ex)

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def save(self):
        self.models_instance.save_model(self.model, self.model_file)


class StockTradingRRL:
    def __init__(self, ticker, balance, models_files_path, model_index=1,
                 window_size=10, batch_size = 32, memory_size = 1000,
                 discount_factor = 0.95, exploration_max = 1.0, exploration_min = 0.01, exploration_decay = 0.995,
                 epochs = 10, train_data = None, test_data = None
                ):

        self.WINDOW_SIZE = window_size
        self.BATCH_SIZE = batch_size
        self.MEMORY_SIZE = memory_size
        self.DISCOUNT_FACTOR = discount_factor
        self.EXPLORATION_MAX = exploration_max
        self.EXPLORATION_MIN = exploration_min
        self.EXPLORATION_DECAY = exploration_decay
        self.EPOCHS = epochs
        #
        self.train_data = train_data
        self.test_data = test_data
        #
        self.ticker = ticker
        self.models_files_path = models_files_path
        self.model_index = model_index

        # Initialize classes here
        self.position = Position(balance=balance)
        self.models_instance = Models(self.models_files_path)

        self.env = None
        self.agent = None

        # print(self.train_data.head(20))

    def initialize_environment_and_agent(self, data):
        self.env = TradingEnv(data, self.position, self.WINDOW_SIZE)
        self.agent = TradingAgent(self.WINDOW_SIZE, data.shape[1], 3, memory_size=self.MEMORY_SIZE,
                                  discount_factor=self.DISCOUNT_FACTOR, exploration_max=self.EXPLORATION_MAX,
                                  exploration_min=self.EXPLORATION_MIN, exploration_decay=self.EXPLORATION_DECAY,
                                  model_name="cnn_lstm", model_index=self.model_index, ticker=self.ticker,
                                  models_instance=self.models_instance)

    def train(self):
        print("AAA train")
        self.initialize_environment_and_agent(self.train_data)
        for epoch in range(self.EPOCHS):
            state, date = self.env.reset()
            # print("A State\n", state)
            done = False
            n_save = 10
            while not done:
                try:
                    action = self.agent.act(state)
                    # print("A action: ", action)
                    next_state, date, reward, done = self.env.step(action)
                    print(date)
                    self.agent.remember(state, action, reward, next_state, done)
                    state = next_state
                    if done:
                        print(f"Epoch {epoch + 1}/{self.EPOCHS} - Final Net Worth: {self.env.position.net_worth}")
                        break
                except Exception as ex:
                    print("Error 20-10-5", ex)
                try:
                    self.agent.replay(self.BATCH_SIZE)
                except Exception as ex:
                    print("Error 20-10-10", ex)
                n_save += 1
                if n_save % 10 == 0:
                    self.agent.save()
                    n_save = 0
            self.agent.save()

    def test(self):
        self.initialize_environment_and_agent(self.test_data)
        state = self.env.reset()
        done = False
        total_reward = 0
        dates = []
        share_holdings = []
        net_worth_list = []
        balance_list = []
        while not done:
            action = self.agent.act(state)  # No exploration, use the learned policy
            next_state, date, reward, done = self.env.step(action)
            print(date)
            self.agent.remember(state, action, reward, next_state, done)
            total_reward += reward
            dates.append(str(date))
            share_holdings.append(round(self.position.holdings))
            net_worth_list.append(round(100*self.position.net_worth)/100)
            balance_list.append(round(100*self.position.balance)/100)
            state = next_state
            if done:
                print(f"Test Finished - Final Net Worth: {self.env.position.net_worth}, Total Reward: {total_reward}")
            self.agent.replay(self.BATCH_SIZE)
        self.agent.save()
        return total_reward, dates, share_holdings, net_worth_list, balance_list


class RRLCNNSLTMAlgo(object):
    def __init__(self, dic):
        # print("90567-8-000 Algo\n", dic, '\n', '-'*50)
        try:
            super(RRLCNNSLTMAlgo, self).__init__()
        except Exception as ex:
            print("Error 9057-010 Algo:\n"+str(ex), "\n", '-'*50)
        # print("MLAlgo\n", self.app)
        # print("90004-020 Algo\n", dic, '\n', '-'*50)
        self.app = dic["app"]

# https://chatgpt.com/c/66e0947c-6714-800c-9probabilitiesef3-3aa45026ed5a
class RRLCNNSLTMDataProcessing(BaseDataProcessing, BasePotentialAlgo, RRLCNNSLTMAlgo):
    def __init__(self, dic):
        print("90567-010 DataProcessing\n", dic, '\n', '-' * 50)
        super().__init__(dic)
        # print("9005 DataProcessing ", self.app)
        self.MODELS_PATH = os.path.join(self.TO_OTHER, "models")
        os.makedirs(self.MODELS_PATH, exist_ok=True)
        # -----
        self.SCALER_PATH = os.path.join(self.TO_OTHER, "scalers")
        os.makedirs(self.SCALER_PATH, exist_ok=True)

    def train(self, dic):
        print("\n90199-RLL train: \n", "="*50, "\n", dic, "\n", "="*50)
        # Load stock data using Yahoo Finance
        ticker = dic["ticker"]
        print("ticker", ticker)
        epoches = int(dic["epoches"])
        balance = 100000
        # ---------------
        dp = DataProcessing(ticker, scalars_files_path=self.SCALER_PATH,
                            days_of_investment=int(dic["days_of_investment"]),
                            years_of_data = float(dic["years_of_data"]))
        train_data, scaler = dp.load_train_data()
        # print(train_data)
        # test_data, scaler = dp.load_test_data()
        # print(test_data)

        st = StockTradingRRL(ticker, balance, self.MODELS_PATH, model_index=1,
                 window_size=10, batch_size = 32, memory_size = 1000,
                 discount_factor = 0.95, exploration_max = 1.0, exploration_min = 0.01, exploration_decay = 0.995,
                 epochs = epoches, train_data = train_data, test_data=None)

        st.train()

        print("End train")

        result = {"status": "ok"}
        return result

    def test(self, dic):
        print("90222-RLL: \n", "="*50, "\n", dic, "\n", "="*50)
        ticker = dic["ticker"]
        ticker = dic["ticker"]
        print("ticker", ticker)
        epoches = int(dic["epoches"])
        balance = 1000
        # ---------------
        dp = DataProcessing(ticker, scalars_files_path=self.SCALER_PATH,
                            days_of_investment=int(dic["days_of_investment"]),
                            years_of_data = float(dic["years_of_data"]))
        test_data, scaler = dp.load_test_data()
        # print(test_data)

        st = StockTradingRRL(ticker, balance, self.MODELS_PATH, model_index=1,
                 window_size=10, batch_size = 32, memory_size = 1000,
                 discount_factor = 0.95, exploration_max = 1.0, exploration_min = 0.01, exploration_decay = 0.995,
                 epochs = epoches, train_data = None, test_data=test_data)

        total_reward, dates, share_holdings, net_worth_list, balance_list = st.test()

        print("End test\n", total_reward, dates, share_holdings, net_worth_list, balance_list)

        result = {"status": "ok", "total_reward": total_reward}
        return result

