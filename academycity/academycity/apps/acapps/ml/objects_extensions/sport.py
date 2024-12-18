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


class SportAlgo(object):
    def __init__(self, dic):  # to_data_path, target_field
        # print("90567-8-000 DNQAlgo\n", dic, '\n', '-'*50)
        try:
            super(SportAlgo, self).__init__()
        except Exception as ex:
            print("Error 9057-010 SportAlgo:\n"+str(ex), "\n", '-'*50)
        # print("MLAlgo\n", self.app)
        # print("90004-020 DNQAlgo\n", dic, '\n', '-'*50)
        self.app = dic["app"]


class Sport(BaseDataProcessing, BasePotentialAlgo, SportAlgo):
    def __init__(self, dic):
        # print("90567-010 DNQDataProcessing\n", dic, '\n', '-' * 50)
        super().__init__(dic)
        # print("9005 DNQDataProcessing ", self.app)

    def train(self, dic):
        print("90155-sort train: \n", "="*50, "\n", dic, "\n", "="*50)

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

        epochs = int(dic["epochs"])
        # ---

        ret= {"my_name":"Amos"}
        result = {"status": "ok sport", "results": ret}
        return result

    def test(self, dic):
        print("90155-dqn: \n", "="*50, "\n", dic, "\n", "="*50)
        episodes = int(dic["episodes"])
        # ---

        result = {"status": "ok sport"}
        return result

