from unittest.mock import inplace

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import logging
# ---
from datetime import datetime, timedelta
import math
import json
from tensorboard.plugins.image.summary import image
from tensorflow.keras.models import Sequential, load_model
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
from matplotlib.dates import (YEARLY, DateFormatter,
                              YearLocator, MonthLocator, DayLocator)
# -----
from ..basic_ml_objects import BaseDataProcessing, BasePotentialAlgo, AbstractModels
from ....core.utils import Debug, log_debug, clear_log_debug
# -----
import yfinance as yf
# -----
from abc import ABC, abstractmethod

class History(object):
   def __init__(self):
    self.history = {}
    self.epoch = None

class SP500(AbstractModels, ABC):
    def __init__(self, dic) -> None:
        super(SP500, self).__init__(dic)
        log_debug("in obj init 12")
        print("B SP500\n", dic)
        self.dic = dic
        self.trainXData = None
        self.testXData = None
        self.trainData = None
        self.testData = None
        # ---
        self.trainTestSplit=0.7
        nunit=15
        ntimestep=10
        # ---
        self.nUnit = nunit
        self.nTimestep = ntimestep
        self.batchSize = int(dic["batchsize"])
        self.nEpoch = int(dic["epochs"])
        # ---
        self.dateCol = "Date"
        self.priceCol = "Adj Close"
        self.volCol = "Volume"
        self.volatilityCol = "VolatRatio"
        self.volumeCol = "VolumeRatio"
        self.momentumCol = "Momentum"
        self.returnCol = "Last1MoReturn"
        self.resultCol = "Fwd1MoReturn"
        self.daysInMonth = 21
        self.cellType = None
        self.featureCols = [self.returnCol, self.momentumCol, self.volatilityCol, self.volumeCol]
        # ---
        log_debug("in obj init 13")
        is_get_data = True
        try:
            is_get_data = dic["is_get_data"]
        except Exception as ex:
            pass
        if is_get_data:
            self.get_data()
        log_debug("in obj init 14")
        # ---
        self.classes = ["cup_handle_event", "not_cup_handle_event"]
        self.nClass = len(self.classes)
        # ---
        log_debug("in obj int 14")
        self.batchSize = int(dic["batchsize"])
        self.nEpoch = int(dic["epochs"])
        # ---
        self.loss = None
        self.optimizer = None
        self.metric = None
        # ---
        # log_debug("in obj int 15")
        self.get_model()
        # log_debug("in obj int 16")
        # ---

    def feature_engineer(self, df: pd.DataFrame) -> pd.DataFrame:

        df.loc[:, self.dateCol] = pd.to_datetime(df.loc[:, self.dateCol])
        # 1 Month lagged returns
        returns = np.zeros(df.shape[0], dtype=np.float32)
        nrow = df.shape[0]
        # print("self.daysInMonth\n", self.daysInMonth)
        returns[self.daysInMonth+1:] = np.divide(df.loc[self.daysInMonth:nrow-2, self.priceCol].values,
                                                 df.loc[0:nrow-self.daysInMonth-2, self.priceCol].values) - 1

        # print("\nK1\n", df.loc[self.daysInMonth:nrow-2, self.priceCol])
        # print("\nK2\n", df.loc[0:nrow-self.daysInMonth-2, self.priceCol])
        #
        # print("\nreturns\n", returns)
        #
        # print(returns.shape)

        df.loc[:, self.returnCol] = returns

        # print("BB1\n", df.tail(40))
        # print("BB2\n", df)

        # momentum factor
        momentum = np.zeros(df.shape[0], dtype=np.float32)
        returns3Mo = np.divide(df.loc[3*self.daysInMonth:nrow-2, self.priceCol].values,
                               df.loc[0:nrow-3*self.daysInMonth-2, self.priceCol].values) - 1
        num = returns[3*self.daysInMonth+1:]
        momentum[3*self.daysInMonth+1:] = np.divide(num, np.abs(num) + np.abs(returns3Mo))
        df.loc[:, self.momentumCol] = momentum
        # print("BB3\n", df)

        # volatility factor
        df.loc[:, self.volatilityCol] = 0
        volatility = np.zeros(nrow, dtype=np.float32)

        rtns = returns[self.daysInMonth+1:2*self.daysInMonth+1]
        sumval = np.sum(rtns)
        sumsq = np.sum(rtns * rtns)

        for i in range(2*self.daysInMonth+1, nrow):
            mean = sumval / self.daysInMonth
            volatility[i] = np.sqrt(sumsq / self.daysInMonth - mean*mean)
            sumval += returns[i] - returns[i-self.daysInMonth]
            sumsq += returns[i] * returns[i] - returns[i-self.daysInMonth] * returns[i-self.daysInMonth]

        oneyr = 12 * self.daysInMonth
        df.loc[:, self.volatilityCol] = 0.0
        for i in range(oneyr+2*self.daysInMonth+1, nrow):
            df.loc[i, self.volatilityCol] = volatility[i] / np.mean(volatility[i-oneyr:i])

        # volume factor
        df.loc[:, self.volumeCol] = 0
        volume = df.loc[:, self.volCol].values
        for i in range(self.daysInMonth, nrow-1):
            df.loc[i+1, self.volumeCol] = volume[i] / np.mean(volume[i-self.daysInMonth:i])

        # result column
        df.loc[:, self.resultCol] = 0.0
        df.loc[0:nrow-self.daysInMonth-1, self.resultCol] = df.loc[self.daysInMonth:, self.returnCol].values

        # print("BB91\n", df.head(50))
        # print("BB92\n", df.tail(50))
        # print("BB93\n", df)

        return df

    def get_data(self, **data):

        years = 10
        end_date = datetime.now()
        start_date = end_date - timedelta(days=math.ceil(365*years))
        # start_date = datetime.strptime('20000103', '%Y%m%d')
        # end_date = datetime.strptime('20220722', '%Y%m%d')
        # ===
        df = yf.download(self.dic["tickers"], start=start_date, end=end_date)
        df.reset_index(inplace=True)
        print("B\n", df)
        df = self.feature_engineer(df)
        # print("B1\n", df)
        # print("B\n", df.columns)

        ntrain = int(self.trainTestSplit * df.shape[0])

        self.trainXData = df.loc[14*self.daysInMonth+1:ntrain, :].reset_index(drop=True)
        self.testXData = df.loc[ntrain:, :].reset_index(drop=True)

        self.trainData = self.prepare_data_for_rnn(self.trainXData)
        self.testData = self.prepare_data_for_rnn(self.testXData)

    def prepare_data_for_rnn(self, df):
        nfeat = len(self.featureCols)
        data = np.zeros((df.shape[0]-self.nTimestep, self.nTimestep, nfeat), dtype=np.float32)
        results = np.zeros((df.shape[0]-self.nTimestep, self.nTimestep), dtype=np.float32)
        raw_data = df[self.featureCols].values
        raw_results = df.loc[:, self.resultCol].values

        for i in range(0, data.shape[0]):
            data[i, :, :] = raw_data[i:i+self.nTimestep, :]
            results[i, :] = raw_results[i:i+self.nTimestep]

        return data, results

    def normalize_data(self, **data):
        trainx = data["trainx"]
        testx = data["testx"]
        return trainx, testx

    def create_rnn_model(self):
        nfeat = len(self.featureCols)
        self.cellType = "LSTM"
        # ---
        self.model = Sequential()
        self.model.add(layers.LSTM(self.nUnit, input_shape=(None, nfeat)))
        #self.model.add(layers.GRU(self.nUnit, input_shape=(None, nfeat)))
        #self.model.add(layers.SimpleRNN(self.nUnit, input_shape=(None, nfeat)))
        self.model.add(layers.Dense(5, activation="relu"))
        self.model.add(layers.Dense(1))
        # ---
        self.model.summary()
        # ---
        self.loss = tf.keras.losses.MeanSquaredError()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.005)
        self.metric = tf.keras.metrics.MeanSquaredError()
        self.model.compile(optimizer=self.optimizer, loss=self.loss, metrics=[self.metric])
        # ---
        self.checkpoint_model()
        # ---

    def get_plots(self):
        r = {"train":{}, "test":{}}
        for d in ["train", "test"]:
            df=eval("self."+d+"XData")
            df[self.dateCol] = df[self.dateCol].dt.strftime('%Y-%m-%d')
            df = df.set_index(keys=[self.dateCol])
            r[d]["x"] = df.index.values.tolist()

            l = df.loc[:, self.priceCol].values.tolist()
            l0 = l[0]
            l = [round(n/l0, 3) for n in l]
            r[d][self.dic["tickers"]] = l

            for i, col in enumerate(self.featureCols):
                # print("="*100, "\n1111111 ", d, col, "\n", "="*100)
                l = df.loc[:, col].values.tolist()
                l = [round(n, 3) for n in l]
                # print(l)
                r[d][col]= l
        return r

    def getConvergenceHistory(self, history, metricName):
        # print(metricName)
        # print(history.epoch, history.history[metricName])
        return {"x": history.epoch, "y": history.history[metricName]}

    def test(self):
        mse = tf.keras.losses.MeanSquaredError()
        dic = {}
        n = 0
        ds = ["train", "test"]
        for X, y in [self.trainData, self.testData]:
            dic[ds[n]] = {}
            predict = self.model.predict(X)

            print(ds[n], "X\n", X)
            print(ds[n], "y\n", y)
            print(ds[n], "predict\n", predict)

            print(ds[n], "y[:, -1]\n", y[:, -1])
            print(ds[n], "predict[:, 0]\n", predict[:, 0])

            dic[ds[n]]["final_loss"] = round(1000*float(mse(y[:, -1], predict[:, 0]).numpy()))/1000
            # baseline model prediction that uses last month's return as prediction for 1 month return
            dic[ds[n]]["baseline_loss"] = round(1000*float(mse(y[:, -1], X[:, -1, 0]).numpy()))/1000
            # plot predicted vs actual vs baseline
            y = [round(x, 3) for x in np.round(y[:, -1], decimals=3).tolist()]
            p = [round(x, 3) for x in np.round(predict[:, 0], decimals=3).tolist()]

            d = pd.DataFrame({"y":y, "p":p})
            d = d.sort_values("y")
            dic[ds[n]]["y"] = d["y"].tolist()
            dic[ds[n]]["predict"] = d["p"].tolist()

            n += 1
        # print(dic)
        return dic

    def train(self):
        dic = {}
        try:
            log_debug("before model.fit")
            history = self.model.fit(self.trainData[0], self.trainData[1],
                                     batch_size=self.batchSize, epochs=self.nEpoch)
            log_debug("after model.fit")
            self.save()
            log_debug("after save")
            # ---
            log_debug("before getConvergenceHistory 1")
            dic[self.metric._name] = self.getConvergenceHistory(history, self.metric._name)
            log_debug("before getConvergenceHistory 2")
            dic["loss"] = self.getConvergenceHistory(history, "loss")
            log_debug("after getConvergenceHistory 1")
        except Exception as ex:
            print("Error 22-22-3", ex)
            log_debug("Error 22-22-3: " + str(ex))
        try:
            self.model.evaluate(self.testData[0], self.testData[1], verbose=2)
            print("C66 End train")
        except Exception as ex:
            print("Error 22-22-5", ex)
            log_debug("Error 22-22-5: " + str(ex))
        return dic


# =========== AcademyCity Object ===========
# Based on the book: Reinforcement Learning for Finance chapter 4.5
class SP500Algo(object):
    def __init__(self, dic):
        # print("90888-8-000 Algo\n", dic, '\n', '-'*50)
        try:
            super(SP500Algo, self).__init__()
        except Exception as ex:
            print("Error 90888-010 Algo:\n" + str(ex), "\n", '-' * 50)
        self.app = dic["app"]

class SP500DataProcessing(BaseDataProcessing, BasePotentialAlgo, SP500Algo):
    def __init__(self, dic):
        # print("908889-010 CHDataProcessing\n", dic, '\n', '-' * 50)
        super().__init__(dic)
        # # print("9005 DataProcessing ", self.app)
        # self.MODELS_PATH = os.path.join(self.TO_OTHER, "models")
        # os.makedirs(self.MODELS_PATH, exist_ok=True)
        # # print(self.MODELS_PATH)
        # # -----
        # self.SCALER_PATH = os.path.join(self.TO_OTHER, "scalers")
        # os.makedirs(self.SCALER_PATH, exist_ok=True)
        # -----

    def upload_train_file(self, dic):
        print("90974-1-3: \n", dic, "\n", "="*50)

        file_path = self.upload_file(dic)["file_path"]
        # print('file_path', file_path)

        result = {"status": "ok"}
        # print(result)
        return result

    def train(self, dic):
        print("\n90466-CH train: \n", "=" * 50, "\n", dic, "\n", "=" * 50)
        clear_log_debug()
        model_name = "rnn"  # dic["model_name"]
        epochs = 40  # int(dic["epochs"])
        batch_size = 10  # int(dic["batch_size"])
        tickers = "SPY"
        # ---------------
        dic = {"model_dir": self.MODELS_PATH, "model_name": model_name,
               "batchsize": batch_size, "epochs": epochs,
               "tickers": tickers,
               "output_dir": self.TO_EXCEL_OUTPUT}
        log_debug("before creating obj CupHandle")
        sp500_obj = SP500(dic)
        print("Z15")
        charts = sp500_obj.train()
        # print("Z13")
        for k in charts:
            charts[k]["y"] = [round(100*x)/100 for x in charts[k]["y"]]
        print("charts\n", charts)

        result = {"status": "ok"}
        return result

    def get_plots(self, dic):
        print("\n90999-SP500 get_plots: \n", "=" * 50, "\n", dic, "\n", "=" * 50)

        clear_log_debug()
        model_name = "rnn"  # dic["model_name"]
        epochs = 40  # int(dic["epochs"])
        batch_size = 10  # int(dic["batch_size"])
        tickers = "SPY"
        # ---------------
        dic = {"model_dir": self.MODELS_PATH, "model_name": model_name,
               "batchsize": batch_size, "epochs": epochs,
               "tickers": tickers,
               "output_dir": self.TO_EXCEL_OUTPUT}
        log_debug("before creating obj CupHandle")
        sp500_obj = SP500(dic)
        print("Z12")
        plots = sp500_obj.get_plots()
        print(plots)

        # # print("Z13")
        # for k in charts:
        #     charts[k]["y"] = [round(100*x)/100 for x in charts[k]["y"]]
        # print("charts\n", charts)

        result = {"status": "ok", "plots": plots}
        return result

    def test(self, dic):
        print("90499-CH: \n", "=" * 50, "\n", dic, "\n", "=" * 50)
        model_name = "rnn"  # dic["model_name"]
        epochs = 40  # int(dic["epochs"])
        batch_size = 10  # int(dic["batch_size"])
        tickers = "SPY"
        # ---------------
        dic = {"model_dir": self.MODELS_PATH, "model_name": model_name,
               "batchsize": batch_size, "epochs": epochs,
               "tickers": tickers,
               "output_dir": self.TO_EXCEL_OUTPUT}
        log_debug("before creating obj CupHandle")
        sp500_obj = SP500(dic)
        test_results = sp500_obj.test()
        # print("Z35")
        # print(test_results)
        result = {"status": "ok", "test_results": test_results}
        # print(result)
        return result

