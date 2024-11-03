import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import math
import json
import tensorflow as tf
import matplotlib.pyplot as plt
import os
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

class CupHandle(AbstractModels, ABC):
    def __init__(self, dic) -> None:
        # print("A FashionMNistClassify\n", dic)
        # log_debug("in obj init of CupHandle 1")
        super(CupHandle, self).__init__(dic)
        log_debug("in obj init of CupHandle 12")
        # print("B CupHandle\n", dic)
        self.dic = dic
        self.trainData = None
        self.testData = None

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
        # print(self.trainData[0][0])
        # ---
        self.classes = ["cup_handle_event", "not_cup_handle_event"]
        self.nClass = len(self.classes)
        # ---
        log_debug("in obj int of FashionMNistClassify 14")
        self.batchSize = int(dic["batchsize"])
        self.nEpoch = int(dic["epochs"])
        # ---
        self.loss = None
        self.optimizer = None
        self.metric = None
        # ---
        log_debug("in obj int of FashionMNistClassify 15")
        self.get_model()
        log_debug("in obj int of FashionMNistClassify 16")
        # ---

    def get_data(self, **data):
        df = pd.read_csv(os.path.join(self.dic["input_dir"], "train.csv"))
        # print("A\n", df, "\n", df.shape)
        df.drop(columns=["Day"], inplace=True)
        # print("B\n", df.columns)
        data = np.transpose(df.reset_index(drop=True).values)
        # print("train B1")
        y_actual = np.array([int(c.startswith("t")) for c in df.columns], dtype=np.int)
        # print(y_actual)
        # -------
        # Split data and label array into train and test sets
        trainx, testx, trainy, testy = train_test_split(
            data, y_actual, test_size=0.05, random_state=42, stratify=y_actual  # Stratify for label balance if needed
        )
        # ------
        trainx, testx = self.normalize_data(trainx=trainx, testx=testx)
        # ---
        self.trainData = (trainx, trainy)
        self.testData = (testx, testy)

    def normalize_data(self, **data):
        trainx = data["trainx"]
        testx = data["testx"]
        def get_image_data(data_to_convert):
            data_final = np.zeros((data_to_convert.shape[0], 20, 20, 1), dtype=np.float32)
            for i in range(data_to_convert.shape[0]):
                for j in range(20):
                    pixel = int(data_to_convert[i, j] * 20)
                    if pixel == 20:
                        pixel = 19
                    data_final[i, j, pixel, 0] = 1
            return data_final
        trainx = get_image_data(trainx)
        testx = get_image_data(testx)
        return trainx, testx

    def create_cnn_model(self):
        self.model = Sequential()
        self.model.add(layers.Conv2D(10, (3, 3), activation="relu", input_shape=(20, 20, 1)))
        self.model.add(layers.AveragePooling2D(pool_size=(2, 2)))
        self.model.add(layers.Conv2D(5, (4, 4), activation="relu"))
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(2, activation="relu"))
        self.model.add(layers.Dense(2))
        # ---
        self.model.summary()
        # ---
        self.loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.002)
        self.metric = tf.keras.metrics.SparseCategoricalAccuracy()
        self.model.compile(optimizer=self.optimizer, loss=self.loss, metrics=[self.metric])
        # ---
        self.checkpoint_model()
        # ---

    def getConfusionMatrix(self, labels: np.ndarray,predictions: np.ndarray):
        predictedLabels = np.argmax(predictions, axis=1)
        # fig, ax = plt.subplots()
        cm = np.zeros((self.nClass, self.nClass),dtype=np.int32)
        for i in range(labels.shape[0]):
            cm[labels[i], predictedLabels[i]] += 1
        return cm

    def getConvergenceHistory(self, history, metricName):
        # print(metricName)
        # print(history.epoch, history.history[metricName])
        return {"x": history.epoch, "y": history.history[metricName]}

    def test(self):
        dic = {}
        n = 0
        ds = ["train", "test"]
        for X, y in [self.trainData, self.testData]:
            log_debug("before model.predict")
            predictClass = self.model.predict(X)
            log_debug("after model.predict")
            cm = self.getConfusionMatrix(y, predictClass)
            log_debug("after getConfusionMatrix")
            dic[ds[n]] = cm
            n +=1
        return dic

    def train(self):
        dic = {}
        try:
            log_debug("before model.fit")
            history = self.model.fit(self.trainData[0], self.trainData[1],
                                    epochs = self.nEpoch)
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
            # ---
            # predictions = self.model(self.trainData[0][:1]).numpy()
            # print(predictions)
            # ---
            # this is a probabilistic model, add a softmax layer at the end
            self.model = tf.keras.Sequential([self.model, tf.keras.layers.Softmax()])
        except Exception as ex:
            print("Error 22-22-5", ex)
            log_debug("Error 22-22-5: " + str(ex))
        return dic

    def predict(self):
        print("predict")

        tickers = self.dic["tickers"]
        if tickers[0] == "ALL":
            url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
            # Read the tables from the Wikipedia page
            tables = pd.read_html(url)
            # The first table on the page contains the S&P 500 tickers
            sp500_table = tables[0]
            # Get the 'Symbol' column, which contains the tickers
            tickers = sp500_table['Symbol'].tolist()
            # print(tickers)

        output_dir = self.dic["output_dir"]
        days_of_data = int(self.dic["days_of_data"])
        # ---
        period_begin = 40
        period_end = 70
        now_date = datetime.now()
        start_train_date = now_date - timedelta(days=math.ceil(days_of_data))
        # ---
        res_df = pd.DataFrame(data={"ticker": [], "days": [], "Begin": [], "End": []})
        for ticker in tickers:
            # -----------
            if ticker in ["BRK.B", "BF.B"]:
                continue
            print(ticker)
            # -----------
            df_ticker = yf.download(ticker, start=start_train_date, end=now_date)
            df_ticker.reset_index(inplace=True)
            # print(df_ticker)
            # -----------
            for period in range(period_begin, period_end):
                # print(ticker, period)
                res_df = self.identify(self.model, df_ticker, period, ticker, res_df)
        res_df = res_df.sort_values(by='Begin')
        # output_file = os.path.join(output_dir, ticker+"_output.csv")
        output_file = os.path.join(output_dir, "all_output.csv")
        print(res_df)
        res_df.to_csv(output_file, index=False)

    def get_plots(self):
        print("plots")
        output_dir = self.dic["output_dir"]
        # ---
        output_file = os.path.join(output_dir, "all_output.csv")
        df = pd.read_csv(output_file)
        df.loc[:, "Begin"] = pd.to_datetime(df.loc[:, "Begin"])
        df.loc[:, "End"] = pd.to_datetime(df.loc[:, "End"])
        ticker = None
        ticker_df = None
        last_ticker = ""
        plots = {}
        for rownum in range(df.shape[0]):
            ticker = df.loc[rownum, "ticker"]
            begin_date = df.loc[rownum, "Begin"]
            end_date = df.loc[rownum, "End"]

            begin_date_ = begin_date.strftime('%Y/%m/%d')
            print(begin_date, end_date)
            # ---
            if ticker != last_ticker:
                ticker_df = yf.download(ticker, start=begin_date, end=end_date)
                ticker_df.reset_index(inplace=True)
                ticker_df["Date"] = ticker_df["Date"].dt.date
                # Convert the 'Date' column to string with a custom format (e.g., 'YYYY/MM/DD')
                ticker_df['Date'] = ticker_df['Date'].apply(lambda x: x.strftime('%Y/%m/%d'))

                t = ticker+":"+begin_date_
                plots[t] = {"index":[], "date":[], "price":[]}
                for index, row in ticker_df.iterrows():
                    plots[t]["date"].append(row["Date"])
                    plots[t]["index"].append(index)
                    plots[t]["price"].append(round(float(row["Adj Close"])*100)/100)
        return plots

    def rescaleXDimension(self, ar, xsize):
        if ar.shape[0] == xsize:
            return ar

        if ar.shape[0] > xsize:
            px = ar
            px2 = np.zeros(xsize, dtype=np.float64)
            px2[0] = px[0]
            px2[-1] = px[-1]
            delta = float(ar.shape[0])/xsize
            for i in range(1, xsize-1):
                k = int(i*delta)
                fac1 = i*delta - k
                fac2 = k + 1 - i*delta
                px2[i] = fac1 * px[k+1] + fac2 * px[k]

            return px2
        raise ValueError("df rows are less than required array elements")

    def identify(self, model, df_ticker, ndays, ticker, res_df):
        px_arr = df_ticker.loc[:, "Adj Close"].values
        date_arr = df_ticker.loc[:, "Date"].values
        days_identified = set(res_df.loc[res_df.loc[:, "ticker"].eq(ticker), "Begin"])
        inp = np.zeros((1, 20, 20, 1), dtype=np.float32)
        for i in range(df_ticker.shape[0] - ndays):
            if date_arr[i] in days_identified:
                continue
            inp[:, :, :, :] = 0
            px = px_arr[i:i+ndays]
            mn = px.min()
            mx = px.max()
            transform_px = np.divide(np.subtract(px, mn), mx-mn)
            transform = self.rescaleXDimension(transform_px, 20)
            for j in range(20):
                vl = int(transform[j] * 20)
                if vl == 20:
                    vl = 19
                inp[0, j, vl, 0] = 1
            outval = model(inp).numpy()
            if outval[0, 1] >= 0.9:
                print("%s from %s - %s dates" % (ticker, date_arr[i], date_arr[i+ndays-1]))
                res_df = res_df.append({"ticker":ticker, "days":ndays, "Begin": date_arr[i], "End": date_arr[i+ndays-1]},
                                       ignore_index=True)
        return res_df


class CHAlgo(object):
    def __init__(self, dic):
        # print("90567-8-000 Algo\n", dic, '\n', '-'*50)
        try:
            super(CHAlgo, self).__init__()
        except Exception as ex:
            print("Error 9057-010 Algo:\n" + str(ex), "\n", '-' * 50)
        self.app = dic["app"]

class CHDataProcessing(BaseDataProcessing, BasePotentialAlgo, CHAlgo):
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
        # print('file_path')
        # print(file_path)
        result = {"status": "ok"}
        # print(result)
        return result

    def train(self, dic):
        print("\n90466-CH train: \n", "=" * 50, "\n", dic, "\n", "=" * 50)

        clear_log_debug()
        model_name = "cnn" # dic["model_name"]
        epochs = 5         # int(dic["epochs"])
        batch_size = 32    # int(dic["batch_size"])
        # ---------------
        # tickers = ["MMM", "AXP", "AAP", "LBA", "CAT", "CVX", "CSCO", "KO", "DOW", "XOM", "GS", "HD", "IBM",
        #                 "INTC", "JNJ", "JPM", "MCD", "MRK", "MSFT", "NKE", "PFE", "PG", "RTX", "TRV", "UNH",
        #                 "VZ", "V", "WMT", "WBA", "DIS"]
        tickers = ["TRV"] # "IBM"
        # ---------------
        dic = {"model_dir": self.MODELS_PATH, "model_name": model_name,
               "batchsize": batch_size, "epochs": epochs,
               "tickers": tickers, "input_dir": self.TO_EXCEL,
               "output_dir": self.TO_EXCEL_OUTPUT}

        log_debug("before creating obj CupHandle")
        ch_obj=CupHandle(dic)

        # not needed
        # ch_obj.get_data()

        # print("Z12")
        charts = ch_obj.train()
        # print("Z13")
        for k in charts:
            charts[k]["y"] = [round(100*x)/100 for x in charts[k]["y"]]
        print("charts\n", charts)

        # plotData(price_data_dir=self.PRICE_PATH, output_dir=self.TO_EXCEL_OUTPUT)



        result = {"status": "ok", "charts": charts}
        return result

    def test(self, dic):
        print("90499-CH: \n", "=" * 50, "\n", dic, "\n", "=" * 50)

        # not used here

        result = {"status": "ok"}
        # print(result)
        return result

    def predict(self, dic):
        print("\n90477-1-CH predic: \n", "=" * 50, "\n", dic, "\n", "=" * 50)

        clear_log_debug()
        model_name = "cnn"  # dic["model_name"]
        epochs = 5  # int(dic["epochs"])
        batch_size = 32  # int(dic["batch_size"])
        ticker = dic["ticker"].upper()
        tickers = [ticker]  # "IBM"
        days_of_data = dic["days_of_data"]
        print(tickers)
        # ---------------
        dic = {"model_dir": self.MODELS_PATH, "model_name": model_name,
               "batchsize": batch_size, "epochs": epochs,
               "days_of_data":days_of_data,
               "tickers": tickers, "input_dir": self.TO_EXCEL,
               "output_dir": self.TO_EXCEL_OUTPUT,
               "is_get_data": False}

        log_debug("before creating obj CupHandle")
        ch_obj = CupHandle(dic)
        ch_obj.predict()

        # plotData(price_data_dir=self.PRICE_PATH, output_dir=self.TO_EXCEL_OUTPUT)

        result = {"status": "ok"}
        return result

    def get_plots(self, dic):
        print("90488-CH: \n", "=" * 50, "\n", dic, "\n", "=" * 50)

        clear_log_debug()
        model_name = "cnn"  # dic["model_name"]
        epochs = 5  # int(dic["epochs"])
        batch_size = 32  # int(dic["batch_size"])
        # ---------------
        dic = {"model_dir": self.MODELS_PATH, "model_name": model_name,
               "batchsize": batch_size, "epochs": epochs,
               "days_of_data":0,
               "tickers": "", "input_dir": self.TO_EXCEL,
               "output_dir": self.TO_EXCEL_OUTPUT,
               "is_get_data": False}
        log_debug("before creating obj CupHandle")
        ch_obj = CupHandle(dic)
        plots = ch_obj.get_plots()

        print(plots)

        result = {"status": "ok", "plots": plots}
        # print(result)
        return result

