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
from tensorflow.keras.layers import Dense, LSTM, Conv2D, Dropout, Flatten, Reshape, MaxPooling2D, LeakyReLU
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from sklearn.utils import class_weight
# -----
from ..basic_ml_objects import BaseDataProcessing, BasePotentialAlgo
# -----
from datetime import datetime, timedelta
import pickle


class SPAlgo(object):
    def __init__(self, dic):
        # print("90567-8-000 Algo\n", dic, '\n', '-'*50)
        try:
            super(SPAlgo, self).__init__()
        except Exception as ex:
            print("Error 9057-010 Algo:\n"+str(ex), "\n", '-'*50)
        # print("MLAlgo\n", self.app)
        # print("90004-020 Algo\n", dic, '\n', '-'*50)
        self.app = dic["app"]


class SPDataProcessing(BaseDataProcessing, BasePotentialAlgo, SPAlgo):
    def __init__(self, dic):
        print("90567-02  DataProcessing\n", dic, '\n', '-' * 50)
        super().__init__(dic)
        # print("9005 DataProcessing ", self.app)
        self.MODELS_PATH = os.path.join(self.TO_OTHER, "models")
        os.makedirs(self.MODELS_PATH, exist_ok=True)
        # -----
        self.SCALER_PATH = os.path.join(self.TO_OTHER, "scalers")
        os.makedirs(self.SCALER_PATH, exist_ok=True)

        self.now_date = None
        self.start_train_date = None
        self.end_train_date = None
        self.start_test_date = None
        self.end_test_date = None
        # ------------------


    def train(self, dic):
        print("\n90199-RLL train: \n", "="*50, "\n", dic, "\n", "="*50)
        # Load stock data using Yahoo Finance
        ticker = dic["ticker"]
        print("ticker", ticker)

        result = {"status": "ok"}
        return result

    def test(self, dic):
        print("90333-SP: \n", "="*50, "\n", dic, "\n", "="*50)
        ticker = dic["ticker"]
        print("ticker", ticker)

        shock_size = 15
        days_of_investment=0
        years_of_data=5
        test_size_ = 0.1
        epochs_ = 100
        dropout_ = 0.15
        time_step = 60  # Number of previous days to use for prediction

        def create_model(input_shape, output_shape):
            model = Sequential()
            model.add(LSTM(units=200, return_sequences=True, input_shape=input_shape))
            model.add(Dropout(dropout_))
            model.add(LSTM(units=200, return_sequences=True))
            model.add(Dropout(dropout_))
            model.add(LSTM(units=150, return_sequences=False))
            model.add(Dropout(dropout_))
            model.add(Dense(units=100, kernel_regularizer=l2(0.01)))  # L  Regularization
            model.add(LeakyReLU())
            model.add(Dense(output_shape, activation='softmax'))

            # Compile the model with Adam optimizer and categorical crossentropy loss
            opt = Adam(learning_rate=0.0001, clipvalue=1.0)  # Clip gradient to avoid exploding gradients
            # model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
            model.compile(optimizer=opt, loss='categorical_hinge', metrics=['accuracy'])
            return model

        features = ['Open', 'High', 'Low', 'Close', 'Volume']

        self.now_date = datetime.now()
        self.start_train_date = self.now_date - timedelta(days=math.ceil(years_of_data * 365))
        self.end_train_date = self.now_date - timedelta(days=days_of_investment - 1)
        self.start_test_date = self.end_train_date
        self.end_test_date = self.now_date

        data = yf.download(ticker, start=self.start_train_date, end=self.end_train_date)
        data = data[['Open', 'High', 'Low', 'Close', 'Volume']]

        data["sHigh"] = data["High"].shift(-1)
        data["sLow"] = data["Low"].shift(-1)
        data["D"] = data["sLow"] - data["Close"]
        data["U"] = data["sHigh"] - data["Close"]
        data.dropna(inplace=True)

        data['Target'] = data.apply(lambda row: (1 if row['U'] >= shock_size else (-1 if row['D'] <= -shock_size else 0))
        if row['U'] >= -row['D'] else (-1 if row['D'] <= -shock_size else (1 if row['U'] >= shock_size else 0)), axis=1)
        # --
        # Check the class distribution in your target variable
        print(data['Target'].value_counts())
        # ---------------------
        x_train, x_test, y_train, y_test = train_test_split(data[['Open', 'High', 'Low', 'Close', 'Volume']], data[['Target']], test_size=test_size_, shuffle=False)
        # print(x_train, x_test, y_train, y_test)

        scaler = MinMaxScaler(feature_range=(0, 1))
        x_train_ = scaler.fit_transform(x_train)
        x_test_ = scaler.transform(x_test)
        # print(x_train_, "\nx_test\n", x_test_)

        # print(scaled_data_)
        y_train_ = np.array(y_train)
        y_test_ = np.array(y_test)
        # print(y_train_, "\ny_test\n", y_test_)
        # ---------------------

        # Prepare the data for LSTM
        def create_dataset(data, target, n_steps):
            X, y = [], []
            for i in range(n_steps, len(data)):
                X.append(data[i - n_steps:i])
                y.append(target[i])
            return np.array(X), np.array(y)

        X_train, y_train = create_dataset(x_train_, y_train_, time_step)
        # print("X\n", X_train, "\ny\n", y_train)
        X_test, y_test = create_dataset(x_test_, y_test_, time_step)
        # print("X\n", X_test, "y\n", y_test)

        # Reshape X_train to (samples, time_steps, number_of_features)
        X_train = np.reshape(X_train,(X_train.shape[0], X_train.shape[1], len(features)))
        X_test = np.reshape(X_test,(X_test.shape[0], X_test.shape[1], len(features)))

        y_train = to_categorical(y_train + 1, num_classes=3)  # Adding 1 to shift -1,0,1 to 0,1,2
        y_test = to_categorical(y_test + 1, num_classes=3)

        # print("X_train\n", X_train, "\nX_test\n", X_test)

        model = create_model(input_shape=(X_train.shape[1], X_train.shape[2]), output_shape=3)
        # --------
        y_train_labels = np.argmax(y_train, axis=1)  # Converts one-hot encoded back to single class labels
        # Compute class weights based on the raw class labels
        class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train_labels),
                                                          y=y_train_labels)
        # Create a dictionary to pass to model.fit
        class_weight_dict = dict(enumerate(class_weights))
        # --------
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        # Train the model
        history = model.fit(X_train, y_train,
                            epochs=epochs_, batch_size=32,
                            validation_data=(X_test, y_test),
                            class_weight=class_weight_dict,callbacks=[early_stop])

        # Evaluate the model
        loss, accuracy = model.evaluate(X_test, y_test)
        print(f'\nTest Accuracy: {accuracy * 100:.2f}%')

        # Make predictions
        predictions = model.predict(X_test)
        # Convert predictions to the class labels (-1, 0, 1)
        predicted_classes = np.argmax(predictions, axis=1) - 1  # Subtract 1 to map back to -1, 0, 1
        print("\npredicted_classes\n", predicted_classes)


        # d = pd.DataFrame({"y": y.flatten(), "p": predictions.flatten()})
        # print(d)
        # c = d.corr()
        # print(c)

        result = {"status": "ok"}
        return result

