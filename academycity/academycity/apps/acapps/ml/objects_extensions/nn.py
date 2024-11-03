import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import numpy as np
from datetime import datetime
import pickle

from keras.models import Model, load_model
from keras.optimizers import Adam, RMSprop
import tensorflow as tf
from tensorflow.keras import layers, models, initializers, optimizers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
#
from ..basic_ml_objects import BaseDataProcessing, BasePotentialAlgo
from ....core.utils import log_debug, clear_log_debug
#
from ....core.utils import log_debug, clear_log_debug
#
import wbdata
#

# --------------------------

class WBAnalysis(object):
    def __init__(self, dic):
        # print("WBAnalysis\n", dic)
        try:
            self.datadir = dic['datadir']
        except Exception as ex:
            print("Error 20-01", ex, "need to provide dir name")
            self.datadir = ""
        try:
            self.model_name = dic['model_name']
        except Exception as ex:
            print("Error 20-02", ex, "need to provide model name")
            self.model_name = "General_name"
        self.checkpoint_file = os.path.join(self.datadir, "checkpoint_"+self.model_name+"_wt")

        log_debug("train_wb 110:" + self.checkpoint_file)

        # print(self.checkpoint_file)
        # ---
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        # ---
        self.model = None
        self.get_model()
        print("Obj creation - model was created")
        # ---
        self.trainData = None
        self.testData = None
        # =--
        self.history = None
        log_debug("train_wb 120:")

        print("End Obj creation")

    # --- Data ---
    def fetch_world_bank_data(self, countries, indicators):
        try:
            log_debug("train_wb 123-1: get_data.")
            # df = wbdata.get_dataframe(indicators, country=countries, date=("1980", "2024"), freq='Y')
            # Fetch data
            # countries = ['US', 'GB', 'CN']  # Example countries: United States, Great Britain, China
            df = wbdata.get_dataframe(indicators, country=countries, freq='Y')

            print("A\n", df)

            # Filter by date range (e.g., from 1980 to 2024)
            # fix error on filtering
            df = df[(df.index.get_level_values('date') > datetime(1980, 1, 1)) &
                             (df.index.get_level_values('date') <= datetime(2024, 12, 31))]

            print("B\n", df)

        except Exception as ex:
            print("Err500-50-5", ex)
            log_debug("train_wb 123-2 Error: get_data." + str(ex))

        df.reset_index(inplace=True)
        # print("AAAdf\n\n", df)
        # Handle missing data
        # df.fillna(method='ffill', inplace=True)


        df = df.dropna()

        print("\n\nBBBBB  df\n\n", df)
        log_debug("train_wb 123-5: get_data.")
        return df

    def normalize_data(self, **data):
        trainx = data["trainx"]
        trainy = data["trainy"]
        testx = data["testx"]
        testy = data["testy"]
        # scale
        trainx = self.scaler_X.fit_transform(trainx)
        trainy = self.scaler_y.fit_transform(trainy).reshape(-1)

        # Transform the test data using the fitted scaler (no fitting here)
        testx = self.scaler_X.transform(testx)
        testy = self.scaler_y.transform(testy).reshape(-1)

        print("Normalized X_train\n", trainx, "\nNormalized X_test", testx)
        print("Normalized y_train\n", trainy, "\nNormalized y_test", testy)

        return (trainx, trainy), (testx, testy)

    def get_data(self, countries, indicators, dep_var, indep_var):

        # A pull data
        df = self.fetch_world_bank_data(countries, indicators)

        log_debug("train_wb 125: data shape: " + str(df.shape))

        # Extract input features and target variable
        # print("\ndf from WB\n", df)
        X = df[indep_var].values
        y = df[dep_var].values.reshape(-1, 1)  # Reshape y for the scaler

        print("X\n", X)
        print("y\n", y)

        # B Split data into training and testing sets
        # NEED TO CHECK SPLIT from random for testing to take only last records

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        print("X_train\n", X_train, "\nX_test", X_test)
        print("y_train\n", y_train, "\ny_test", y_test)

        # C Normalize the data
        self.trainData, self.testData = self.normalize_data(trainx=X_train, trainy=y_train, testx=X_test, testy=y_test)

        log_debug("train_wb 120: data normalized.")

        # D prepare data for the model
        # print("\nAAtrain_data\n", train_data, "\nBBtest_data\n", test_data)
        self.trainData = tf.data.Dataset.from_tensor_slices((self.trainData[0], self.trainData[1]))
        self.trainData = self.trainData.batch(32).shuffle(buffer_size=1024).prefetch(tf.data.AUTOTUNE)

    # --- End Data ---

    # --- Model ---
    def save(self):
        tf.keras.models.save_model(self.model, self.checkpoint_file, overwrite=True)

    def checkpoint_model(self):
        if not os.path.exists(self.checkpoint_file):
            # self.model.predict(np.ones((20, 28, 28), dtype=np.float32))
            self.save()
        else:
            self.model = tf.keras.models.load_model(self.checkpoint_file)

    def get_model(self):
        self.model = tf.keras.models.Sequential()
        self.model.add(tf.keras.layers.Dense(1, activation='linear', input_shape=(4, )))
        self.model.compile(optimizer='adam', loss='mse')
        # ---
        self.checkpoint_model()
        # ---
    # --- End Model ---

    def get_convergence_history(self, metric_name):
        # print(metric_name)
        # print(self.history.epoch, self.history.history[metric_name])
        y = self.history.history[metric_name]
        y = [round(1000*h)/1000 for h in y]
        return {"x": self.history.epoch, "y": y}

    def train(self, dic):
        countries = dic["countries"]  # Add any countries you want to analyze
        indicators = dic["indicators"]
        dep_var = dic["dep_var"]
        indep_var = dic["indep_var"]
        epochs_ = dic["epochs"]
        # ---
        log_debug("train_wb 122: got data.")

        self.get_data(countries, indicators, dep_var, indep_var)

        log_debug("train_wb 130: got data.")

        # print(self.trainData, "\n\n", self.testData)
        # ---
        self.history = self.model.fit(self.trainData, epochs=epochs_, batch_size=32, validation_data=self.testData)

        log_debug("train_wb 140: finished fit data.")

        matrices = {}
        for k in ["loss", "val_loss"]:
            matrices[k] = self.get_convergence_history(metric_name=k)
        # ---
        log_debug("train_wb 150: finished get_convergence_history.")
        # ---
        weights, biases = self.model.layers[0].get_weights()
        weights, biases = weights.reshape(-1).tolist(), biases.reshape(-1).tolist()
        weights = [round(1000*w)/1000 for w in weights]
        biases = [round(1000*b)/1000 for b in biases]
        # ---
        predictions = self.model.predict(self.testData[0])
        # ---
        log_debug("train_wb 160: finished model.predict.")
        # ---
        p = self.scaler_y.inverse_transform(predictions).reshape(-1).tolist()
        p = [round(x) for x in p]
        a = self.scaler_y.inverse_transform(self.testData[1].reshape(1, -1)).reshape(-1).tolist()
        a = [round(x) for x in a]
        ret = {"matrices":matrices, "a":a, "p": p, "weights":weights, "biases": biases}
        # print(ret)
        return ret


class NNAlgo(object):
    def __init__(self, dic):  # to_data_path, target_field
        # print("90567-8-000 Algo\n", dic, '\n', '-'*50)
        try:
            super(NNAlgo, self).__init__()
        except Exception as ex:
            print("Error 9057-010 Algo:\n"+str(ex), "\n", '-'*50)

        self.app = dic["app"]


class NNDataProcessing(BaseDataProcessing, BasePotentialAlgo, NNAlgo):
    def __init__(self, dic):
        # print("90567-010 DataProcessing\n", dic, '\n', '-' * 50)
        super().__init__(dic)
        # print("9005 DataProcessing ", self.app)
        self.PATH = os.path.join(self.TO_OTHER, "nn")
        os.makedirs(self.PATH, exist_ok=True)
        # print(f'{self.PATH}')
        self.model = None
        self.lose_list = None
        clear_log_debug()
        #

    def load(self, file_name):
        self.model = load_model(file_name)
        log_debug("model loaded: " +str(file_name))

    def save(self, file_name):
        self.model.save(file_name)
        log_debug("model saved: " +str(file_name))

    # For Simple one independent variable.
    def train(self, dic):
        print("90155-nn: \n", "="*50, "\n", dic, "\n", "="*50)
        epochs = int(dic["epochs"])
        continue_train = int(dic["continue_train"])
        data_only = int(dic["data_only"])
        a_ = float(dic["a"])
        b_ = float(dic["b"])
        num_samples_ = int(dic["num_samples"])
        sigma_ = float(dic["sigma"])

        # Generate sample data
        def generate_data(num_samples=1500, true_a=25.0, true_b=0.7, sigma=20):
            X = np.random.uniform(0, 1, num_samples)*300
            Y = true_a + true_b * X + np.random.normal(0, sigma, num_samples)
            return X, Y

        def create_model():
            model = models.Sequential()
            model.add(layers.Dense(1, input_shape=(1,), activation='linear'
            #                        kernel_initializer=initializers.Zeros(),
            #                        bias_initializer=initializers.Zeros())
                                   ))  # One neuron, linear activation
            model.compile(optimizer='adam', loss='mean_squared_error')
            # model.compile(optimizer='RMSprop', loss='mean_squared_error')
            return model

        # Generate data
        # X_train, Y_train = generate_data(num_samples=num_samples_,true_a=a_, true_b=b_,sigma=sigma_)

        if data_only == 1:
            result = {"status": "ok nn", "data": {"x": X_train.tolist(), "y": Y_train.tolist(),
                                                  "a": 0, "b": 0}}
            return result

        model_path = f'{self.PATH}/pickles/{"nn.pkl"}'
        model_path_l = f'{self.PATH}/pickles/{"nn_lose.pkl"}'
        # print(model_path)

        if model_path and os.path.exists(model_path) and continue_train==1:
            print(f"Loading model from {model_path}")
            self.model = load_model(model_path)
            if model_path_l and os.path.exists(model_path_l):
                with open(model_path_l, 'rb') as file:
                    self.lose_list = pickle.load(file)
        else:
            print("Creating new model")
            self.model = create_model()
            self.lose_list = []

        history = self.model.fit(X_train, Y_train, epochs=epochs, verbose=1)
        self.save(model_path)

        loss_values = history.history['loss']
        for k in range(len(loss_values)):
            if (k++1) % 100 == 0:
                self.lose_list.append(loss_values[k])

        # print("self\n", self.lose_list)

        with open(model_path_l, 'wb') as file:
            pickle.dump(self.lose_list, file)

        # Extract the learned parameters (a and b)
        weights = self.model.get_weights()
        # print(weights)

        learned_b = round(100*weights[0][0][0])/100  # Slope (b)
        learned_a = round(100*weights[1][0])/100  # Intercept (a)

        # print(f"Learned parameters: a = {learned_a}, b = {learned_b}")

        result = {"status": "ok nn", "data":{"x":X_train.tolist(), "y":Y_train.tolist(),
                                             "a": learned_a, "b": learned_b,
                                             "loss_values": self.lose_list}}

        return result

    def train_wb(self, dic):

        print("9019-nnwb: \n", "="*50, "\n", dic, "\n", "="*50)
        # epochs = int(dic["epochs"])

        log_debug("train_wb 100")
        countries=[dic["country"]]
        epochs = int(dic["epochs"])

        dic["datadir"] = self.MODELS_PATH
        dic["model_name"] = "wb_analysis"

        wba = WBAnalysis(dic)
        log_debug("train_wb 200")

        dic_ = {"countries":countries,
                "indicators": {
                                 'NY.GDP.PCAP.CD': 'gdp_per_capita',
                                 'NE.EXP.GNFS.CD': 'exports_per_capita',
                                 'SE.XPD.TOTL.GD.ZS': 'education_per_capita',
                                 'BX.GSR.ROYL.CD': 'natural_resources_per_capita',
                                 'BX.KLT.DINV.WD.GD.ZS': 'high_tech_investment_per_capita'
                             },
                "dep_var": 'gdp_per_capita',
                "indep_var": ['exports_per_capita', 'education_per_capita', 'natural_resources_per_capita',
                                           'high_tech_investment_per_capita'],
                "epochs": epochs
                }

        results = wba.train(dic_)

        result = {"status": "ok wbb nn", "results": results}
        return result


    # For Multiple Independent variables - World Bank Example
    # We create a special Object to manage this example.

