import matplotlib as mpl

mpl.use('Agg')

"""
 to_data_path_ is the place datasets are kept
 topic_id name of the chapter to store images
"""

import numpy as np
from ..ml.basic_ml_objects import BaseDataProcessing, BasePotentialAlgo
from .objects_extensions.netanya_college import NDataProcessing


# --- CV ----
from .objects_extensions.cv import CVDataProcessing
# --- APPO ----
from .objects_extensions.sport import Sport
from .objects_extensions.A_1_ppo import A1PPODataProcessing
#
from .objects_extensions.rpo import RPODataProcessing
from .objects_extensions.ttdqn import TDQNDataProcessing
# from .objects_extensions.rl_dnq_ch6 import DQNDataProcessing
#
from .objects_extensions.netanya_college import NDataProcessing
from .objects_extensions.rl_dnq_cnn import DNQCNNDataProcessing
from .objects_extensions.rl_dnq import DNQDataProcessing
from .objects_extensions.reinforcement import RIDataProcessing
from .objects_extensions.reinforcement_finance import RIFDataProcessing
from .objects_extensions.rrl import RRLDataProcessing
from .objects_extensions.rrl_cnn_sltm import RRLCNNSLTMDataProcessing
from .objects_extensions.nn import NNDataProcessing
from .objects_extensions.mlnn import MLNNDataProcessing
from .objects_extensions.predict_shocks import SPDataProcessing
from .objects_extensions.simple_nn import SNNDataProcessing
from .objects_extensions.cup_handle import CHDataProcessing
from .objects_extensions.rnn_sp500 import SP500DataProcessing
#
from .objects_extensions.test import Test
#
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, SimpleRNN
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
#
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten
#


class MAlgo(object):
    def __init__(self, dic):  # to_data_path, target_field
        # print("90004-000 MLAlgo\n", dic, '\n', '-'*50)
        try:
            super(MAlgo, self).__init__()
        except Exception as ex:
            print("Error 90004-010 MAlgo:\n"+str(ex), "\n", '-'*50)
        # print("MLAlgo\n", self.app)
        # print("90004-020 MLAlgo\n", dic, '\n', '-'*50)
        self.app = dic["app"]


class MLDataProcessing(BaseDataProcessing, BasePotentialAlgo, MAlgo):
    def __init__(self, dic):
        # print("90005-000 MLDataProcessing\n", dic, '\n', '-' * 50)
        super().__init__(dic)
        # print("9005 MLDataProcessing ", self.app)


    # _1  I adjusted this function to work after uploading Eli data
    def mlp(self, dic):
        # print("90121-5: \n", "="*50, "\n", dic, "\n", "="*50)
        # load mnist dataset
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        # compute the number of labels
        num_labels = len(np.unique(y_train))
        # convert to one-hot vector
        y_train = to_categorical(y_train)
        y_test = to_categorical(y_test)
        # image dimensions (assumed square)
        image_size = x_train.shape[1]
        input_size = image_size * image_size
        # resize and normalize
        x_train = np.reshape(x_train, [-1, input_size])
        x_train = x_train.astype('float32') / 255
        x_test = np.reshape(x_test, [-1, input_size])
        x_test = x_test.astype('float32') / 255
        # network parameters
        batch_size = 128
        hidden_units = 256
        dropout = 0.45
        # model is a 3-layer MLP with ReLU and dropout after each layer
        model = Sequential()
        model.add(Dense(hidden_units, input_dim=input_size))
        model.add(Activation('relu'))
        model.add(Dropout(dropout))
        model.add(Dense(hidden_units))
        model.add(Activation('relu'))
        model.add(Dropout(dropout))
        model.add(Dense(num_labels))
        # this is the output for one-hot vector
        model.add(Activation('softmax'))
        print("model.summary()")
        model.summary()
        # file_path = os.path.join(self.PICKLE_PATH, "mlp-mnist.png")
        # plot_model(model, to_file=file_path, show_shapes=True)
        # loss function for one-hot vector
        # use of adam optimizer
        # accuracy is good metric for classification tasks
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        # train the network
        model.fit(x_train, y_train, epochs=20, batch_size=batch_size)
        # validate the model on test dataset to determine generalization
        _, acc = model.evaluate(x_test,
                                y_test,
                                batch_size=batch_size,
                                verbose=0)
        print("\nTest accuracy: %.1f%%" % (100.0 * acc))

        result = {"status": "ok"}
        return result

    def cnn(self, dic):
        print("90122-TEST1: \n", "="*50, "\n", dic, "\n", "="*50)
        # load mnist dataset
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        # compute the number of labels
        num_labels = len(np.unique(y_train))
        # convert to one-hot vector
        y_train = to_categorical(y_train)
        y_test = to_categorical(y_test)
        # input image dimensions
        image_size = x_train.shape[1]
        # resize and normalize
        x_train = np.reshape(x_train, [-1, image_size, image_size, 1])
        x_test = np.reshape(x_test, [-1, image_size, image_size, 1])
        x_train = x_train.astype('float32') / 255
        x_test = x_test.astype('float32') / 255
        # network parameters
        # image is processed as is (square grayscale)
        input_shape = (image_size, image_size, 1)
        batch_size = 128
        kernel_size = 3
        pool_size = 2
        filters = 64
        dropout = 0.2
        # model is a stack of CNN-ReLU-MaxPooling
        model = Sequential()
        model.add(Conv2D(filters=filters,
                         kernel_size=kernel_size,
                         activation='relu',
                         input_shape=input_shape))
        model.add(MaxPooling2D(pool_size))
        model.add(Conv2D(filters=filters,
                         kernel_size=kernel_size,
                         activation='relu'))
        model.add(MaxPooling2D(pool_size))
        model.add(Conv2D(filters=filters,
                         kernel_size=kernel_size,
                         activation='relu'))
        model.add(Flatten())
        # dropout added as regularizer
        model.add(Dropout(dropout))
        # output layer is 10-dim one-hot vector
        model.add(Dense(num_labels))
        model.add(Activation('softmax'))
        model.summary()
        # plot_model(model, to_file='cnn-mnist.png', show_shapes=True)
        # loss function for one-hot vector
        # use of adam optimizer
        # accuracy is good metric for classification tasks
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        # train the network
        model.fit(x_train, y_train, epochs=10, batch_size=batch_size)
        _, acc = model.evaluate(x_test,
                                y_test,
                                batch_size=batch_size,
                                verbose=0)
        print("\nTest accuracy: %.1f%%" % (100.0 * acc))

        result = {"status": "ok CNN"}
        return result

    def rnn(self, dic):
        print("90122-TEST1: \n", "="*50, "\n", dic, "\n", "="*50)
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        # compute the number of labels
        num_labels = len(np.unique(y_train))
        # convert to one-hot vector
        y_train = to_categorical(y_train)
        y_test = to_categorical(y_test)
        # resize and normalize
        image_size = x_train.shape[1]
        x_train = np.reshape(x_train, [-1, image_size, image_size])
        x_test = np.reshape(x_test, [-1, image_size, image_size])
        x_train = x_train.astype('float32') / 255
        x_test = x_test.astype('float32') / 255
        # network parameters
        input_shape = (image_size, image_size)
        batch_size = 128
        units = 256
        dropout = 0.2
        # model is RNN with 256 units, input is 28-dim vector 28 timesteps
        model = Sequential()
        model.add(SimpleRNN(units=units,
                            dropout=dropout,
                            input_shape=input_shape))
        model.add(Dense(num_labels))
        model.add(Activation('softmax'))
        model.summary()
        # plot_model(model, to_file='rnn-mnist.png', show_shapes=True)
        # loss function for one-hot vector
        # use of sgd optimizer
        # accuracy is good metric for classification tasks
        model.compile(loss='categorical_crossentropy',
                      optimizer='sgd',
                      metrics=['accuracy'])
        # train the network
        model.fit(x_train, y_train, epochs=20, batch_size=batch_size)
        _, acc = model.evaluate(x_test,
                                y_test,
                                batch_size=batch_size,
                                verbose=0)
        print("\nTest accuracy: %.1f%%" % (100.0 * acc))

        result = {"status": "ok RNN", "acc": acc}
        return result


