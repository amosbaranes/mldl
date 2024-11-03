from ..basic_ml_objects import BaseDataProcessing, BasePotentialAlgo
from ....core.utils import log_debug, clear_log_debug

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import random
import gym
import numpy as np
from collections import deque
import pickle
# ---------------------
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
from PIL import Image
# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)


class MLNNAlgo(object):
    def __init__(self, dic):  # to_data_path, target_field
        # print("90567-8-000 Algo\n", dic, '\n', '-'*50)
        try:
            super(MLNNAlgo, self).__init__()
        except Exception as ex:
            print("Error 9057-010 Algo:\n"+str(ex), "\n", '-'*50)

        self.app = dic["app"]


class MLNNDataProcessing(BaseDataProcessing, BasePotentialAlgo, MLNNAlgo):
    def __init__(self, dic):
        # print("90567-010 DataProcessing\n", dic, '\n', '-' * 50)
        super().__init__(dic)
        # print("9005 DataProcessing ", self.app)
        self.PATH = os.path.join(self.TO_OTHER, "mlnn")
        os.makedirs(self.PATH, exist_ok=True)
        # print(f'{self.PATH}')

        models = os.path.join(self.PATH, "models")
        os.makedirs(models, exist_ok=True)
        # print(f'{models}')

        pickles = os.path.join(self.PATH, "pickles")
        os.makedirs(pickles, exist_ok=True)
        # print(f'{pickles}')

        self.model_path = f'{models}/{"mlnn.h5"}'

        self.model_path_l = f'{pickles}/{"nn_lose.pkl"}'
        # print(self.model_path_l)

        self.model = None
        self.lose_list = None
        self.epochs = 10
        self.continue_train = 0

    def create_model(self):
        model = Sequential([
            Flatten(input_shape=(28, 28)),  # Flatten the 28x28 images into a 784-dimensional vector
            Dense(128, activation='relu'),  # Hidden layer with 128 neurons and ReLU activation
            Dense(64, activation='relu'),  # Hidden layer with 64 neurons and ReLU activation
            Dense(10, activation='softmax')
            # Output layer with 10 neurons (one for each digit) and softmax activation
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def create_model_cnn(self):
        model = Sequential([
            Flatten(input_shape=(28, 28)),  # Flatten 28x28 images to a 784-dimensional vector
            Dense(512, activation='relu'),  # First dense layer with 51  neurons
            BatchNormalization(),  # Apply Batch Normalization
            Dropout(0.2),  # Add Dropout to prevent overfitting
            Dense(256, activation='relu'),  # Second dense layer with 256 neurons
            BatchNormalization(),  # Apply Batch Normalization
            Dropout(0.2),  # Add Dropout to prevent overfitting
            Dense(128, activation='relu'),  # Third dense layer with 128 neurons
            Dense(10, activation='softmax')  # Output layer with 10 neurons (one for each digit)
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def load(self):
        if self.model_path and os.path.exists(self.model_path) and self.continue_train==1:
            # print(f"Loading model from {self.model_path}")
            # self.model = load_model(self.model_path)

            # Load weights into a new model (make sure the architecture is the same)
            self.model = self.create_model()
            self.model.load_weights(self.model_path)

            if self.model_path_l and os.path.exists(self.model_path_l):
                with open(self.model_path_l, 'rb') as file:
                    self.lose_list = pickle.load(file)
        else:
            # print("Creating new model")
            self.model = self.create_model()
            self.lose_list = []

    def save(self):
        # self.model.save(self.model_path)

        self.model.save_weights(self.model_path)

        print(f"Saving model to {self.model_path}")
        with open(self.model_path_l, 'wb') as file:
            pickle.dump(self.lose_list, file)

    def train(self, dic):
        print("90155-nn: \n", "="*50, "\n", dic, "\n", "="*50)
        (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
        # Normalize the images to values between 0 and 1
        train_images = train_images / 255.0
        test_images = test_images / 255.0
        # Convert labels to one-hot encoding
        train_labels = to_categorical(train_labels)
        test_labels = to_categorical(test_labels)
        self.epochs = int(dic["epochs"])
        self.continue_train = int(dic["continue_train"])
        self.load()
        if self.epochs > 0:
            history = self.model.fit(train_images, train_labels, epochs=self.epochs, batch_size=32)
            loss_values = history.history['loss']
            for k in range(len(loss_values)):
                self.lose_list.append(loss_values[k])
            self.save()
        # print(self.lose_list)
        # Check layer configurations after loading the model
        # for layer in self.model.layers:
        #     print(f"Layer: {layer.name}, Trainable: {layer.trainable}")

        # Evaluate the model
        test_loss, test_acc = self.model.evaluate(test_images, test_labels)
        print(f"Test accuracy: {test_acc:.4f}")
        print("Test loss: \n", test_loss)
        # ----------------
        result = {"status": "ok nn", "data":{"loss_values": self.lose_list, "test_accuracy":round(100*test_acc)/100,
                                             "test_loss":round(1000*test_loss)/1000}}
        return result

    def get_image(self, dic):
        print("90200-mlnn: \n", "="*50, "\n", dic, "\n", "="*50)
        (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
        # Normalize the images to values between 0 and 1
        train_images = train_images / 255.0
        test_images = test_images / 255.0
        # -----------------
        index = random.randint(0, len(test_images) - 1)
        img_ = test_images[index]
        img = Image.fromarray((img_ * 255).astype('uint8'), mode='L')
        # print("AA\n", test_labels[index])
        file_name_ = 'test_image'+str(test_labels[index])+'.png'
        file_name = self.TO_MEDIA + '/' + file_name_
        file_name_ = "/media"+file_name.split("media")[1]
        # -------------
        train_labels = to_categorical(train_labels)
        test_labels = to_categorical(test_labels)
        train_images = np.expand_dims(train_images[1], axis=0)
        train_labels = np.expand_dims(train_labels[1], axis=0)
        self.epochs = 1
        self.continue_train = 1
        self.load()
        history = self.model.fit(train_images, train_labels, epochs=self.epochs, batch_size=32)
        # ----------------
        img.save(file_name)
        # ----------------
        image = np.expand_dims(img_, axis=0)  # Add batch dimension
        prediction = self.model.predict(image)
        predicted_label = np.argmax(prediction)
        # -------
        result = {"status": "ok nn", "data":{'index': index, 'prediction': int(predicted_label),"file_name":file_name_}}
        return result
