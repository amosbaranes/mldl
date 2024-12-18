from ..basic_ml_objects import BaseDataProcessing, BasePotentialAlgo
from ....core.templatetags.core_tags import model_name
from ....core.utils import log_debug, clear_log_debug
#
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
#
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

# ----------------

class VAE(Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstructed = self.decoder(z)
        return reconstructed

    def compute_loss(self, inputs, outputs, z_mean, z_log_var):
        # Ensure proper flattening and calculation
        reconstruction_loss = tf.reduce_mean(
            tf.keras.losses.binary_crossentropy(
                tf.keras.backend.flatten(inputs),
                tf.keras.backend.flatten(outputs)
            )
        )
        kl_loss = -0.5 * tf.reduce_mean(
            1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
        )

        total_loss = reconstruction_loss + kl_loss
        return total_loss, reconstruction_loss, kl_loss

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstructed = self.decoder(z)
            total_loss, reconstruction_loss, kl_loss = self.compute_loss(data, reconstructed, z_mean, z_log_var)
            # Compute gradients and apply them
            grads = tape.gradient(total_loss, self.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        # Return a dictionary of losses
        return {"loss": total_loss, "reconstruction_loss": reconstruction_loss, "kl_loss": kl_loss}
# ----------------

class A1PPOAlgo(object):
    def __init__(self, dic):
        # print("90567-8-000 Algo\n", dic, '\n', '-'*50)
        try:
            super(A1PPOAlgo, self).__init__()
        except Exception as ex:
            print("Error 9057-010 Algo:\n"+str(ex), "\n", '-'*50)
        # print("MLAlgo\n", self.app)
        # print("90004-020 Algo\n", dic, '\n', '-'*50)
        self.app = dic["app"]

# https://chatgpt.com/c/66e0947c-6714-800c-9probabilitiesef3-3aa45026ed5a
class A1PPODataProcessing(BaseDataProcessing, BasePotentialAlgo, A1PPOAlgo):
    def __init__(self, dic):
        print("90567-010 RRLDataProcessing\n", dic, '\n', '-' * 50)
        super().__init__(dic)

    def train(self, dic):
        print("\ntrain: \n", "="*50, "\n", dic, "\n", "="*50)
        import tensorflow as tf
        from tensorflow.keras import layers, Model

        # Latent space dimension
        latent_dim = 2

        # Encoder
        encoder_inputs = layers.Input(shape=(28, 28, 1), name="encoder_input")
        x = layers.Conv2D(32, kernel_size=3, activation="relu", strides=2, padding="same")(encoder_inputs)
        x = layers.Conv2D(64, kernel_size=3, activation="relu", strides=2, padding="same")(x)
        x = layers.Flatten()(x)
        x = layers.Dense(16, activation="relu")(x)
        z_mean = layers.Dense(latent_dim, name="z_mean")(x)
        z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)

        # Reparameterization trick
        def sampling(args):
            z_mean, z_log_var = args
            epsilon = tf.random.normal(shape=tf.shape(z_mean))
            return z_mean + tf.exp(0.5 * z_log_var) * epsilon

        z = layers.Lambda(sampling, name="z")([z_mean, z_log_var])

        # Encoder model
        encoder = Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
        encoder.summary()

        # Decoder
        latent_inputs = layers.Input(shape=(latent_dim,), name="z_sampling")
        x = layers.Dense(7 * 7 * 64, activation="relu")(latent_inputs)
        x = layers.Reshape((7, 7, 64))(x)
        x = layers.Conv2DTranspose(64, kernel_size=3, activation="relu", strides=2, padding="same")(x)
        x = layers.Conv2DTranspose(32, kernel_size=3, activation="relu", strides=2, padding="same")(x)
        decoder_outputs = layers.Conv2DTranspose(1, kernel_size=3, activation="sigmoid", padding="same")(x)

        # Decoder model
        decoder = Model(latent_inputs, decoder_outputs, name="decoder")
        decoder.summary()

        # Variational Autoencoder (VAE) Model

        # Instantiate and compile VAE
        vae = VAE(encoder, decoder)
        vae.compile(optimizer=tf.keras.optimizers.Adam())

        # Load and preprocess MNIST dataset
        (x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()
        x_train = x_train.astype("float32") / 255.0
        x_test = x_test.astype("float32") / 255.0
        x_train = tf.expand_dims(x_train, -1)
        x_test = tf.expand_dims(x_test, -1)

        # Train the VAE
        print("A10-3")
        vae.fit(x_train, epochs=10, batch_size=128, validation_data=(x_test, x_test))
        print("A10-4")

        print("A10 \n")

        print("Training is done")
        result = {"status": "ok"}
        return result
