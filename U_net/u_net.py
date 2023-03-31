# -*- coding: utf-8 -*-
import numpy as np
import os
import glob
import cv2
from keras.utils import array_to_img
import matplotlib.pyplot as plt
import tensorflow as tf


class SegmentationUnet:
    def __init__(self):
        self.__img_dim = 512
        self.build_model()
        try:
            self.load_model(os.path.join("model_unet", "model.h5"))
        except:
            print("Nao foi possivel encontrar o modelo treinado")
            pass

    def load_data(self, image_dir, mask_dir):
        """
        load_data

        This is a static method used to load image data from a given directory.

        Parameters:

            image_dir (str): The directory path where the images are stored.
            mask_dir (str): The directory path where the images masks are stored.
            colormode (str, optional): The color mode of the images. Default value is "rgb".

        Returns:

            numpy array: The numpy array with the loaded image data.
        """

        dataset_img = np.zeros(
            (len(os.listdir(image_dir)), self.__img_dim, self.__img_dim, 3),
            dtype=np.uint8,
        )
        dataset_mask = np.zeros(
            (len(os.listdir(mask_dir)), self.__img_dim, self.__img_dim, 1),
            dtype=np.bool_,
        )

        for index, image in enumerate(glob.glob(os.path.join(image_dir, "*.*"))):
            img = cv2.imread(image)
            img = cv2.resize(img, (self.__img_dim, self.__img_dim))
            dataset_img[index] = img
            mask = np.zeros((self.__img_dim, self.__img_dim, 1), dtype=np.bool_)

        for index, image in enumerate(glob.glob(os.path.join(mask_dir, "*.*"))):
            mask_ = cv2.imread(image)
            mask_ = cv2.cvtColor(mask_, cv2.COLOR_BGR2GRAY)
            mask_ = np.expand_dims(
                cv2.resize(mask_, (self.__img_dim, self.__img_dim)), axis=-1
            )
            mask = np.maximum(mask, mask_)
            dataset_mask[index] = mask_

        return dataset_img, dataset_mask

    @staticmethod
    def train_test_split(X_data, y_data):
        """
        Splits the data into training and testing sets.

        Parameters:
        X_data -- input data array
        y_data -- output data array

        Returns:
        X_train -- input training dataset
        y_train -- output training dataset
        X_test -- input test dataset
        y_test -- output test dataset
        """
        X_train, X_test = (
            X_data[: int(len(X_data) * 0.8)],
            X_data[int(len(X_data) * 0.8) :],
        )
        y_train, y_test = (
            y_data[: int(len(y_data) * 0.8)],
            y_data[int(len(y_data) * 0.8) :],
        )
        return X_train, y_train, X_test, y_test

    def build_model(self):
        num_classes = 1

        self.inputs = tf.keras.layers.Input((self.__img_dim, self.__img_dim, 3))

        # encode
        self.c1 = tf.keras.layers.Conv2D(
            16,
            (3, 3),
            activation="relu",
            kernel_initializer="he_normal",
            padding="same",
        )(self.inputs)
        self.c1 = tf.keras.layers.Dropout(0.1)(self.c1)
        self.c1 = tf.keras.layers.Conv2D(
            16,
            (3, 3),
            activation="relu",
            kernel_initializer="he_normal",
            padding="same",
        )(self.c1)
        self.b1 = tf.keras.layers.BatchNormalization()(self.c1)
        self.r1 = tf.keras.layers.ReLU()(self.b1)
        self.p1 = tf.keras.layers.MaxPooling2D((2, 2))(self.r1)

        self.c2 = tf.keras.layers.Conv2D(
            32,
            (3, 3),
            activation="relu",
            kernel_initializer="he_normal",
            padding="same",
        )(self.p1)
        self.c2 = tf.keras.layers.Dropout(0.1)(self.c2)
        self.c2 = tf.keras.layers.Conv2D(
            32,
            (3, 3),
            activation="relu",
            kernel_initializer="he_normal",
            padding="same",
        )(self.c2)
        self.b2 = tf.keras.layers.BatchNormalization()(self.c2)
        self.r2 = tf.keras.layers.ReLU()(self.b2)
        self.p2 = tf.keras.layers.MaxPooling2D((2, 2))(self.r2)

        self.c3 = tf.keras.layers.Conv2D(
            64,
            (3, 3),
            activation="relu",
            kernel_initializer="he_normal",
            padding="same",
        )(self.p2)
        self.c3 = tf.keras.layers.Dropout(0.2)(self.c3)
        self.c3 = tf.keras.layers.Conv2D(
            64,
            (3, 3),
            activation="relu",
            kernel_initializer="he_normal",
            padding="same",
        )(self.c3)
        self.b3 = tf.keras.layers.BatchNormalization()(self.c3)
        self.r3 = tf.keras.layers.ReLU()(self.b3)
        self.p3 = tf.keras.layers.MaxPooling2D((2, 2))(self.r3)

        self.c4 = tf.keras.layers.Conv2D(
            128,
            (3, 3),
            activation="relu",
            kernel_initializer="he_normal",
            padding="same",
        )(self.p3)
        self.c4 = tf.keras.layers.Dropout(0.2)(self.c4)
        self.c4 = tf.keras.layers.Conv2D(
            128,
            (3, 3),
            activation="relu",
            kernel_initializer="he_normal",
            padding="same",
        )(self.c4)
        self.b4 = tf.keras.layers.BatchNormalization()(self.c4)
        self.r4 = tf.keras.layers.ReLU()(self.b4)
        self.p4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(self.r4)

        self.c5 = tf.keras.layers.Conv2D(
            256,
            (3, 3),
            activation="relu",
            kernel_initializer="he_normal",
            padding="same",
        )(self.p4)
        self.b5 = tf.keras.layers.BatchNormalization()(self.c5)
        self.r5 = tf.keras.layers.ReLU()(self.b5)
        self.c5 = tf.keras.layers.Dropout(0.3)(self.r5)
        self.c5 = tf.keras.layers.Conv2D(
            256,
            (3, 3),
            activation="relu",
            kernel_initializer="he_normal",
            padding="same",
        )(self.c5)

        # bridge
        self.b6 = tf.keras.layers.Conv2D(
            512,
            (3, 3),
            activation="relu",
            kernel_initializer="he_normal",
            padding="same",
        )(self.c5)
        self.b6 = tf.keras.layers.BatchNormalization()(self.b6)
        self.b6 = tf.keras.layers.ReLU()(self.b6)

        # decode
        self.u7 = tf.keras.layers.Conv2DTranspose(
            128, (2, 2), strides=(2, 2), padding="same"
        )(self.b6)
        self.u7 = tf.keras.layers.concatenate([self.u7, self.c4])
        self.c7 = tf.keras.layers.Conv2D(
            128,
            (3, 3),
            activation="relu",
            kernel_initializer="he_normal",
            padding="same",
        )(self.u7)
        self.c7 = tf.keras.layers.Dropout(0.2)(self.c7)
        self.c7 = tf.keras.layers.Conv2D(
            128,
            (3, 3),
            activation="relu",
            kernel_initializer="he_normal",
            padding="same",
        )(self.c7)
        self.b7 = tf.keras.layers.BatchNormalization()(self.c7)
        self.r7 = tf.keras.layers.ReLU()(self.b7)

        self.u8 = tf.keras.layers.Conv2DTranspose(
            64, (2, 2), strides=(2, 2), padding="same"
        )(self.r7)
        self.u8 = tf.keras.layers.concatenate([self.u8, self.c3])
        self.c8 = tf.keras.layers.Conv2D(
            64,
            (3, 3),
            activation="relu",
            kernel_initializer="he_normal",
            padding="same",
        )(self.u8)
        self.c8 = tf.keras.layers.Dropout(0.2)(self.c8)
        self.c8 = tf.keras.layers.Conv2D(
            64,
            (3, 3),
            activation="relu",
            kernel_initializer="he_normal",
            padding="same",
        )(self.c8)
        self.b8 = tf.keras.layers.BatchNormalization()(self.c8)
        self.r8 = tf.keras.layers.ReLU()(self.b8)

        self.u9 = tf.keras.layers.Conv2DTranspose(
            32, (2, 2), strides=(2, 2), padding="same"
        )(self.r8)
        self.u9 = tf.keras.layers.concatenate([self.u9, self.c2])
        self.c9 = tf.keras.layers.Conv2D(
            32,
            (3, 3),
            activation="relu",
            kernel_initializer="he_normal",
            padding="same",
        )(self.u9)
        self.c9 = tf.keras.layers.Dropout(0.1)(self.c9)
        self.c9 = tf.keras.layers.Conv2D(
            32,
            (3, 3),
            activation="relu",
            kernel_initializer="he_normal",
            padding="same",
        )(self.c9)
        self.b9 = tf.keras.layers.BatchNormalization()(self.c9)
        self.r9 = tf.keras.layers.ReLU()(self.b9)

        self.u10 = tf.keras.layers.Conv2DTranspose(
            16, (2, 2), strides=(2, 2), padding="same"
        )(self.r9)
        self.u10 = tf.keras.layers.concatenate([self.u10, self.c1], axis=3)
        self.c10 = tf.keras.layers.Conv2D(
            16,
            (3, 3),
            activation="relu",
            kernel_initializer="he_normal",
            padding="same",
        )(self.u10)
        self.c10 = tf.keras.layers.Dropout(0.1)(self.c10)
        self.c10 = tf.keras.layers.Conv2D(
            16,
            (3, 3),
            activation="relu",
            kernel_initializer="he_normal",
            padding="same",
        )(self.c10)
        self.b10 = tf.keras.layers.BatchNormalization()(self.c10)
        self.r10 = tf.keras.layers.ReLU()(self.b10)

        self.outputs = tf.keras.layers.Conv2D(
            num_classes, (1, 1), activation="sigmoid"
        )(self.r10)

        self.model = tf.keras.Model(inputs=[self.inputs], outputs=[self.outputs])

        return self.model

    def train(
        self,
        X_train: np.array,
        y_train: np.array,
        X_val: np.array,
        y_val: np.array,
        batch_size: int,
        epochs: int,
    ):
        """
        Function to train the model with given train and validation data.

        Parameters:
            X_train (numpy array): A numpy array containing the training data.
            y_train (numpy array): A numpy array containing the training labels.
            X_val (numpy array): A numpy array containing the validation data.
            y_val (numpy array): A numpy array containing the validation labels.
            batch_size (int): An integer representing the batch size used during training.
            epochs (int): An integer representing the number of training epochs.

        Returns:
            None
        """
        self.model.compile(
            optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
        )
        self.model.fit(
            X_train,
            y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(X_val, y_val),
        )

    def evaluate(self, X_test, y_test, batch_size):
        """
        Evaluate the model on test data.

        Parameters:
        -----------
        X_test: numpy array
            Array of test data samples.
        y_test: numpy array
            Array of target values for the test samples.
        batch_size: int
            Number of samples to be used in one forward/backward pass.

        Returns:
        --------
        None
        """
        results = self.model.evaluate(X_test, y_test, batch_size=batch_size)
        print("Test loss:", results[0])
        print("Test accuracy:", results[1])

    def predict(self, X_test):
        """
        Predict class probabilities for the input test data.

        Parameters:
        -----------
        X_test: numpy array
            Array of test data samples.
        batch_size: int
            Number of samples to be used in one forward/backward pass.

        Returns:
        --------
        predictions: numpy array
            Array of predicted class probabilities.
        """
        predictions = self.model.predict(X_test)
        return predictions

    def save_model(self, model_path):
        """
        Saves the trained model.

        Parameters:
        model_path -- The path to save the model.

        Returns:
        None
        """
        self.model.save(model_path)

    def load_model(self, model_path):
        """
        Load a pre-trained model from a file.

        Parameters:
        -----------
        model_file: str
            The file path of the pre-trained model.

        Returns:
        --------
        None
        """
        self.model.load_weights(model_path)
