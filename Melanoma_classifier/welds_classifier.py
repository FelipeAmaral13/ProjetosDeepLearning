import numpy as np
import os
import cv2
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
import tensorflow as tf
from keras.callbacks import ModelCheckpoint
from sklearn.utils import shuffle


class MelanomaClass:
    def __init__(self):
        self.class_names = ['benign', 'malignant']
        self.class_names_label = {class_name:i for i, class_name in enumerate(self.class_names)}
        self.nb_classes = len(self.class_names)
        self.IMAGE_SIZE = 32
        self.model = None
        self.history = None

    def load_data(self):
        datasets = [os.path.join('Dataset', 'melanoma_cancer_dataset', 'train'), os.path.join('Dataset', 'melanoma_cancer_dataset', 'test')]
        output = []
        for dataset in datasets:        
            images = []
            labels = []        

            for folder in os.listdir(dataset):
                label = self.class_names_label[folder]
                for file in tqdm(os.listdir(os.path.join(dataset, folder))):
                    img_path = os.path.join(os.path.join(dataset, folder), file)
                    try:
                        image = cv2.imread(img_path)
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        image = cv2.resize(image, (self.IMAGE_SIZE, self.IMAGE_SIZE))
                    except:
                        print(img_path)
                        raise 'Error! Image not read'
                    images.append(image)
                    labels.append(label)
            images = np.array(images, dtype = 'float32')
            labels = np.array(labels, dtype = 'int32')   
            output.append((images, labels))

        return output


    def plot_sample_images(self):
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        for i, class_name in tqdm(enumerate(self.class_names)):
            files = os.listdir(os.path.join('Dataset', 'melanoma_cancer_dataset', 'train', class_name))
            file_path = os.path.join('Dataset', 'melanoma_cancer_dataset', 'train', class_name, random.choice(files))
            try:
                image = cv2.imread(file_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            except:
                print(file_path)
                raise 'Error! Image not read'
            if i == 0:
                axes[i].imshow(image)
                axes[i].set_title('Benign')
            else:
                axes[i].imshow(image)
                axes[i].set_title('Malignant')
        plt.show()



    def train(self, batch_size=2, epochs=20, validation_split=0.2):
        (train_images, train_labels), (test_images, test_labels) = self.load_data()
        train_images, train_labels = shuffle(train_images, train_labels)
        train_images = train_images / 255.0 
        test_images = test_images / 255.0

        self.model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(6, (5, 5), activation='relu', input_shape=(self.IMAGE_SIZE, self.IMAGE_SIZE, 3)),
            tf.keras.layers.AveragePooling2D(pool_size=(2, 2)),
            tf.keras.layers.Conv2D(16, (5, 5), activation='relu'),
            tf.keras.layers.AveragePooling2D(pool_size=(2, 2)),
            tf.keras.layers.Dense(120, activation='relu'),
            tf.keras.layers.Flatten(),            
            tf.keras.layers.Dense(84, activation='relu'),
            tf.keras.layers.Dense(2, activation='softmax')
        ])
        checkpoint_callback = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True)
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        self.history = self.model.fit(
            train_images,
            train_labels,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(test_images, test_labels),
            validation_split=validation_split,
            callbacks=[checkpoint_callback]
            )


    def evaluate(self):
        if self.model is None:
            raise Exception('Model not trained yet.')
        (train_images, train_labels), (test_images, test_labels) = self.load_data()
        test_images = test_images / 255.0
        test_loss = self.model.evaluate(test_images, test_labels)
        return test_loss
    
    def save_model(self):
        self.model.save(os.path.join('model_cnn','cnn_model.h5'), save_format='tf')

    def load_model(self):
        model_ = tf.keras.models.load_model(os.path.join('best_model.h5'))
        return model_

    def predict(self, image_path):
        self.model_ = self.load_model()
        if self.model_ is None:
            raise Exception('Model not trained yet.')
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.IMAGE_SIZE, self.IMAGE_SIZE))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)
        prediction = self.model_.predict(img)
        predicted_class_idx = np.argmax(prediction)
        predicted_class = self.class_names[predicted_class_idx]
        return predicted_class


melanoma_class = MelanomaClass()
melanoma_class.plot_sample_images()
melanoma_class.train()
melanoma_class.evaluate()
