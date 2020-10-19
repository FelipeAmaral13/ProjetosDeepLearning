# Bibliotecas para criar o modelo
from __future__ import print_function
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D

# Bibliotecas
from Split_Dataset import load_data_training_and_test
import numpy as np
import cv2
import matplotlib.pyplot as plt


(x_train, y_train), (x_test, y_test) = load_data_training_and_test("cats_vs_dogs")

# Reshape os labels de (2000,) para (2000,1) e teste data  (1000,) para (1000,1)
y_train = y_train.reshape(y_train.shape[0], 1)
y_test = y_test.reshape(y_test.shape[0], 1)

# Setando as imagens como type float32
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# Normalizar os dados
x_train /= 255
x_test /= 255

print(f'Lendo as imagens de treinamento. Total de {x_train.shape[0]} imagens lidas.Formato das imagens são {x_train.shape[1]}x{x_train.shape[2]}.Total de canais das imagens {x_train.shape[-1]}')
print(f'Lendo as imagens de treinamento. Total de {y_train.shape[0]} labels')

print(f'Lendo as imagens de teste. Total de {x_test.shape[0]} imagens lidas. Formato das imagens são {x_test.shape[1]}x{x_test.shape[2]}. Total de canais das imagens {x_test.shape[-1]}')
print(f'Lendo as imagens de teste. Total de {y_test.shape[0]} labels')

# Criando a CNN model
batch_size = 16
epochs = 20

img_rows = x_train[0].shape[0]
img_cols = x_train[1].shape[0]
input_shape = (img_rows, img_cols, 3)

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

print(model.summary())

# Treinar o modelo
history = model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test),
          shuffle=True)

# Salvar o modelo
model.save("ModeloTreinado_V1.h5")

# Avaliando a perfmorce do modelo
scores = model.evaluate(x_test, y_test, verbose=1)
print('Teste Loss:', scores[0])
print('Teste accuracy:', scores[1])

# testar modelo

classifier = load_model('ModeloTreinado_V1.h5') # Lendo o modelo treinado

def draw_test(name, pred, input_im):
    BLACK = [0,0,0]
    if pred == "[0]":
        pred = "cat"
    if pred == "[1]":
        pred = "dog"
    expanded_image = cv2.copyMakeBorder(input_im, 0, 0, 0, imageL.shape[0] ,cv2.BORDER_CONSTANT,value=BLACK)
    #expanded_image = cv2.cvtColor(expanded_image, cv2.COLOR_GRAY2BGR)
    cv2.putText(expanded_image, str(pred), (400, 70) , cv2.FONT_HERSHEY_COMPLEX_SMALL,4, (0,255,0), 2)
    cv2.imshow(name, expanded_image)


for i in range(0,10):
    rand = np.random.randint(0,len(x_test))
    input_im = x_test[rand]

    imageL = cv2.resize(input_im, None, fx=2, fy=2, interpolation = cv2.INTER_CUBIC)
    #cv2.imshow("Test Image", imageL)

    input_im = input_im.reshape(1,150,150,3) 
    
    ## Pred
    res = str(classifier.predict_classes(input_im, 1, verbose = 0)[0])

    draw_test("Prediction", res, imageL) 
    cv2.waitKey(0)

cv2.destroyAllWindows()

# Plotting Loss Curva

history_dict = history.history

loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
epochs = range(1, len(loss_values) + 1)

acc_values = history_dict['accuracy']
val_acc_values = history_dict['val_accuracy']

line1 = plt.plot(epochs, val_loss_values, label='Validation/Test Loss')
line2 = plt.plot(epochs, loss_values, label='Training Loss')
plt.setp(line1, linewidth=2.0, marker = '+', markersize=10.0)
plt.setp(line2, linewidth=2.0, marker = '4', markersize=10.0)
plt.xlabel('Epocas') 
plt.ylabel('Loss')
plt.grid(True)
plt.legend()
plt.show()

line1 = plt.plot(epochs, val_acc_values, label='Validation/Test Accuracy')
line2 = plt.plot(epochs, acc_values, label='Training Accuracy')
plt.setp(line1, linewidth=2.0, marker = '+', markersize=10.0)
plt.setp(line2, linewidth=2.0, marker = '4', markersize=10.0)
plt.xlabel('Epocas') 
plt.ylabel('Accuracy')
plt.grid(True)
plt.legend()
plt.show()