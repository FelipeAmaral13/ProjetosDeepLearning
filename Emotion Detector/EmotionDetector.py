###################################################################
# O objetivo desse programa é fazer a classificacao de modelos a  #
# da emocao captada pela webcam.                                  #
###################################################################


# Bibliotecas
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ler modelo treinado
classifier = load_model('emotion_little_vgg.h5')

# Pasta das imagens de validacao
validation_data_dir = './fer2013/validation'



num_features = 64
num_labels = 6
batch_size = 64
width, height = 48, 48

validation_datagen = ImageDataGenerator(rescale=1./255)


# Pegando as classes

validation_generator = validation_datagen.flow_from_directory(
        validation_data_dir,
        color_mode = 'grayscale',
        target_size=(width, height),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False)

class_labels = validation_generator.class_indices
class_labels = {v: k for k, v in class_labels.items()}
classes = list(class_labels.values())
print(class_labels)


# testando na Webcam

import cv2
import numpy as np
from time import sleep
from tensorflow.keras.preprocessing.image import img_to_array

face_classifier = cv2.CascadeClassifier('./Haarcascades/haarcascade_frontalface_default.xml')

def face_detector(img):

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Deteccao de faces
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    
    if faces is ():
        return (0,0,0,0), np.zeros((48,48), np.uint8), img
    
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]

    try:
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation = cv2.INTER_AREA)
    except:
        return (x,w,y,h), np.zeros((48,48), np.uint8), img
    return (x,w,y,h), roi_gray, img

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()
    rect, face, image = face_detector(frame)
    if np.sum([face]) != 0.0:
        roi = face.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)

        # make a prediction on the ROI, then lookup the class
        preds = classifier.predict(roi)[0]
        label = class_labels[preds.argmax()]  
        label_position = (rect[0] + int((rect[1]/2)), rect[2] + 25)
        cv2.putText(image, label, label_position , cv2.FONT_HERSHEY_SIMPLEX,2, (0,255,0), 3)
    else:
        #cv2.putText(image, "No Face Found", (20, 60) , cv2.FONT_HERSHEY_SIMPLEX,2, (0,255,0), 3)
        pass
        
    cv2.imshow('Emotion Face Detect', image)
    if cv2.waitKey(1) == 27: # Aperte Esc para sair
        break
        
cap.release()
cv2.destroyAllWindows()      