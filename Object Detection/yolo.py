import cv2 as cv
import numpy as np

cap = cv.VideoCapture('2.mp4')
whT = 320
confThreshold = 0.3
nmsThreshold = 0.4

# LOAD MODEL
# Coco Names
classesFile = "coco.names"
classNames = []
with open(classesFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')
print(classNames)

colors= np.random.uniform(0, 255, size=(len(classNames), 3)) # Cores para o boundingBoxes 


# Model Files
modelConfiguration = "yolov3-tiny.cfg"  # config. dos modelo
modelWeights = "yolov3-tiny.weights"  # pesos pre-treinados


net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
# OpenCV como Bakend. Não CPU
net.setPreferableBackend(cv.dnn. DNN_BACKEND_DEFAULT )
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)


def findObjects(outputs, frame):
    '''
    Funcao para encontrar os obejtos. 
    '''

    hT, wT, cT = frame.shape
     
    bbox = [] # Vetor para o boundingBox
    classIds = [] # Vator para as classes
    confs = [] # Confianca da classificacao

    for output in outputs:
        for det in output:
            scores = det[5:] # Pagar os cinco primeiros valores da analise da rede. (cx, cy, w, h, confianca da class)
            classId = np.argmax(scores) # Pegar a classificacao
            confidence = scores[classId] # Pegar a confianca da classificacao
            # Filtrar nossa classificacao
            if confidence > confThreshold:
                w, h = int(det[2]*wT), int(det[3]*hT)
                x, y = int((det[0]*wT)-w/2), int((det[1]*hT)-h/2)
                bbox.append([x, y, w, h])
                classIds.append(classId)
                confs.append(float(confidence))

    indices = cv.dnn.NMSBoxes(bbox, confs, confThreshold, nmsThreshold) # evitar varias caixas. Supressao Maxima

    # BoundingBoxes
    for i in indices:
        i = i[0]
        box = bbox[i]
        x, y, w, h = box[0], box[1], box[2], box[3]
        # print(x,y,w,h)
        cv.rectangle(frame, (x, y), (x+w, y+h), colors[classIds[i]], 2)
        cv.putText(frame, f'{classNames[classIds[i]].upper()} {int(confs[i]*100)}%',
                   (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 0.6, colors[classIds[i]], 2)


while True:
    ret, frame = cap.read()

    # Criar 4-dimensional blob da imagem. 
    # Opcionalmente, redimensiona e recorta a imagem do centro, subtrai os valores médios, dimensiona os valores por fator de escala, troca os canais Azul e Vermelho.
    blob = cv.dnn.blobFromImage(
        frame, 1 / 255, (whT, whT), [0, 0, 0], 1, crop=False)  
    net.setInput(blob)

    layersNames = net.getLayerNames() # Camadas da rede CNN treinada
    # Não usamos o valor zero das camadas de saida por isso a subtracao com 1.
    # Pegar o nome das camadas de saida -> print(outputNames)
    outputNames = [(layersNames[i[0] - 1])
                   for i in net.getUnconnectedOutLayers()] # Na saida temos tres tipos. Por isso um vetor com tres posicoes em net.getUnconnectedOutLayers()

    outputs = net.forward(outputNames) # Saida das camadas da rede com a confianca predita
    findObjects(outputs, frame)

    cv.imshow('Image', frame)

    if cv.waitKey(1) == 27:
        break

cv.destroyAllWindows()
cap.release()
