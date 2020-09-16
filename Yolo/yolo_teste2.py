import cv2
import numpy as np
import time

#Load YOLO
#net = cv2.dnn.readNet("yolov3.weights","yolov3.cfg")
net = cv2.dnn.readNet("yolov3-tiny.weights","yolov3-tiny.cfg") 

# Classes 
classes = []
with open("coco.names","r") as f:
    classes = [line.strip() for line in f.readlines()]

#print(classes)


layer_names = net.getLayerNames() # estrutura da rede pre-treinada
outputlayers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()] # Camadas da rede nao conectadas

colors= np.random.uniform(0, 255, size=(len(classes), 3)) # Cores para o boundingBoxes 


#loading image
cap=cv2.VideoCapture(0) 

while True:
    ret,frame= cap.read() 
    
    height, width, channels = frame.shape

    #Detectando objetos 
    # Funcao blob - basicamente extrai as caracteristicas dos objs
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (320, 320), (0, 0, 0), True, crop=False)         
    net.setInput(blob) # Passar as caracteristicas obtidas no blob na rede
    outs = net.forward(outputlayers) # Extrair todas as informacoes do objeto captado
    #print(outs[1])

    """
    Com todas as informacoes obtidas e preciso fazer as classificacoes, primeiramente encontra-se um score
    para cada informacao do 'outs'. Depois pega-se a classe com o maior score encontrada e compara-se com uma 
    confianca.
    """

    class_ids=[]
    confidences=[]
    boxes=[]
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                #Objeto detectato
                center_x= int(detection[0]*width)
                center_y= int(detection[1]*height)
                w = int(detection[2]*width)
                h = int(detection[3]*height)

                # Coord retang
                x=int(center_x - w/2)
                y=int(center_y - h/2)

                boxes.append([x,y,w,h]) # Pegando as areas do retang
                confidences.append(float(confidence)) # Porcentagem do obj em relacao a confianca
                class_ids.append(class_id) #classe do obj

    # Eliminar caixas duplas do mesmo objeto
    indexes = cv2.dnn.NMSBoxes(boxes,confidences,0.5,0.4)

    # Fazer os boundigboxes
    for i in range(len(boxes)):
        if i in indexes:
            x,y,w,h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence= confidences[i]
            color = colors[class_ids[i]]
            cv2.rectangle(frame,(x,y),(x+w,y+h),color,2)
            cv2.putText(frame,label+" "+str(round(confidence,2)),(x,y+30),cv2.FONT_HERSHEY_PLAIN,1,(255,255,255),2)
            


    cv2.imshow("Image",frame)
    key = cv2.waitKey(1) 
    
    if key == 27: #esc key stops 
        break
    
cap.release()    
cv2.destroyAllWindows()

 