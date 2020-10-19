# Bibliotecas
import cv2
import numpy as np
import sys
import os
from os import listdir
from os.path import isfile, join
import shutil
import matplotlib.pyplot as plt

# Caminho onde se encontra o programa .py
mypath = os.getcwd()

# Verificar os arquivos no dataset e quantifica-los
file_names = [f for f in listdir(mypath + "\\images") if isfile(join(mypath + "\\images", f))]

print(str(len(file_names)) + ' Imagens lidas')

# Split as imagens lidas em Treino teste/validacao dataset

# Extrair 1000 para os dados de treinamento e 500 para nosso conjunto de validação
dog_count = 0
cat_count = 0
training_size = 1000
test_size = 500
training_images = []
training_labels = []
test_images = []
test_labels = []
size = 150
dog_dir_train = mypath + "\\train\\dogs"
cat_dir_train = mypath + "\\train\\cats"
dog_dir_val = mypath + "\\validation\\dogs"
cat_dir_val = mypath + "\\validation\\cats"

def make_dir(directory):
        if os.path.exists(directory):
            shutil.rmtree(directory)
        os.makedirs(directory)

make_dir(dog_dir_train)
make_dir(cat_dir_train)
make_dir(dog_dir_val)
make_dir(cat_dir_val)

def getZeros(number):
    if(number > 10 and number < 100):
        return "0"
    if(number < 10):
        return "00"
    else:
        return ""

for i, file in enumerate(file_names):
    
    if file_names[i][0] == "d":
        dog_count += 1
        image = cv2.imread(mypath + '\\images\\' + file)
        image = cv2.resize(image, (size, size), interpolation = cv2.INTER_AREA)
        if dog_count <= training_size:
            training_images.append(image)
            training_labels.append(1)
            zeros = getZeros(dog_count)
            cv2.imwrite(dog_dir_train + '\\' + 'dog'  + str(zeros) + str(dog_count) + ".jpg", image)
        if dog_count > training_size and dog_count <= training_size+test_size:
            test_images.append(image)
            test_labels.append(1)
            zeros = getZeros(dog_count-1000)
            cv2.imwrite(dog_dir_val + '\\' + 'dog'  + str(zeros) + str(dog_count-1000) + ".jpg", image)
            
    if file_names[i][0] == "c":
        cat_count += 1
        image = cv2.imread(mypath + '\\images\\' + file)
        image = cv2.resize(image, (size, size), interpolation = cv2.INTER_AREA)
        if cat_count <= training_size:
            training_images.append(image)
            training_labels.append(0)
            zeros = getZeros(cat_count)
            cv2.imwrite(cat_dir_train + '\\' + 'cat' + str(zeros) + str(cat_count) + ".jpg", image)
        if cat_count > training_size and cat_count <= training_size+test_size:
            test_images.append(image)
            test_labels.append(0)
            zeros = getZeros(cat_count-1000)
            cv2.imwrite(cat_dir_val + '\\' + 'cat' + str(zeros) + str(cat_count-1000) + ".jpg", image)

    if dog_count == training_size+test_size and cat_count == training_size+test_size:
        break

print("Dados de Treinamento e Teste Extradidos com Sucesso")


# Usando o numpy para armazenar nosso datas como NPZ files
np.savez('cats_vs_dogs_training_data.npz', np.array(training_images))
np.savez('cats_vs_dogs_training_labels.npz', np.array(training_labels))
np.savez('cats_vs_dogs_test_data.npz', np.array(test_images))
np.savez('cats_vs_dogs_test_labels.npz', np.array(test_labels))

def load_data_training_and_test(datasetname):
    '''
    Função responsável pela leitura do DataSet.
    Entrada: Dataset
    Saída: Dado para treino/teste, rótulo do dado de treino/teste
    '''
    
    npzfile = np.load(datasetname + "_training_data.npz")
    train = npzfile['arr_0']
    
    npzfile = np.load(datasetname + "_training_labels.npz")
    train_labels = npzfile['arr_0']
    
    npzfile = np.load(datasetname + "_test_data.npz")
    test = npzfile['arr_0']
    
    npzfile = np.load(datasetname + "_test_labels.npz")
    test_labels = npzfile['arr_0']

    return (train, train_labels), (test, test_labels)

