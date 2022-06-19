# Transfering Learning usando VGG19:

## Transfering Learning:
Transfering Learning é aproveitar o aprendizado de um modelo pré-treinado, para que você não precise treinar um novo modelo do zero, ou seja, pesos obtidos dos modelos podem ser reutilizados em outras tarefas de visão computacional.

Os modelos pré-treinados geralmente são treinados em grandes conjuntos de dados como Imagenet. 

Esses modelos podem ser usados diretamente na previsão de novas tarefas ou integrados ao processo de treinamento de um novo modelo. Incluir os modelos pré-treinados em um novo modelo leva a um menor tempo de treinamento e menor erro de generalização.

O aprendizado de transferência é particularmente muito útil quando você tem um pequeno conjunto de dados de treinamento. Nesse caso, você pode, por exemplo, usar os pesos dos modelos pré-treinados para inicializar os pesos do novo modelo. Como você verá mais adiante, o aprendizado por transferência também pode ser aplicado a problemas de processamento de linguagem natural.


## Modelo VGG19:

VGG é uma rede neural convolucional profundidade que possui 19 camadas. Foi construído e treinado por K. Simonyan e A. Zisserman na Universidade de Oxford em 2014. 
A rede VGG-19 é treinada usando mais de 1 milhão de imagens coloridas de 224x244 piexels do banco de dados ImageNet, por isso pode-se importar os pesos treinados do ImageNet. 
e pode-se classificar até 1000 objetos.

![0 E6BE6GDv-53smX0B](https://user-images.githubusercontent.com/5797933/174499699-2775b5b7-a175-45f5-8267-2c3ca479354e.jpg)

