# GANs – Generative Adversarial Networks

# Objetivo:

Repositorio com estudos feito utilizando a rede GAN

Redes Adversárias Generativas (GANs) são arquiteturas de redes neurais profundas compostas por duas redes colocadas uma contra a outra (daí o nome “adversárias”).
Uma rede neural, chamada de gerador, gera novas instâncias de dados, enquanto a outra, o discriminador, avalia sua autenticidade; ou seja, o discriminador decide se cada instância de dados que ele analisa pertence ou não ao conjunto de dados de treinamento real (a imagem abaixo demonstra isso).

![](https://www.deeplearningbook.com.br/wp-content/uploads/2019/09/gan_schema.png)

# Instalação

## Clone the repository:

`git clone <repository-url>`

## Instalar as dependencias:

`pip install -r requirements.txt`

# Dataset:

https://www.kaggle.com/datasets/jessicali9530/celeba-dataset

## Uso

* Importe os módulos necessários e defina a classe FaceGan.

* Inicialize o objeto FaceGan. Isso configurará a semente aleatória, o dispositivo, o conjunto de dados, o carregador de dados e o registrador.

* Carregue o conjunto de dados e crie um carregador de dados usando o método load_dataset.

* Plote uma grade de imagens reais de treinamento usando o método plot_samples.

* Carregue os modelos gerador e discriminador e inicialize seus pesos usando o método load_models.

* Treine o modelo FaceGAN usando o método train_model. Esse método retorna listas de perdas do gerador e do discriminador, bem como uma lista de imagens geradas em intervalos específicos.

* Salve os modelos gerador e discriminador usando o método save_models.

* Carregue o modelo gerador a partir de um ponto de verificação salvo e gere amostras usando o método load_generator_model.

* Avalie o processo de treinamento exibindo as perdas do gerador e do discriminador durante o treinamento, além das imagens reais e falsas usando o método evaluate.

![erro_g_d](https://github.com/FelipeAmaral13/ProjetosDeepLearning/assets/5797933/4c03e64b-b964-41fa-8092-9bf098dd3fd6)
![evaluate](https://github.com/FelipeAmaral13/ProjetosDeepLearning/assets/5797933/88ea72e4-d9ae-48ec-8d9a-ee5c55c758d1)
