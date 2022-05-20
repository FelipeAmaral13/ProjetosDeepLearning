# GANs – Generative Adversarial Networks

# Objetivo:

Repositorio com estudos feito utilizando a rede GAN

Redes Adversárias Generativas (GANs) são arquiteturas de redes neurais profundas compostas por duas redes colocadas uma contra a outra (daí o nome “adversárias”).
Uma rede neural, chamada de gerador, gera novas instâncias de dados, enquanto a outra, o discriminador, avalia sua autenticidade; ou seja, o discriminador decide se cada instância de dados que ele analisa pertence ou não ao conjunto de dados de treinamento real (a imagem abaixo demonstra isso).

![](https://www.deeplearningbook.com.br/wp-content/uploads/2019/09/gan_schema.png)

Digamos que estamos tentando fazer algo mais banal do que imitar a Mona Lisa. Geraremos números escritos à mão, como os encontrados no conjunto de dados MNIST, retirado do mundo real. O objetivo do discriminador, quando mostrada uma instância do verdadeiro conjunto de dados MNIST, é reconhecer aqueles que são autênticos.

Enquanto isso, o gerador está criando novas imagens sintéticas que são transmitidas ao discriminador. O gerador gera as imagens fake na esperança de que elas também sejam consideradas autênticas, mesmo sendo falsas. O objetivo do gerador é gerar dígitos manuscritos cada vez melhores. O objetivo do discriminador é identificar imagens falsas do gerador. Ou seja, são duas redes adversárias, uma discriminativa (padrão que já estudamos até aqui no livro) e uma generativa que, em termos gerais, faz o oposto das redes discriminativas.

Aqui estão as etapas de uma GAN:

    * O gerador considera números aleatórios e retorna uma imagem.
    * Essa imagem gerada é inserida no discriminador ao lado de um fluxo de imagens tiradas do conjunto de dados real e verdadeiro.
    * O discriminador obtém imagens reais e falsas e retorna probabilidades, um número entre 0 e 1, com 1 representando uma previsão de imagem autêntica e 0 representando previsão de imagens falsas (geradas pela rede generativa).

![Fonte](https://www.deeplearningbook.com.br/introducao-as-redes-adversarias-generativas-gans-generative-adversarial-networks/)
