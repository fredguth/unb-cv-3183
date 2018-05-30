

Objetivo
O objetivo desta atividade é propor e avaliar um método de reconhecimento de objetos para a base de imagens CalTech101[caltech101].


Introdução
Reconhecimento de Objetos lida com a identificação de objetos em imagens. Quando humanos vêem uma imagem, podem facilmente identificar objetos, pessoas, lugares;algo tão natural para nós é uma tarefa bastante complexa para um algoritmo e, até pouco tempo, com resultados nada animadores.

O momento crucial no crescimento meteórico do interesse por deep learning se deu em 2012, justamente na maior competição de reconhecimento de objetos, a  ImageNet Large Scale Visual Recognition Challenge  (ILSVRC)[goodfellow]. Com o paper "ImageNet Classification with Deep Convolutional Neural Networks", Alex Krizhevsky et al foram os primeiros a usar redes neurais convolucionais profundas (RNCs) na competição e ganharam por larga margem. Desde então, técnicas baseadas em RNCs tem sido as mais bem sucedidas para este problema.

Redes Neurais Convolucionais Profundas

Redes Neurais Convolucionais são modelos computacionais inspirados na biologia do cortex visual. O cortex visual tem pequenas regiões de células sensíveis a regiões específicas do campo visual. [hinton]

A ideia é resolver o problema da representação do conhecimento introduzindo representações que são expressas em termos de outras representações mais simples[goodfellow]. Em essência, um RNC é apenas uma função matemática que mapeia um conjunto de valores de entrada a valores de saída, formada pela composição de várias funções mais simples.  Com um número suficiente de composições, camadas, é possível obter funções de alta complexidade[hinton, goodfellow]. <imagem convolutional>

Em tarefas de classificação, as camadas amplificam características da entrada que são importantes para discriminação das amostras e suprime variações irrelevantes. Uma imagem, entra como um tensor de valores de pixel,  a primeira camada tipicamente representam a presença ou ausência de bordas em determinadas orientações e localizações na imagem. A segunda, detecta padões simples e arranjos de bordas. A terceira pode compor os padrões simples em combinações maiores que correspondem com partes de objetos, as camadas subsequentes irão detectar objetos como combinações dessas partes. [hinton]

Transferência de Conhecimento

Em nosso dia a dia, transferimos conhecimento a todo momento. Aprender a tocar piano, facilita aprender tocar órgão. Reconhecer maçãs talvez ajude a reconhecer peras. Pessoas conseguem inteligentemente aplicar conhecimento prévio para resolver novos problemas com maior eficácia e eficiência.[Sinno] E algoritmos?

Pesquisa em transferêcia de conhecimento tem atraído mais e mais atenção desde 1995, quando foi tema de  um workshop na NIPS-95 que discutiu a necessidade de métodos de aprendizado de máquina que retém e reusam conhecimento previamente obtido[Sinno]. 

No contexto de RCPs para reconhecimento visual de objetos, fica claro que as camadas iniciais: capazes de reconhecer bordas, padrões e partes de objetos, treinadas com um conjunto de imagens para reconhecer determinados rótulos, pode ser usada em outro conjunto de imagens totalmente diferente ainda que os rótulos também sejam diferentes. A capacidade de utilizar os pesos resultantes de treinamentos com milhões de imagens representa uma grande economia de processamento e uma ferramenta muito útil. 


Hiperparâmetros e Ajuste fino

O aspecto fundamental do deep learning é que as camadas de características não são extraídas manualmente por pessoas; elas são aprendidas a partir dos dados usando procedimentos de aprendizados genéricos. Como consequência, RCPs podem ser retreinadas para diferentes tarefas de reconhecimento e classificação, permitindo-se aproveitar redes pré-existentes. <imagem ml vs dl>

Para isso, entretanto, é preciso ajustar a rede para o problema em questão. Esse ajuste é obtido variando hiperparâmetros: learning rate, número de épocas, tamanho do batch, função de ativação, inicialização, dropout, etc. De certa forma, pode-se pensar que até a arquitetura utilizada (ResNet, Inception, LeNet, etc) é um parâmetro.  

É interessante notar que deep learning é epistemologicamente muito mais próximo das ciências naturais do que do resto da Ciência da Computação.  Em deep learning o resultado empírico é crucial, o ajuste de hiperparâmetros pode ter tanto ou mais valor do que o desenvolvimento de novas arquiteturas e a criatividade em pensar formas de visualizar o resultados pode levar a novos insights. Sendo Ciência da Computação acostumada a presar o método dedutivo antes de tudo, tais características geram desconfiança e estranheza.  Por outro lado, abre-se caminho para novas pessoas, com outras formas de pensar.

A Base Caltech101

Caltech 101 is a data set of digital images created in September 2003 and compiled by Fei-Fei Li, Marco Andreetto, Marc 'Aurelio Ranzato and Pietro Perona at the California Institute of Technology[wikipedia]. It is intended to facilitate Computer Vision research and techniques and is most applicable to techniques involving image recognition classification and categorization. Caltech 101 contains a total of 9,146 images, split between 101 distinct object categories (faces, watches, ants, pianos, etc.) and a background category. Provided with the images are a set of annotations describing the outlines of each image

Advantages
Caltech 101 has several advantages over other similar data sets:

Uniform size and presentation:
Almost all the images within each category are uniform in image size and in the relative position of interest objects. 
Low level of clutter/occlusion:

Weaknesses
Weaknesses to the Caltech 101 data set[3][14] may be conscious trade-offs, but others are limitations of the data set. Papers that rely solely on Caltech 101 are frequently rejected.

Weaknesses include:

The data set is too clean: Images are very uniform in presentation, aligned from left to right, and usually not occluded. As a result, the images are not always representative of practical inputs that the algorithm might later expect to see. Under practical conditions.unrealistic.

Limited number of categories:

Some categories contain few images:
Certain categories are not represented as well as others, containing as few as 31 images.
This means that {\displaystyle \mathrm {N} _{\mathrm {train} }\leq 30} \mathrm{N}_{\mathrm{train}} \le 30. The number of images used for training must be less than or equal to 30, which is not sufficient for all purposes.


Método

O método de reconhecimento de objetos desenvolvido nesse projeto foi fortemente influenciado por [fastai] e constitui-se das seguintes etapas:
1. Definir arquitetura RCP e obter rede pré-treinada com imagens da base ImageNet
2. Aumentar ao máximo a base de dados
  - Cross Validation
  - Transformação de Imagens
3. Otimização da Taxa de Aprendizado (Learning Rate)
4. Otimização em Tempo de Teste
  - Test-Time Augmentation
  - Test-Time model average


Aumento do conjunto de dados

Para um modelo de deep learning generalizar melhor, é preciso treinar com mais dados. Na prática, entretanto,os dados que temos são limitados. No presente projeto, por exemplo, temos apenas 5 imagens de treinamento por classe (rótulo) e não há conjunto definido de validação, ou seja, deveremos separar um conjunto de validação a partir da pequena base de treinamento.  

Tentamos contornar esse problema de duas formas:
Validação Cruzada
Validação Cruzada (Cross Validation) é uma técnica para criar uma base de validação a partir da base de treinamento. Em uma validação cruzada k-fold, a amostra original é aleatoriamente particionada em subamostra em k partes de igual tamanho, mutualmente exclusivas.  Depois do particionamento, uma das subamostras é utilizada para teste e as demais k-1 para treinamento. Ao final de k interações temos 5 modelos treinados em bases diferentes.
<imagem>

Transformação de imagens
Uma outra forma de contornar o problema de falta de dados é adcionar dados falsos no conjunto de treinamento com características que sabemos serem similares aos dados reais. Para diversos problemas gerar dados falsos não é uma abordagem viável. Mas em problemas visuais, sabemos que podemos aplicar operações de translação, rotação, escala e alteração cromática em imagens e melhorar os resultados do treinamento.
<imagem>

Otimização da Taxa de Aprendizado (Learning Rate)
Modelos de RCPs são normalmente treinados usando um otimizador por gradiente descendente estocástico (Stochastic Gradient Descent ou SGD). Há diversos tipos de otimizadores: Adam, RMSProp, Adagrad, etc. Todos permitem a definição da taxa de aprendizado (learning rate), que é o quanto o otimizador deve mover os pesos na direção do gradiente para um determinado mini-batch.

Com uma taxa pequena, o treinamento é mais confiável, mas exige mais tempo e processamento para chegar a um valor mínimo da função de perda. Se a taxa é maior, anda-se a "passos mais largos", mas o treinamento pode não convergir ou até divergir. 

Para nos guiar na decisão da taxa mais adequada para o nosso problema, usamos a técnica descrita em [Cyclical]: começa-se a treinar a rede com uma taxa de aprendizado bem baixa e a aumentamos exponencialmente a cada batch. 

Na figura ?? vemos que de início a função de perda melhora muito lentamente, e depois começa a acelerar a partir de xxx, até que por volta de ... a taxa de aprendizado se torna grande demais e o treinamento diverge, ou seja, a perda começa a aumentar. 

A ideia é selecionar o ponto no gráfico com a perda mais rápida. No nosso exemplo, a perda diminui rapidamente entre as taxas de treinamento .... e ..... Selecionar uma boa taxa inicial para o treinamento é apenas o começo, a ideia mais interessante da técnica é como se altera a taxa de treinamento durante o treinamento.  O mais comum é definir uma taxa de decaimento, mas fugindo ao senso comum, [cyclical] sugere uma variação cíclica em que a taxa de aprendizado pode sofrer aumentos repentinos. A figura xx mostra um gráfico com a função de variação da taxa de decaimento que usamos em nossos treinamentos.

A justificativa é facilmente entendida na figura xx do próprio artigo.  A maneira convencional nos ajuda a chegar em um mínimo da função de perda que pode ser local. Já usando uma variação cíclica, podemos chegar a vários mínimos diferentes, permitindo-se até obter um mínimo global, uma ideia que, apesar do artigo não mencionar, remonta a outra mais antiga, Otimização por Recozimento Simulado (Simulated Annealing Optimization)[anneling].






Resultados

Arquitetura
Otimizador
Função de Perda
Métrica

-

MACHINE TYPE: GPU+ HOURLY
REGION: NY2
PRIVATE IP: 
10.64.63.8
COPIED
PUBLIC IP: 
184.105.217.44
COPIED
RAM: 30 GB
CPUS: 8
HD: 107.5 KB / 50 GB
GPU: 8 GB
NETWORK: PAPERSPACE




A Survey on Transfer Learning
Sinno Jialin Pan and Qiang Yang Fellow, IEEE