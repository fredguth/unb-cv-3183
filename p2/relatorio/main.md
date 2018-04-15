main.md

#Introdução

Uma câmera é um instrumento de aquisição de imagens. Conhecendo seus parâmetros intrínsecos, como distância focal e distorção da lente, e extrínsecos, sua rotação e translação no sistema de coordenadas do mundo real, é possível estimar a posição 3D de um objeto a partir de sua imagem[tese], o que possibilita diversas aplicações: por exemplo, a mensuração da altura de pessoas registradas em vídeos de camêras de segurança ou a estimativas de posições de atletas em campo, entre outras.

##Objetivos
Os objetivos deste projeto são a aplicação prática da teoria de calibração de câmeras e o desenvolvimento de uma "régua visual", capaz de medir um objeto através da sua imagem.

 Mais especificamente deseja-se que sejam desenvolvidos programas usando a biblioteca OpenCV capazes de:

1) medir um segmento de reta em imagens através de cliques de mouse
2) realizar a calibração de uma câmera digital, armazenando os parâmetros intrísecos e os coeficientes de distorções em arquivos XML.
3) realizar a calibração de uma câmera digital a partir de diferentes distâncias da câmera, calculando os parâmetros extrínsecos da mesma e avaliando a diferença dos resultados
4) Com os parâmetros intrísecos e extrínsecos conhecidos, medir um objeto através de sua imagem e comparar com suas dimensões reais
5) Analisar os resultados obtidos

#Revisão Teórica

##Modelo de Câmera Estenopeica com Coordenadas Homogêneas
O modelo de câmera estenopeica (pinhole) faz um mapeamento geométrico do mundo 3D para o plano da imagem 2D.[Unicamp]
imagem, f, alpha, px, py [Qut]

Se os pontos do mundo (X) e da imagem (x) são representados por coordenadas homogêneas, podemos expressar matematicamente a projeção da câmera como uma matriz[Tese]:

lambdax = PX,

onde lambda é um fator de escala e P é a matriz 3x4 de projeção, também chamada matriz de calibração.

Sendo X coordenadas euclidianas, P pode ser decomposto em duas entidades geométricas: os parâmetros intrísecos e extrísecos de calibração [tese]

P = K(Rt), onde t é -R . C~ (2) [livro]

Os parâmetros intrísecos de calibração descrevem a transformação entre a imagem ideal e a imagem em pixels

K = (fI|po);

e os extrínsecos são a rotação e translação que transformam pontos no espaço do objeto para pontos no espaço da imagem e vice-versa. [tese]

Como há 6 graus de liberdade nos parâmetros extrínsecos e 5 nos intrísecos, é necessário pelo menos 6 correspondências {xi <-> Xi} do mesmo ponto no espaço da imagem e no espaço do objeto para obter P [tese]. 

Como há um erro inerente nas medidas experimentais, para melhorar a qualidade da estimativa é preciso usar n > 6 correspondências (como será visto na seção ..., usaremos 48). Como não há uma única matriz P que resolve esse sistema de equações é adcionar restrições.  

Um método comum é adicionar a restrição p34 = 0[tese, livro], mas essa abordagem não garante que não existam configurações em que o resultado com a restrição adicional é degenerado. Uma melhor melhor abordagem[tese] é fazer:
P = arg min....
onde d(xi ,P'Xi) é a distancia euclidiana entre o ponto observado e o estimado.

A biblioteca OpenCV usa essa última abordagem e aplica o método Levenberg-Marquant para resolver a minimização. 
## Parâmetros extrínsecos em função dos intrísecos e da matriz de calibração

Os parâmetros extrínsecos podem ser calculados a partir do conhecimento de P, K.  Isso será útil na seção [Metodologia] e, portanto, descrevemos aqui um método.

A partir da decomposição da matriz de calibração da câmera já descrita em (2), temos
x = KRt [X Y Z 1].T = K [ r1 r2 r3 t] [X Y Z 1].T
onde t = -RC~ e ri é a i-ésima coluna da matriz R.

:. P = K[r1 r2 r3 t] = [p1 p2 p3]
1/lambda K-1 [p1 p2 p3] = [r1 r2 r3 t]
r1 = 1/lambda K -1 p1
r2 = 1/lambda K -1 p2
r3 = r1 x r2
t = 1/lambda K -1 p3
[Unicamp]

onde lambda = ||K -1 p1 || = ||K -1 p2||

##distorção radial e tangencial

O modelo até aqui descrito descreve uma câmera ideal, mas as lentes das câmeras reais podem gerar distorções.  Essas distorções também são parâmetros intrínsecos que precisam ser considerados. 

A distorção radial causa uma curvatura no mapeamento de retas[unicamp].
imagem curva -> reta

A correção dessa distorção pode ser modelada da seguinte maneira: 
.. math::

    x_{corrected} = x( 1 + k_1 r^2 + k_2 r^4 + k_3 r^6) \\
    y_{corrected} = y( 1 + k_1 r^2 + k_2 r^4 + k_3 r^6)

Outra distorção comum é a tangencial, que ocorre quando o plano da lente não está alinhado perfeitamente em paralelo ao plano da imagem. Para corrigir:

.. math::

    x_{corrected} = x + [ 2p_1xy + p_2(r^2+2x^2)] \\
    y_{corrected} = y + [ p_1(r^2+ 2y^2)+ 2p_2xy]


Esses cinco parâmetros são conhecidos como coeficientes de distorção:

.. math::

    Coeficientes  \; de \; distorção=(k_1 \hspace{10pt} k_2 \hspace{10pt} p_1 \hspace{10pt} p_2 \hspace{10pt} k_3)

[opencv-câmera calibration]

#Metodologia
O modelo da câmera estenopeica e seus parâmetros foram descritos na seção . Nesta seção, descrevem-se como estimá-los experimentalmente.

##Materiais
Foram utilizados o seguintes materiais:
Uma tábua de compensado de WW x HH;
Papel contact
Um padrão de calibração xadrez impresso em papel A4
Uma trena
Uma régua
Computador MacBook Pro (Retina, 13-inch, Early 2015), Processador Intel Core i5 2,7 GHz, 8GB de RAM
- Python 3.6.3 :: Anaconda custom (64-bit)
- OpenCV 3.4.0


##Mensuração de segmentos de pixels em imagens
Desenvolveu-se um programa que simplesmente abre uma imagem jpg e captura cliques do mouse formando uma linha entre o primeiro e o segundo clique. 

Calcula-se também a distância:
||p2 - p1||_2

onde  p1 e p2 são vetores que representam os pontos obtidos.  A norma L2 é calculada usando a função np.linalg.norm.

##Obtenção dos parâmetros intrínsecos
planos
z =0
xadrez 
- captura de imagens
- estimativa dos parâmetros
-- opencv
##Obtenção dos parâmetros extrínsecos

##Recuperação de Poses 3D

###Algoritmo
1. O primeiro passo da calibração consiste na captura de diversas imagens de um padrão de calibração, como o onipresente tabuleiro de xadrez, em diversas orientações e posições. 

2. Os cantos dos quadrados são detectados usando OpenCV (findChessboardCorners) e apenas com esses pontos já é possível obter os parâmetros intrínsecos K e os coeficientes de calibração k1, k2, p1, p2, utilizando a função calibrateCamera da openCV.


3. Dado que já temos os coeficientes de distorção, retificamos as imagens da câmera e computamos a correspondencia entre pontos da imagem e do espaço do objeto, atribuindo como origem do sistema de coordenadas do mundo, o ponto de intersecção do canto superior esquerdo.

4. Com os pares de pontos correpondentes da imagem e do mundo, estimam-se os parâmetros extrínsecos da câmera da seguinte forma:
