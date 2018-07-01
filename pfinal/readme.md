
## Desenvolvido em:
* Mac OSX 10.13.3 
* Python 3.6.3 :: Anaconda custom (64-bit)
* OpenCV 3.4.0

## Base de dados

###./data
* ./data/dataset: 14195 imagens divididas em diretorios com o nome da categoria (sku)
* ./data/boti: base estratificada em 9.084 imagens de treinamento (train), 2.272 de validação (valid) e 2.839 de teste (test)
* ./data/boti/mag_test: base de teste no domínio das revistas com 53 imagens de 45 categorias
* ./data/mags: base de imagens de revista
* ./data/videos: videos originais usados
* ./data/GT: base de groundtruth foreground/background usada para tentar remover background de imagens. 

###Programas:
Os programas boti-2.py, boti-3.py, boti-4.py e boti-4-b.py são implementações do classificador de objetos in-domain.
São uma evolução do classificador, portanto, utilizar apenas boti-4-b.py

* GTmaker.py: cria máscaras de foreground/background a partir das imagens
* balanceDataset.py: faz balanceamento de cor nas imagens
* green_screen.py: parecido com o PD1, segmenta cores próximas em imagens
* images2GTs.py: assim como o GTmaker, cria máscaras de foreground/background
* images2quants.py: cria imagens quantizadas com menos 64 cores apenas, usando k-means
* importMags.py: faz download das imagens a partir da base no S3
* movie2images.py: converte vídeos em datasets de imagens
* quants2mask.py: converte imagem quantizada em máscara
* split_dataset.py: faz divisão estratificada do dataset (./data/dataset) em bases de treinamento, validação e teste (./data/boti)

As bibliotecas necessárias estão em requirements.txt

Para rodar os programas:
`python nome_do_programa.py`

Gerar relatório

* pdflatex main.tex
* bibtex main
* pdflatex main.tex

Relatório:
* ./relatorio/main.pdf

