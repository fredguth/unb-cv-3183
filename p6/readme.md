
Desenvolvido em:
* Mac OSX 10.13.3 
* Python 3.6.3 :: Anaconda custom (64-bit)
* OpenCV 3.4.0

As bibliotecas necessárias estão em requirements.txt

Para rodar rastreador KCF:
`python r3.py`
(é preciso alterar o código para mudar o arquivo de vídeo)

Para rodar o rastreador com filtro de Kalman:
`python r3.py`
é preciso alterar o código para mudar o arquivo de vídeo)

Para gerar as imagens do relatório:
`jupyter notebook stats.ipynb`

Para gerar estatísticas do relatório:
`python stats.py`

Gerar relatório

* pdflatex main.tex
* bibtex main
* pdflatex main.tex

Relatório:
* ./relatorio/main.pdf

Arquivos de dados
* Todos os arquivos .npy são utilizados por stats.py e stats.ipynb para gerar estatísticas para o relatório