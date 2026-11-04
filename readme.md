# Relações entre personagens de livros hospedados no Projeto Gutenberg

Este projeto baixa um livro diretamente do site do Projeto Gutenberg, e encontra os personagens no livro. Depois, gera um grafo capitulo por capitulo, e no final gera uma animação da evolução das relações, o grafo final e tabelas sobre as relações.

Precisa de Python 3.10 ou uma versão maior.

## Antes de usar
É necessario instalar as dependencias, se não estiverem instalados. Não é preciso ambiente virtual.
```
pip install -r dep.txt
```
Também é necessario colocar uma chave Groq.
```
set GROQ_API_KEY=sua_chave_aqui
```

## Como usar

```
py main.py
```
O programa pede:
1. O link do livro no Gutenberg, se nenhum for dado, ele automaticamente usa o Moby Dick.
2. Se o livro é em primeira pessoa. Se sim, é necessario informar o nome do personagem principal do livro.

Também é possivel diretamente passar o link no comando para pular o primeiro input.
```
py main.py https://www.gutenberg.org/cache/epub/2701/pg2701-images.html
```

A primeira vez que é executado, ele automaticamente instala o modelo spaCy (`en_core_web_sm`).

## Resultados
Salvos em `output/<nome_do_livro>/`:
- `grafo_final.png`: O grafo final das relações.
- `evolution.gif`: A evolução do grafo capitulo a capitulo, de forma animada.
- `evolution/`: Um PNG de cada relação por capitulo. Inclui relações de capitulos antigos.
- `capitulo.csv`: Resumo geral de relações por capítulo, incluindo numero de personagems detectados, numero de relações, novos personagems em cada capitulo, e a maior relação do capitulo.
- `relacoes.csv`: Pares de personagens, e o peso da relação entre eles.
- `personagens.csv`: Menções por personagem.
