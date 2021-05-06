#################################################################################
# Universidade Federal do Rio de Janeiro
# Disciplina: Inteligência Artificial - EEL891
# Professor: Heraldo L. S. Almeida
# Desenvolvedor: Chritian Marques de Oliveira Silva
# DRE: 117.214.742
# Trabalho: Desafio das 8 rainhas utilizando Algoritimo Genético
#################################################################################

# PANORAMA GERAL
# 1) Serão 6 indivíduos a cada geração - cada indivídio será um tabuleiro
# 2) Um tabuleiro 8x8 com 8 rainhas
# 3) Cada rainha ataca em qualquer ponto no movimento em "+" e em "x" até o limite do tabuleiro 
# 4) Objetivo - Posicionar todas elas no tabuleiro de modo que nenhuma possa atacar outra 
# 5) Forma de quantificar o melhor:
#   5.1) Depois do posicionamento das rainhas, cada possibilidade de ataque será contado (se em uma linha poder ser mais de um, também será contado)
#   5.2) Os melhores indivíduos serão os com as menores pontuações (o objetivo é no final ter 0 ataques)
#   5.3) O mínimo de ataques é 0 (melhor disposição - nenhum se ataca) e o máximo é 56 (pior disposição - todos se atacam)
#   5.4) A inaptidão local seria a quantidade de ataques de um individuo, a inaptidão global é a quantidade de ataques da soma dos individuos
# 6) Cromossomo dos individuos: vetor 8x2 com a posição de cada rainha

# REGRAS
# 1) Crossover:
#   1.1) Escolha dos individuos (dentre os melhores) pais de cada novo individuo
#   1.2) Escolha de quais rainhas de cada par serão utilizadas
# 2) Mutação:
#   2.1) Caso haja repetição de linhas na escolha das rainhas do pior individuo da dupla, são inseridas novas rainhas (linhas que faltam e colunas aleatórias)
#   2.2) Caso estabilize X vezes com o mesmo número de ataques, cada individuo recebe N rainhas aleatórias
# 3) Seleção:
#   3.1) Elitismo - Os 4 menores ataques serão os melhores, os quais habilitam esses individuos para o Crossover

import random
import numpy as np

NUM_TAB = 6
MAX_LIN_COL = NUM_RAINHAS = 8
MAX_ESTABILIZACAO_ATK = 5
NUM_RAINHAS_MUTACAO = 2
# ALEATORIO = False
ALEATORIO = True

tabuleiro1 = np.array([
    [1, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 1]])

tabuleiro2 = np.array([
    [0, 0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 0]])

tabuleiro3 = np.array([
    [1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 1, 0, 0]])

tabuleiro4 = np.array([
    [0, 0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 1, 0],
    [1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0]])

tabuleiro5 = np.array([
    [0, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0]])

tabuleiro6 = np.array([
    [0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0],
    [0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 1, 0, 0, 1],
    [0, 0, 0, 0, 0, 1, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0]])

gabarito = np.array([
    [0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0],
    [1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1]])


def MostrarTabuleiro(tab):
    for linha in range(0, MAX_LIN_COL):
        for coluna in range(0, MAX_LIN_COL):
            print(tab[linha][coluna], ' ', end='')
        print('')

def MostrarTabuleiros(tabs):
    print("|     Tabela 1    |     Tabela 2    |     Tabela 3    |     Tabela 4    |     Tabela 4    |     Tabela 6    |")

    for linha in range(0, MAX_LIN_COL):
        for indice, tab in enumerate(tabs):
            print(end='| ')
            for coluna in range(0, MAX_LIN_COL):
                print(tab[linha][coluna], end='')
                if (coluna < MAX_LIN_COL):
                    print(end=' ')
            print(end='')
        print('|')
        
def MostrarResultadoGeracao(tabs, geracao):
    inaptidaoLocal = [0]*len(tabs)
    inaptidaoGlobal = ContarAtaquesGeracao(tabs)
    print()
    print("--------------------------------------------------------------------------------------------------------------")
    print("GERAÇÃO: {:05}                        INAPTIDÃO GLOBAL: {:02}/168".format(geracao, inaptidaoGlobal))
    print("--------------------------------------------------------------------------------------------------------------")
    MostrarTabuleiros(tabs)
    for i, tab in enumerate(tabs):
        inaptidaoLocal[i] = ContarAtaquesTabuleiro(tabs[i])
    print("|      {:02}/28      |      {:02}/28      |      {:02}/28      |      {:02}/28      |      {:02}/28      |      {:02}/28      |\n\n".format(
        inaptidaoLocal[0], inaptidaoLocal[1], inaptidaoLocal[2], inaptidaoLocal[3], inaptidaoLocal[4], inaptidaoLocal[5]))
    

# Tabela de valores retornados:
    # +------------------+
    # | Nº RAINHAS | ATK |
    # +------------------+
    # |     0      |  0  |
    # |     1      |  0  |
    # |     2      |  1  |
    # |     3      |  3  |
    # |     4      |  6  |
    # |     5      |  10 |
    # |     6      |  15 |
    # |     7      |  21 |
    # |     8      |  28 |
    # +------------------+
def CalcularNumeroAtaque(soma):
    return (soma*(soma - 1))//2

# Retorna o numero de ataques em um tabuleiro
def ContarAtaquesTabuleiro(tab):
    atk = 0
    # Ataques lineares: +
    for elemento in range(0, MAX_LIN_COL):
        atk += CalcularNumeroAtaque(tab[elemento, :].sum()) # Horizontal
        atk += CalcularNumeroAtaque(tab[:, elemento].sum()) # Vertical
    # Ataques inclinados: x
    for indice in range(0, MAX_LIN_COL):
        # Cima-esquerda para baixo-direita 
        atk += CalcularNumeroAtaque(np.diag(tab, k=indice).sum()) # Diagonal central e acima
        if (indice != 0):
            atk += CalcularNumeroAtaque(np.diag(tab, k=-indice).sum()) # Abaixo da diagonal central
        # Cima-direita para baixo-esquerda
        atk += CalcularNumeroAtaque(np.diag(np.flip(tab, axis=1), k=indice).sum()) # Diagonal central e acima
        if (indice != 0):
            atk += CalcularNumeroAtaque(np.diag(np.flip(tab, axis=1), k=-indice).sum()) # Abaixo da diagonal central
    return atk

def ContarAtaquesGeracao(tabs):
    atk = 0
    for indice, tab in enumerate(tabs):
        atk += ContarAtaquesTabuleiro(tab)
    return atk

def ObterPosicoesRainhas(tab):
    posicoes = np.array(np.where(tab == 1)).T
    return posicoes

# random.randint(0, NUM_RAINHAS)

if __name__ == "__main__":

    if (not ALEATORIO):
        random.seed(0)

    tabs = np.array([tabuleiro1, tabuleiro2, tabuleiro3, tabuleiro4, tabuleiro5, tabuleiro6])

    MostrarResultadoGeracao(tabs, 1)
    