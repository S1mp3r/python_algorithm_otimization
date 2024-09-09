import numpy as np
import matplotlib.pyplot as plt

t = 0
A = 10
p = 20
N = 100
nd = 4
mutation_rate = 0.01

x_l, x_u = -10, 10

# Função Rastrigin
def f_Rastrigin(x):
    return A * p + np.sum(x**2 - A * np.cos(2 * np.pi * x), axis=-1)

def f_fitness(x):
    return f_Rastrigin(x) + 1

# Decodificação
def decode(cromossomos):
    max_val = 2**(nd*p) - 1

    decodificado = cromossomos.dot(2**np.arange(nd*p)[::-1])
    
    return x_l + decodificado * (x_u - x_l) / max_val


def roulette(aptidao):
    probabilidade = aptidao / np.sum(aptidao)

    soma_acumulada = np.cumsum(probabilidade)

    indices_selecionados = []

    for _ in range(N):
        r = np.random.uniform(0, 1)
        i = np.searchsorted(soma_acumulada, r)
        indices_selecionados.append(i)
    return np.array(indices_selecionados)


def cross(pais):
    pais1 = pais[:N//2]
    pais2 = pais[N//2:]
    
    mask = np.random.rand(N//2, nd*p) < 0.85
    
    filhos = np.zeros((N, nd*p), dtype=int)
    
    for i in range(N//2):
        filhos[i*2] = np.where(mask[i], pais1[i], pais2[i])
        filhos[i*2 + 1] = np.where(mask[i], pais2[i], pais1[i])
    
    return filhos


def mutation(populacao):
    mutacoes = np.random.rand(N, nd*p) < mutation_rate
    populacao_mutada = np.logical_xor(populacao, mutacoes).astype(int)
    return populacao_mutada




population = np.random.randint(0, 2, (N, nd*p))

while t < 100:
    aptidao = f_fitness(decode(population))
    
    # Seleção
    indices = roulette(aptidao)
    pais = population[indices]
    
    # Cruzamento
    filhos = cross(pais)
    
    population = mutation(filhos)
    
    t += 1

bp = 1