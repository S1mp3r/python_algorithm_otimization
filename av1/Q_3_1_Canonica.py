import numpy as np
import matplotlib.pyplot as plt

# Parâmetros do algoritmo genético
t = 0
A = 10
p = 20
N = 100
nd = 4
mutation_rate = 0.01
max_generations = 100
tolerance = 1e-6
no_change_generations = 10

x_l, x_u = -10, 10

# Função Rastrigin
def f_Rastrigin(x):
    return A * len(x) + sum([(xi**2 - A * np.cos(2 * np.pi * xi)) for xi in x])

# Função de aptidão
def f_fitness(x):
    return f_Rastrigin(x) + 1

def decode(cromossomos):
    max_val = 2**(nd*p) - 1
    decodificado = cromossomos.dot(2**np.arange(nd*p)[::-1])

    decoded_values = np.zeros((N, p)) #Feito para cada cromossomo separadamente
    
    for i in range(p):
        bits_for_variable = cromossomos[:, i*nd:(i+1)*nd]
        decodificado_var = bits_for_variable.dot(2**np.arange(nd)[::-1])
        decoded_values[:, i] = x_l + decodificado_var * (x_u - x_l) / (2**nd - 1)
    
    return decoded_values


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

# Plotagem da população
def plot_population(population, generation):
    decoded_pop = decode(population)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    #Plot so de 3d pq 20d eh loucura... (se eu plotar 1d e der um merge eh 20d??? fica ai o pensamento)
    ax.scatter(decoded_pop[:, 0], decoded_pop[:, 1], decoded_pop[:, 2], c='red', marker='o')
    ax.set_title(f'Population at Generation {generation}')
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_zlabel('X3')
    plt.show()


population = np.random.randint(0, 2, (N, nd*p))
best_fitness = np.inf
generations_no_change = 0

while t < max_generations:
    aptidao = f_fitness(decode(population))
    best_current_fitness = np.min(aptidao)
    
    if np.abs(best_fitness - best_current_fitness) < tolerance:
        generations_no_change += 1
    else:
        generations_no_change = 0
    
    best_fitness = min(best_fitness, best_current_fitness)
    
    if generations_no_change >= no_change_generations:
        break
    
    # Selecao
    indices = roulette(aptidao)
    pais = population[indices]
    
    # Cruzamento
    filhos = cross(pais)
    
    # Mutacao
    population = mutation(filhos)
    
    if t % 10 == 0:
        plot_population(population, t)

    t += 1

bp = 1