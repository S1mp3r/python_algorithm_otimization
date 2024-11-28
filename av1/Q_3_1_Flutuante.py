import numpy as np
import matplotlib.pyplot as plt

# Parâmetros do algoritmo genético
t = 0
A = 10
p = 20
N = 100
mutation_rate = 0.01
max_generations = 100
tolerance = 1e-6
no_change_generations = 10
recombination_rate = 0.9  # Maior que 85% para cruzamento
eta_c = 2  # Parâmetro do SBX para cruzamento
eta_m = 20  # Parâmetro para mutação Gaussiana

x_l, x_u = -10, 10

# Função Rastrigin
def f_Rastrigin(x):
    return A * len(x) + sum([(xi**2 - A * np.cos(2 * np.pi * xi)) for xi in x])

# Função de aptidão
def f_fitness(x):
    return f_Rastrigin(x) + 1

# Função de seleção por torneio
def tournament_selection(aptidao, k=3):
    selected_indices = []
    for _ in range(N):
        tournament_indices = np.random.randint(0, N, k)
        best_idx = tournament_indices[np.argmin(aptidao[tournament_indices])]
        selected_indices.append(best_idx)
    return np.array(selected_indices)

# Recombinação via SBX
def sbx_crossover(pais, eta_c=2):
    filhos = np.empty_like(pais)
    for i in range(0, N, 2):
        if np.random.rand() < recombination_rate:
            parent1, parent2 = pais[i], pais[i+1]
            beta = np.empty_like(parent1)
            for j in range(len(parent1)):
                u = np.random.rand()
                if u <= 0.5:
                    beta[j] = (2 * u)**(1 / (eta_c + 1))
                else:
                    beta[j] = (1 / (2 * (1 - u)))**(1 / (eta_c + 1))
            child1 = 0.5 * ((1 + beta) * parent1 + (1 - beta) * parent2)
            child2 = 0.5 * ((1 - beta) * parent1 + (1 + beta) * parent2)
            filhos[i], filhos[i+1] = child1, child2
        else:
            filhos[i], filhos[i+1] = pais[i], pais[i+1]
    return np.clip(filhos, x_l, x_u)

# Mutação Gaussiana
def gaussian_mutation(populacao, eta_m=20):
    for i in range(N):
        if np.random.rand() < mutation_rate:
            mutacao = np.random.randn(p) * (x_u - x_l) / eta_m
            populacao[i] = np.clip(populacao[i] + mutacao, x_l, x_u)
    return populacao

# Plotagem da população
def plot_population(population, generation):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    #Plot só em 3d para simplificação
    ax.scatter(population[:, 0], population[:, 1], population[:, 2], c='red', marker='o')
    ax.set_title(f'Population at Generation {generation}')
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_zlabel('X3')
    plt.show()

# Inicialização da população
population = np.random.uniform(x_l, x_u, (N, p))
best_fitness = np.inf
generations_no_change = 0

# Critério de convergência e loop principal
while t < max_generations:
    aptidao = np.array([f_fitness(ind) for ind in population])
    best_current_fitness = np.min(aptidao)

    if np.abs(best_fitness - best_current_fitness) < tolerance:
        generations_no_change += 1
    else:
        generations_no_change = 0

    best_fitness = min(best_fitness, best_current_fitness)

    if generations_no_change >= no_change_generations:
        break

    # Seleção por torneio
    indices = tournament_selection(aptidao)
    pais = population[indices]

    # Recombinação via SBX
    filhos = sbx_crossover(pais, eta_c)

    # Mutação Gaussiana
    population = gaussian_mutation(filhos, eta_m)

    if t % 10 == 0:
        plot_population(population, t)

    t += 1

bp = 1