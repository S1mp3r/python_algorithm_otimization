import numpy as np
import matplotlib.pyplot as plt

# Função Rastrigin
def f_Rastrigin(x, A=10):
    return A * len(x) + sum([(xi**2 - A * np.cos(2 * np.pi * xi)) for xi in x])

# Função de aptidão
def f_fitness(x):
    return f_Rastrigin(x) + 1

def algoritmo_genetico_Nao_Canonico(N=100, p=20, max_generations=100, mutation_rate=0.01, recombination_rate=0.9, eta_c=2, eta_m=20, x_l=-10, x_u=10):
    def sbx_crossover(pais):
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

    def gaussian_mutation(populacao):
        for i in range(N):
            if np.random.rand() < mutation_rate:
                mutacao = np.random.randn(p) * (x_u - x_l) / eta_m
                populacao[i] = np.clip(populacao[i] + mutacao, x_l, x_u)
        return populacao

    def tournament_selection(aptidao, k=3):
        selected_indices = []
        for _ in range(N):
            tournament_indices = np.random.randint(0, N, k)
            best_idx = tournament_indices[np.argmin(aptidao[tournament_indices])]
            selected_indices.append(best_idx)
        return np.array(selected_indices)

    population = np.random.uniform(x_l, x_u, (N, p))
    best_fitness = np.inf
    t = 0
    generations_no_change = 0
    tolerance = 1e-6
    no_change_generations = 10

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

        # Selecao
        indices = tournament_selection(aptidao)
        pais = population[indices]

        # SBX
        filhos = sbx_crossover(pais)

        # Mutacao Gaussiana
        population = gaussian_mutation(filhos)

        t += 1

    return best_fitness



def algoritmo_genetico_Canonico(N=100, p=20, nd=4, max_generations=100, mutation_rate=0.01, x_l=-10, x_u=10):
    def decode(cromossomos):
        decoded_values = np.zeros((N, p)) 
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
        mask = np.random.rand(N//2, nd*p) < 0.85
        filhos = np.zeros((N, nd*p), dtype=int)
        for i in range(N//2):
            filhos[i*2] = np.where(mask[i], pais[i], pais[i + N//2])
            filhos[i*2 + 1] = np.where(mask[i], pais[i + N//2], pais[i])
        return filhos

    def mutation(populacao):
        mutacoes = np.random.rand(N, nd*p) < mutation_rate
        populacao_mutada = np.logical_xor(populacao, mutacoes).astype(int)
        return populacao_mutada

    population = np.random.randint(0, 2, (N, nd*p))
    best_fitness = np.inf
    t = 0
    generations_no_change = 0
    tolerance = 1e-6
    no_change_generations = 10

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

        t += 1

    return best_fitness



def comparar_algoritmos(num_rodadas=100):
    resultados_alg1 = []
    resultados_alg2 = []

    for _ in range(num_rodadas):
        resultados_alg1.append(algoritmo_genetico_Nao_Canonico())
        resultados_alg2.append(algoritmo_genetico_Canonico())



    def calcular_metricas(resultados):
        menor_valor = np.min(resultados)
        maior_valor = np.max(resultados)
        media = np.mean(resultados)
        desvio_padrao = np.std(resultados)
        return menor_valor, maior_valor, media, desvio_padrao

    metricas_alg1 = calcular_metricas(resultados_alg1)
    metricas_alg2 = calcular_metricas(resultados_alg2)



    print("Comparação entre os Algoritmos Genéticos:")
    print(f"{'Métrica':<20}{'Algoritmo 1':<20}{'Algoritmo 2':<20}")
    print(f"{'Menor Aptidão':<20}{metricas_alg1[0]:<20}{metricas_alg2[0]:<20}")
    print(f"{'Maior Aptidão':<20}{metricas_alg1[1]:<20}{metricas_alg2[1]:<20}")
    print(f"{'Média Aptidão':<20}{metricas_alg1[2]:<20}{metricas_alg2[2]:<20}")
    print(f"{'Desvio Padrão':<20}{metricas_alg1[3]:<20}{metricas_alg2[3]:<20}")

# Executar comparação
comparar_algoritmos()

bp = 1