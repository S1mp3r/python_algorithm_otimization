import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import classificadores_Gaussianos

def calcula_estatisticas(acuracias):
    media = np.mean(acuracias)
    desvio_padrao = np.std(acuracias)
    valor_max = np.max(acuracias)
    valor_min = np.min(acuracias)
    return media, desvio_padrao, valor_max, valor_min

data = np.loadtxt("EMGsDataset.csv", delimiter=',').T

N = 50000
c = 5
p = 2

lambda_regs = [0, 0.25, 0.5, 0.75, 1]

accuracies_mqo = []
accuracies_gaussian = []
accuracies_gaussian_equal_cov = []
accuracies_gaussian_aggregated_cov = []
accuracies_naive_bayes = []
accuracies_gaussian_regularized = {lambda_reg: [] for lambda_reg in lambda_regs}

plt.scatter(data[data[:, 2] == 1, 0], data[data[:, 2] == 1, 1], color='green', edgecolor='k', label='Neutro')
plt.scatter(data[data[:, 2] == 2, 0], data[data[:, 2] == 2, 1], color='blue', edgecolor='k', label='Sorriso')
plt.scatter(data[data[:, 2] == 3, 0], data[data[:, 2] == 3, 1], color='orange', edgecolor='k', label='Sobrancelhas levantadas')
plt.scatter(data[data[:, 2] == 4, 0], data[data[:, 2] == 4, 1], color='red', edgecolor='k', label='Surpreso')
plt.scatter(data[data[:, 2] == 5, 0], data[data[:, 2] == 5, 1], color='teal', edgecolor='k', label='Rabugento')
plt.legend()
# plt.show()

X = data[:, :2]
Y = data[:, 2].astype(int)

rounds = 10 # Número de rodadas solicitado
for _ in range(rounds):
    # Gera os dados novos para a computação, os mesmos dados para cada algoritmo
    indices = np.random.permutation(N)
    split_point = int(N * 0.8)
    train_indices = indices[:split_point]
    test_indices = indices[split_point:]
    
    # X e Y de treinamento e de teste
    X_train = X[train_indices, :]
    Y_train = Y[train_indices]
    X_test = X[test_indices, :]
    Y_test = Y[test_indices]

    # Intercepto para o MQO
    X_train_bias = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
    X_test_bias = np.hstack((np.ones((X_test.shape[0], 1)), X_test))

    # Codificação one-hot para o MQO
    Y_train_one_hot = np.zeros((Y_train.size, c))
    Y_train_one_hot[np.arange(Y_train.size), Y_train - 1] = 1  # Subtrai 1 porque as classes começam em 1

    # MQO Tradicional
    W = classificadores_Gaussianos.MQO(X_train_bias, Y_train_one_hot)
    Y_pred_mqo = X_test_bias @ W
    Y_pred_labels_mqo = np.argmax(Y_pred_mqo, axis=1) + 1  # Soma 1 para corresponder aos rótulos originais
    bp =1

    accuracy_mqo = np.mean(Y_pred_labels_mqo == Y_test)
    accuracies_mqo.append(accuracy_mqo)

    # Classificador Gaussiano Tradicional
    predictions_gaussian_trad = classificadores_Gaussianos.classificador_Gaussiano_Trad(X_test, X_train, Y_train, c)
    accuracy_gaussian = np.mean(predictions_gaussian_trad == Y_test)
    accuracies_gaussian.append(accuracy_gaussian)
    bp =1
    # Classificador Gaussiano Com Covariâncias Iguais
    predictions_gaussian_cov_iguais = classificadores_Gaussianos.classificador_Gaussiano_Cov_Iguais(X_test, X_train, Y_train, c)
    accuracy_gaussian_equal_cov = np.mean(predictions_gaussian_cov_iguais == Y_test)
    accuracies_gaussian_equal_cov.append(accuracy_gaussian_equal_cov)

    # Classificador Gaussiano Com Matriz Agregada
    predictions_gaussian_agreg = classificadores_Gaussianos.classificador_Gaussiano_Matriz_Agregada(X_test, X_train, Y_train, c)
    accuracy_gaussian_aggregated_cov = np.mean(predictions_gaussian_agreg == Y_test)
    accuracies_gaussian_aggregated_cov.append(accuracy_gaussian_aggregated_cov)

    # Classificador Gaussiano Regularizado Friedman
    for lamb in lambda_regs:
        predictions_gaussian_lamb = classificadores_Gaussianos.classificador_Gaussiano_Friedman(X_test, X_train, Y_train, c, lamb)
        accuracy_gaussian_reg = np.mean(predictions_gaussian_lamb == Y_test)
        accuracies_gaussian_regularized[lamb].append(accuracy_gaussian_reg)
    
    # Classificador Gaussiano Naive Bayes
    predictions_gaussian_bayes = classificadores_Gaussianos.classificador_Gaussiano_Naive_Bayes(X_test, X_train, Y_train, c)
    accuracy_naive_bayes = np.mean(predictions_gaussian_bayes == Y_test)
    accuracies_naive_bayes.append(accuracy_naive_bayes)

# Cálculo das estatísticas para cada modelo
estatisticas_mqo = calcula_estatisticas(accuracies_mqo)
estatisticas_gaussian = calcula_estatisticas(accuracies_gaussian)
estatisticas_gaussian_equal_cov = calcula_estatisticas(accuracies_gaussian_equal_cov)
estatisticas_gaussian_aggregated_cov = calcula_estatisticas(accuracies_gaussian_aggregated_cov)
estatisticas_naive_bayes = calcula_estatisticas(accuracies_naive_bayes)
estatisticas_gaussian_regularized = {lamb: calcula_estatisticas(accuracies_gaussian_regularized[lamb]) for lamb in lambda_regs}

# Organizando os resultados em um DataFrame
modelos = [
    'MQO tradicional',
    'Classificador Gaussiano Tradicional',
    'Classificador Gaussiano (Cov iguais)',
    'Classificador Gaussiano (Cov Agregada)',
    'Classificador de Bayes Ingênuo',
]

medias = [
    estatisticas_mqo[0],
    estatisticas_gaussian[0],
    estatisticas_gaussian_equal_cov[0],
    estatisticas_gaussian_aggregated_cov[0],
    estatisticas_naive_bayes[0],
]

desvios = [
    estatisticas_mqo[1],
    estatisticas_gaussian[1],
    estatisticas_gaussian_equal_cov[1],
    estatisticas_gaussian_aggregated_cov[1],
    estatisticas_naive_bayes[1],
]

maiores = [
    estatisticas_mqo[2],
    estatisticas_gaussian[2],
    estatisticas_gaussian_equal_cov[2],
    estatisticas_gaussian_aggregated_cov[2],
    estatisticas_naive_bayes[2],
]

menores = [
    estatisticas_mqo[3],
    estatisticas_gaussian[3],
    estatisticas_gaussian_equal_cov[3],
    estatisticas_gaussian_aggregated_cov[3],
    estatisticas_naive_bayes[3],
]

# Ajuste na lista de lambdas para evitar duplicatas
lambda_regs_to_include = [lamb for lamb in lambda_regs if lamb not in [0, 1]]

# Adiciona os classificadores regularizados sem duplicatas
for lamb in lambda_regs_to_include:
    modelos.append(f'Classificador Gaussiano Regularizado ({lamb})')
    stats = estatisticas_gaussian_regularized[lamb]
    medias.append(stats[0])
    desvios.append(stats[1])
    maiores.append(stats[2])
    menores.append(stats[3])

# Criando o DataFrame com os resultados
df_resultados = pd.DataFrame({
    'Modelos': modelos,
    'Média': medias,
    'Desvio Padrão': desvios,
    'Maior Valor': maiores,
    'Menor Valor': menores
})

# Formatação para três casas decimais
df_resultados['Média'] = df_resultados['Média'].map('{:.3f}'.format)
df_resultados['Desvio Padrão'] = df_resultados['Desvio Padrão'].map('{:.3f}'.format)
df_resultados['Maior Valor'] = df_resultados['Maior Valor'].map('{:.3f}'.format)
df_resultados['Menor Valor'] = df_resultados['Menor Valor'].map('{:.3f}'.format)

# Configuração do gráfico para exibir a tabela visualmente
fig, ax = plt.subplots(figsize=(10, len(modelos) * 0.5))  # Ajuste do tamanho da figura conforme o número de modelos
ax.axis('tight')
ax.axis('off')

# Conversão do DataFrame para exibição com o matplotlib
table_data = df_resultados.values.tolist()
col_labels = df_resultados.columns

# Exibe a tabela com a largura personalizada das colunas
table = ax.table(cellText=table_data, colLabels=col_labels, loc='center', cellLoc='center', colWidths=[0.35, 0.08, 0.13, 0.11, 0.11])

# Ajuste do layout e exibição
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.2)
plt.title('Resultados dos Modelos de Classificação')
plt.show()

bp = 1