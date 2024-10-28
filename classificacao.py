import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Funções para os classificadores

def gaussian_density(x, mean, cov):
    size = len(mean)
    det_cov = np.linalg.det(cov)
    if det_cov == 0:
        det_cov += 1e-6
    norm_const = 1.0 / (np.power((2 * np.pi), size / 2) * np.sqrt(det_cov))
    x_mu = x - mean
    inv_cov = np.linalg.inv(cov)
    result = norm_const * np.exp(-0.5 * (x_mu @ inv_cov @ x_mu.T))
    return result

def classificador_gaussiano(X, means, covariances):
    predictions = []
    for x in X:
        probabilities = [gaussian_density(x, mean, cov) for mean, cov in zip(means, covariances)]
        predictions.append(np.argmax(probabilities) + 1)
    return np.array(predictions)

def classificador_gaussiano_reg(X, means, cov):
    predictions = []
    for x in X:
        probabilities = [gaussian_density(x, mean, cov) for mean in means]
        predictions.append(np.argmax(probabilities) + 1)
    return np.array(predictions)

def mqo(X, y):
    return np.linalg.pinv(X.T @ X) @ X.T @ y

def gaussian_density_naive(x, mean, var):
    norm_const = 1.0 / np.sqrt(2 * np.pi * var)
    return norm_const * np.exp(-0.5 * ((x - mean) ** 2) / var)

def classificador_naive_bayes(X, means, variances):
    predictions = []
    for x in X:
        probabilities = [
            np.prod([gaussian_density_naive(x[i], mean[i], var[i]) for i in range(len(mean))])
            for mean, var in zip(means, variances)
        ]
        predictions.append(np.argmax(probabilities) + 1)
    return np.array(predictions)

def calcula_estatisticas(acuracias):
    media = np.mean(acuracias)
    desvio_padrao = np.std(acuracias)
    valor_max = np.max(acuracias)
    valor_min = np.min(acuracias)
    return media, desvio_padrao, valor_max, valor_min

data = np.loadtxt("EMGsDataset.csv", delimiter=',')
data = data.T

N = 50000
p = 2
c = 5 

X = data[:, :2]
Y = data[:, 2].astype(int)

classes = [1, 2, 3, 4, 5]
labels = ['Neutro', 'Sorriso', 'Sobrancelhas levantadas', 'Surpreso', 'Rabugento']
colors = ['green', 'blue', 'orange', 'red', 'teal']

for c_label, label, color in zip(classes, labels, colors):
    plt.scatter(X[Y == c_label, 0], X[Y == c_label, 1], color=color, edgecolor='k', label=label, alpha=0.5)
plt.legend()
plt.xlabel('Sensor 1')
plt.ylabel('Sensor 2')
plt.title('Visualização inicial dos dados')
plt.show()

# Simulações por Monte Carlo

R = 20
lambda_regs = [0, 0.25, 0.5, 0.75, 1]

accuracies_mqo = []
accuracies_gaussian = []
accuracies_gaussian_equal_cov = []
accuracies_gaussian_aggregated_cov = []
accuracies_naive_bayes = []
accuracies_gaussian_regularized = {lambda_reg: [] for lambda_reg in lambda_regs}

for _ in range(R):

    indices = np.random.permutation(N)
    split_point = int(N * 0.8)
    train_indices = indices[:split_point]
    test_indices = indices[split_point:]
    
    X_train = X[train_indices, :]
    Y_train = Y[train_indices]
    X_test = X[test_indices, :]
    Y_test = Y[test_indices]
    
    # MQO Tradicional
    Y_train_one_hot = np.zeros((Y_train.size, c))
    Y_train_one_hot[np.arange(Y_train.size), Y_train - 1] = 1
    
    # bias
    X_train_bias = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
    X_test_bias = np.hstack((np.ones((X_test.shape[0], 1)), X_test))
    
    # treino
    W = mqo(X_train_bias, Y_train_one_hot)
    Y_pred_mqo = X_test_bias @ W
    Y_pred_labels_mqo = np.argmax(Y_pred_mqo, axis=1) + 1
    accuracy_mqo = np.mean(Y_pred_labels_mqo == Y_test)
    accuracies_mqo.append(accuracy_mqo)
    
    means = []
    covariances = []
    variances = []  # Naive Bayes
    epsilon = 1e-6  # regularizacao
    
    for i in range(1, c + 1):
        class_data = X_train[Y_train == i, :]
        mean = np.mean(class_data, axis=0)
        cov = np.cov(class_data.T) + epsilon * np.eye(p)
        var = np.var(class_data, axis=0) + epsilon  # Naive Bayes
        means.append(mean)
        covariances.append(cov)
        variances.append(var)
    

    # Classificador Gaussiano Tradicional
    predictions_gaussian = classificador_gaussiano(X_test, means, covariances)
    accuracy_gaussian = np.mean(predictions_gaussian == Y_test)
    accuracies_gaussian.append(accuracy_gaussian)
    
    # Classificador Gaussiano com Covariâncias Iguais
    shared_cov = np.cov(X_train.T) + epsilon * np.eye(p)
    predictions_gaussian_equal_cov = classificador_gaussiano_reg(X_test, means, shared_cov)
    accuracy_gaussian_equal_cov = np.mean(predictions_gaussian_equal_cov == Y_test)
    accuracies_gaussian_equal_cov.append(accuracy_gaussian_equal_cov)
    
    # Classificador Gaussiano com Matriz Agregada
    total_samples = X_train.shape[0]
    aggregated_cov = sum(
        covariances[i] * (X_train[Y_train == (i + 1)].shape[0] / total_samples)
        for i in range(c)
    )
    predictions_gaussian_aggregated_cov = classificador_gaussiano_reg(X_test, means, aggregated_cov)
    accuracy_gaussian_aggregated_cov = np.mean(predictions_gaussian_aggregated_cov == Y_test)
    accuracies_gaussian_aggregated_cov.append(accuracy_gaussian_aggregated_cov)
    
    # Classificador de Bayes Ingênuo
    predictions_naive_bayes = classificador_naive_bayes(X_test, means, variances)
    accuracy_naive_bayes = np.mean(predictions_naive_bayes == Y_test)
    accuracies_naive_bayes.append(accuracy_naive_bayes)
    
    # Classificador Gaussiano Regularizado (Friedman)
    for lambda_reg in lambda_regs:
        regularized_covariances = [
            (1 - lambda_reg) * covariances[i] + lambda_reg * aggregated_cov for i in range(c)
        ]
        predictions = classificador_gaussiano(X_test, means, regularized_covariances)
        accuracy = np.mean(predictions == Y_test)
        accuracies_gaussian_regularized[lambda_reg].append(accuracy)



estatisticas_mqo = calcula_estatisticas(accuracies_mqo)
estatisticas_gaussian = calcula_estatisticas(accuracies_gaussian)
estatisticas_gaussian_equal_cov = calcula_estatisticas(accuracies_gaussian_equal_cov)
estatisticas_gaussian_aggregated_cov = calcula_estatisticas(accuracies_gaussian_aggregated_cov)
estatisticas_naive_bayes = calcula_estatisticas(accuracies_naive_bayes)

estatisticas_gaussian_regularized = {}
for lambda_reg in lambda_regs:
    estatisticas_gaussian_regularized[lambda_reg] = calcula_estatisticas(accuracies_gaussian_regularized[lambda_reg])





# Apresentação dos resultados em uma tabela
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

for lambda_reg in lambda_regs:
    modelos.append(f'Classif. Gaussiano Regularizado ({lambda_reg})')
    stats = estatisticas_gaussian_regularized[lambda_reg]
    medias.append(stats[0])
    desvios.append(stats[1])
    maiores.append(stats[2])
    menores.append(stats[3])




# Criação do DataFrame com os dados da tabela
df_resultados = pd.DataFrame({
    'Modelos': modelos,
    'Média': medias,
    'Desvio Padrão': desvios,
    'Maior Valor': maiores,
    'Menor Valor': menores
})

df_resultados['Média'] = df_resultados['Média'].map('{:.3f}'.format)
df_resultados['Desvio Padrão'] = df_resultados['Desvio Padrão'].map('{:.3f}'.format)
df_resultados['Maior Valor'] = df_resultados['Maior Valor'].map('{:.3f}'.format)
df_resultados['Menor Valor'] = df_resultados['Menor Valor'].map('{:.3f}'.format)




fig, ax = plt.subplots(figsize=(10, len(modelos) * 0.5))
ax.axis('tight')
ax.axis('off')

table_data = df_resultados.values.tolist()
col_labels = df_resultados.columns

table = ax.table(cellText=table_data, colLabels=col_labels, loc='center', cellLoc='center', colWidths=[0.35, 0.08, 0.13, 0.11, 0.11])

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.2)
plt.title('Resultados dos Modelos de Classificação')
plt.show()