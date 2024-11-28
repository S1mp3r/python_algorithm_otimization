import numpy as np
import matplotlib.pyplot as plt

def calcula_estatisticas(rss_list):
    # Calcula média, desvio padrão, valor máximo e mínimo
    media = np.mean(rss_list)
    desvio_padrao = np.std(rss_list)
    valor_max = np.max(rss_list)
    valor_min = np.min(rss_list)
    return media, desvio_padrao, valor_max, valor_min

def mqo(X, y):
    return np.linalg.pinv(X.T @ X) @ X.T @ y

def mqo_regularizado(X, y, lambd):
    I = np.eye(X.shape[1])
    return np.linalg.pinv(X.T @ X + lambd * I) @ X.T @ y

def MQO_Default(data, rounds):
    rss_mqo = []
    rss_media = []

    # Prepara os dados para regressão
    X_raw = data[:, 0]
    y = data[:, 1]

    plt.scatter(X_raw, y, color='blue')
    plt.title('Visualização inicial dos dados')
    plt.xlabel('Velocidade do Vento')
    plt.ylabel('Potência Gerada')
    plt.show()

    # Adiciona o termo de intercepto na matriz X
    X = np.vstack([np.ones(len(X_raw)), X_raw]).T
    y = y.reshape(-1, 1)

    for _ in range(rounds):
        indices = np.random.permutation(len(X))
        split = int(len(X) * 0.8)
        train_idx, test_idx = indices[:split], indices[split:]
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Modelo MQO
        beta_mqo = mqo(X_train, y_train)
        y_pred_mqo = X_test @ beta_mqo
        rss_mqo.append(np.sum((y_test - y_pred_mqo) ** 2))

        # Modelo de média
        y_pred_media = np.full_like(y_test, np.mean(y_train))
        rss_media.append(np.sum((y_test - y_pred_media) ** 2))

    # Armazena resultados
    resultados['Média da variável dependente'] = calcula_estatisticas(rss_media)
    resultados['MQO tradicional'] = calcula_estatisticas(rss_mqo)

def MQO_Tikhonov(data, rounds, lambdas):
    rss_regularizado = {lambd: [] for lambd in lambdas}

    # Prepara os dados para regressão regularizada
    X_raw = data[:, 0]
    y = data[:, 1]

    plt.scatter(X_raw, y, color='blue')
    plt.title('Visualização inicial dos dados')
    plt.xlabel('Velocidade do Vento')
    plt.ylabel('Potência Gerada')
    plt.show()

    # Adiciona o termo de intercepto na matriz X
    X = np.vstack([np.ones(len(X_raw)), X_raw]).T
    y = y.reshape(-1, 1)

    for _ in range(rounds):
        indices = np.random.permutation(len(X))
        split = int(len(X) * 0.8)
        train_idx, test_idx = indices[:split], indices[split:]
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Avalia o modelo regularizado para cada lambda
        for lambd in lambdas:
            beta_reg = mqo_regularizado(X_train, y_train, lambd)
            y_pred_reg = X_test @ beta_reg
            rss_regularizado[lambd].append(np.sum((y_test - y_pred_reg) ** 2))

    # Armazena resultados
    for lambd in lambdas:
        modelo_nome = f"MQO regularizado ({lambd})"
        resultados[modelo_nome] = calcula_estatisticas(rss_regularizado[lambd])

def exibir_tabela(resultados):
    # Exibe a tabela com os resultados
    modelos = list(resultados.keys())
    metricas = ['Média', 'Desvio-Padrão', 'Maior Valor', 'Menor Valor']

    dados_tabela = []
    for modelo in modelos:
        linha = [modelo]
        estatisticas = resultados[modelo]
        linha.extend([f'{valor:.3f}' for valor in estatisticas])
        dados_tabela.append(linha)

    fig, ax = plt.subplots(figsize=(12, len(modelos)))
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=dados_tabela, colLabels=['Modelos'] + metricas, loc='center', cellLoc='center', colWidths=[0.25, 0.12, 0.13, 0.13, 0.13])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)
    plt.show()

# Configuração e execução dos modelos
rounds_T = 500
resultados = {}
datasMQO = np.loadtxt('aerogerador.dat')
MQO_Default(datasMQO, rounds_T)

lambdas = [0.25, 0.5, 0.75, 1]
MQO_Tikhonov(datasMQO, rounds_T, lambdas)
exibir_tabela(resultados)
