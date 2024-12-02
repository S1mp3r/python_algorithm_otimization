import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import algoritmos as codigo
import os


def calcular_metricas(y_true, y_pred):
    TP = np.sum((y_true == 1) & (y_pred == 1))  # Verdadeiros positivos
    TN = np.sum((y_true == 0) & (y_pred == 0))  # Verdadeiros negativos
    FP = np.sum((y_true == 0) & (y_pred == 1))  # Falsos positivos
    FN = np.sum((y_true == 1) & (y_pred == 0))  # Falsos negativos

    acuracia = (TP + TN) / (TP + TN + FP + FN)
    especificidade = TN / (TN + FP) if (TN + FP) > 0 else 0
    sensibilidade = TP / (TP + FN) if (TP + FN) > 0 else 0

    return acuracia, especificidade, sensibilidade, [[TN, FP], [FN, TP]]

def calcular_metricas_negativas(y_true, y_pred):
    TP = np.sum((y_true == 1) & (y_pred == 1))  # Verdadeiros positivos
    TN = np.sum((y_true == -1) & (y_pred == -1))  # Verdadeiros negativos
    FP = np.sum((y_true == -1) & (y_pred == 1))  # Falsos positivos
    FN = np.sum((y_true == 1) & (y_pred == -1))  # Falsos negativos

    acuracia = (TP + TN) / (TP + TN + FP + FN)
    especificidade = TN / (TN + FP) if (TN + FP) > 0 else 0
    sensibilidade = TP / (TP + FN) if (TP + FN) > 0 else 0

    return acuracia, especificidade, sensibilidade, [[TN, FP], [FN, TP]]

def plotar_matriz_confusao_com_metricas(matriz, titulo, acuracia, sensibilidade, especificidade):
    plt.figure()
    plt.title(f"{titulo}\nAcurácia: {acuracia:.4f} | Sensibilidade: {sensibilidade:.4f} | Especificidade: {especificidade:.4f}")
    plt.imshow(matriz, cmap='Blues', interpolation='nearest')
    plt.colorbar(label="Contagem")
    plt.xticks([0, 1], ["0 (Negativo)", "1 (Positivo)"])
    plt.yticks([0, 1], ["0 (Negativo)", "1 (Positivo)"])
    plt.xlabel("Previsão")
    plt.ylabel("Real")
    for i in range(2):
        for j in range(2):
            plt.text(j, i, f"{matriz[i][j]}", ha='center', va='center', color="black")
    plt.show()


data = np.loadtxt("spiral.csv", delimiter=",")
x_raw = data[:, :2]
y_raw = data[:, 2].astype(int)

plt.figure(figsize=(8, 6))
sns.scatterplot(x=x_raw[:, 1], y=x_raw[:, 0], hue=y_raw, palette="Set1", s=100)
plt.title("Visualização Inicial dos Dados")
plt.xlabel("Característica 1")
plt.ylabel("Característica 2")
plt.show()

acuracias = {"Perceptron": [], "ADALINE": [], "MLP - TANH": []}
matrizes = {"Perceptron": [], "ADALINE": [], "MLP - TANH": []}
acuracias_train = {"MLP - TANH": []}
especificidades_train = {"MLP - TANH": []}
sensibilidades_train = {"MLP - TANH": []}

learningR = 1e-3
precisionR = 1e-4
precisionR_MLP = 1e-5
epocas_max = 600
epocas_max_MLP = 1500
hidden_layers = [10]
activation_tanh = 'tanh'
activation_sigmoid = 'sigmoid'
max_rounds = 2

clear = lambda: os.system('cls')

clear()

for rodada in range(max_rounds):

    print(f"RODADA ATUAL: {rodada+1} DE {max_rounds}")

    indices = np.random.permutation(len(x_raw))
    split = int(len(x_raw) * 0.8)
    train_idx, test_idx = indices[:split], indices[split:]
    X_train, X_test = x_raw[train_idx], x_raw[test_idx]
    y_train, y_test = y_raw[train_idx], y_raw[test_idx]

    # Ajustando y_test para {0, 1}
    y_test_bin = ((y_test + 1) // 2).astype(int)

    # Ajustando y_test para {0, 1}
    y_test_bin_negativo = 2 * y_test_bin - 1

    w_perceptron = codigo.simplePerceptron(X_train,
                                           y_train,
                                           epocas_max=epocas_max,
                                           lr=learningR
                                           )[-1]
    
    X_test_perceptron = np.concatenate((-np.ones((1, X_test.shape[0])), X_test.T))
    y_pred_perceptron = np.array([codigo.sign((w_perceptron.T @ X_test_perceptron[:, i:i+1])[0, 0]) for i in range(X_test_perceptron.shape[1])])

    y_pred_perceptron_bin = ((y_pred_perceptron + 1) // 2).astype(int)

    acc_p, esp_p, sens_p, matriz_p = calcular_metricas(y_test_bin, y_pred_perceptron_bin)
    acuracias["Perceptron"].append(acc_p)
    matrizes["Perceptron"].append(matriz_p)

    # ADALINE
    w_adaline = codigo.ADALINE(X_train,
                               y_train,
                               epocas_max=epocas_max,
                               lr=learningR,
                               pr=precisionR
                               )[-1]
    
    y_pred_adaline = np.array([w_adaline.T @ X_test_perceptron[:, i] for i in range(X_test.shape[0])]).flatten()
    y_pred_adaline_bin = np.array([codigo.sign_ajustavel(u, first=1, second=0, third=0) for u in y_pred_adaline])

    acc_a, esp_a, sens_a, matriz_a = calcular_metricas(y_test_bin, y_pred_adaline_bin)
    acuracias["ADALINE"].append(acc_a)
    matrizes["ADALINE"].append(matriz_a)

    # MLP - Tanh
    model_mlp_tanh, precision_mlp_tanh = codigo.MLP(X_train,
                           y_train,
                           hidden_layers=hidden_layers,
                           learning_rate=learningR,
                           epocas_max=epocas_max_MLP,
                           activation=activation_tanh,
                           precision=precisionR_MLP
                           )
    
   
    y_pred_mlp_train_tanh = codigo.MLP_predict(model_mlp_tanh, X_train)
    y_pred_mlp_bin_train_tanh = np.array([codigo.sign_ajustavel(u, first=1, second=0, third=-1) for u in y_pred_mlp_train_tanh.flatten()])

    acc_mlp_train_tanh, esp_mlp_train_tanh, sens_mlp_train_tanh, matrix_mlp = calcular_metricas_negativas(y_train, y_pred_mlp_bin_train_tanh)
    acuracias_train["MLP - TANH"].append(acc_mlp_train_tanh)
    especificidades_train["MLP - TANH"].append(esp_mlp_train_tanh)
    sensibilidades_train["MLP - TANH"].append(sens_mlp_train_tanh)
    acuracias["MLP - TANH"].append(acc_mlp_train_tanh)
    matrizes["MLP - TANH"].append(matrix_mlp)

    clear()

for modelo in acuracias:
        # Encontrar a melhor rodada
    melhor_idx = np.argmax(acuracias[modelo])
    matriz_melhor = matrizes[modelo][melhor_idx]
    acuracia_melhor = acuracias[modelo][melhor_idx]
    especificidade_melhor = matriz_melhor[0][0] / (matriz_melhor[0][0] + matriz_melhor[0][1]) if (matriz_melhor[0][0] + matriz_melhor[0][1]) > 0 else 0
    sensibilidade_melhor = matriz_melhor[1][1] / (matriz_melhor[1][1] + matriz_melhor[1][0]) if (matriz_melhor[1][1] + matriz_melhor[1][0]) > 0 else 0

    # Plotar a melhor matriz de confusão
    plotar_matriz_confusao_com_metricas(
        matriz_melhor, 
        f"Matriz de Confusão Melhor - {modelo}", 
        acuracia_melhor, 
        sensibilidade_melhor, 
        especificidade_melhor
    )

    # Encontrar a pior rodada
    pior_idx = np.argmin(acuracias[modelo])
    matriz_pior = matrizes[modelo][pior_idx]
    acuracia_pior = acuracias[modelo][pior_idx]
    especificidade_pior = matriz_pior[0][0] / (matriz_pior[0][0] + matriz_pior[0][1]) if (matriz_pior[0][0] + matriz_pior[0][1]) > 0 else 0
    sensibilidade_pior = matriz_pior[1][1] / (matriz_pior[1][1] + matriz_pior[1][0]) if (matriz_pior[1][1] + matriz_pior[1][0]) > 0 else 0

    # Plotar a pior matriz de confusão
    plotar_matriz_confusao_com_metricas(
        matriz_pior, 
        f"Matriz de Confusão Pior - {modelo}", 
        acuracia_pior, 
        sensibilidade_pior, 
        especificidade_pior
    )
acuracia_mlp_tanh_test = acuracias["MLP - TANH"]
especificidade_mlp_tanh_test = [m[0][0] / (m[0][0] + m[0][1]) if (m[0][0] + m[0][1]) > 0 else 0 for m in matrizes["MLP - TANH"]]
sensibilidade_mlp_tanh_test = [m[1][1] / (m[1][1] + m[1][0]) if (m[1][1] + m[1][0]) > 0 else 0 for m in matrizes["MLP - TANH"]]

# Extrair métricas de treinamento
acuracia_mlp_tanh_train = acuracias_train["MLP - TANH"]
especificidade_mlp_tanh_train = especificidades_train["MLP - TANH"]
sensibilidade_mlp_tanh_train = sensibilidades_train["MLP - TANH"]

# Análise de Underfitting e Overfitting
# Underfitting: Baixa acurácia tanto no treinamento quanto no teste
# Overfitting: Alta acurácia no treinamento e baixa no teste

media_acuracia_train = np.mean(acuracia_mlp_tanh_train)
media_acuracia_test = np.mean(acuracia_mlp_tanh_test)

if media_acuracia_train < 0.8 and media_acuracia_test < 0.8:
    print("O modelo MLP - TANH está apresentando underfitting.")
elif media_acuracia_train - media_acuracia_test > 0.1:
    print("O modelo MLP - TANH está apresentando overfitting.")
else:
    print("O modelo MLP - TANH está balanceado, sem sinais claros de underfitting ou overfitting.")

# Definir as métricas a serem analisadas
metrics = ['Acurácia', 'Sensibilidade', 'Especificidade']
statistics = ['Média', 'Desvio-Padrão', 'Maior Valor', 'Menor Valor']

# Inicializar uma lista para armazenar os dados da tabela
table_data = []

for modelo in acuracias:
    # Acurácia
    acc = acuracias[modelo]
    acc_mean = np.mean(acc)
    acc_std = np.std(acc)
    acc_max = np.max(acc)
    acc_min = np.min(acc)
    table_data.append([modelo, 'Acurácia', f"{acc_mean:.4f}", f"{acc_std:.4f}", f"{acc_max:.4f}", f"{acc_min:.4f}"])

    # Sensibilidade
    sens = [m[1][1] / (m[1][1] + m[1][0]) if (m[1][1] + m[1][0]) > 0 else 0 for m in matrizes[modelo]]
    sens_mean = np.mean(sens)
    sens_std = np.std(sens)
    sens_max = np.max(sens)
    sens_min = np.min(sens)
    table_data.append([modelo, 'Sensibilidade', f"{sens_mean:.4f}", f"{sens_std:.4f}", f"{sens_max:.4f}", f"{sens_min:.4f}"])

    # Especificidade
    esp = [m[0][0] / (m[0][0] + m[0][1]) if (m[0][0] + m[0][1]) > 0 else 0 for m in matrizes[modelo]]
    esp_mean = np.mean(esp)
    esp_std = np.std(esp)
    esp_max = np.max(esp)
    esp_min = np.min(esp)
    table_data.append([modelo, 'Especificidade', f"{esp_mean:.4f}", f"{esp_std:.4f}", f"{esp_max:.4f}", f"{esp_min:.4f}"])

# Criar um DataFrame para organizar os dados
df_table = pd.DataFrame(table_data, columns=['Modelo', 'Métrica', 'Média', 'Desvio-Padrão', 'Maior Valor', 'Menor Valor'])

# Plotar a tabela usando matplotlib
fig, ax = plt.subplots(figsize=(12, 8))
ax.axis('off')  # Remover eixos
table = ax.table(cellText=df_table.values,
                    colLabels=df_table.columns,
                    cellLoc='center',
                    loc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.5)
plt.title('Estatísticas das Métricas dos Modelos', fontsize=14, pad=20)
plt.show()

# Definir o número de rodadas
rodadas = np.arange(1, max_rounds + 1)

# Inicializar dicionários para armazenar as métricas de teste e treinamento
acuracias_test = {}
acuracias_train_plot = {}

for modelo in acuracias:
    acuracias_test[modelo] = acuracias[modelo]
    if modelo == "MLP - TANH":
        acuracias_train_plot[modelo] = acuracias_train[modelo]
    else:
        acuracias_train_plot[modelo] = [None] * max_rounds  # Preencher com None para modelos sem treinamento

# Função para plotar a curva de aprendizado para Acurácia
def plotar_curva_acuracia():
    plt.figure(figsize=(10, 6))
    for modelo in acuracias:
        plt.plot(rodadas, acuracias_test[modelo], marker='o', label=f'{modelo} - Teste')
    plt.title('Curva de Aprendizagem - Acurácia')
    plt.xlabel('Rodada')
    plt.ylabel('Acurácia')
    plt.xticks(rodadas)
    plt.legend()
    plt.grid(True)
    plt.show()

# Chamar a função de plotagem
plotar_curva_acuracia()