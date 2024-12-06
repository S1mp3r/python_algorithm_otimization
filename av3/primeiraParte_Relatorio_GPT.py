import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import algoritmos_GPT as codigo
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


data = np.loadtxt("av3/spiral.csv", delimiter=",")
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

learningR = 1e-3
precisionR = 1e-4
precisionR_MLP = 1e-5
epocas_max = 600
epocas_max_MLP = 1500
hidden_layers = [10]
activation_tanh = 'tanh'
activation_sigmoid = 'sigmoid'
max_rounds = 5

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
    
    y_pred_mlp_tanh = codigo.MLP_predict(model_mlp_tanh, X_test)
    y_pred_mlp_bin_tanh = np.array([codigo.sign_ajustavel(u, first=1, second=0, third=-1) for u in y_pred_mlp_tanh.flatten()])

    acc_mlp_tanh, esp_mlp_tanh, sens_mlp_tanh, matriz_mlp_tanh = calcular_metricas_negativas(y_test_bin_negativo, y_pred_mlp_bin_tanh)
    acuracias["MLP - TANH"].append(acc_mlp_tanh)
    matrizes["MLP - TANH"].append(matriz_mlp_tanh)

    clear()

for modelo in acuracias:
    # Cálculo das métricas médias
    matriz_media = np.mean(matrizes[modelo], axis=0)
    matriz_media_int = matriz_media.astype(int)
    acuracia_media = np.mean(acuracias[modelo])
    especificidade_media = matriz_media[0, 0] / (matriz_media[0, 0] + matriz_media[0, 1])
    sensibilidade_media = matriz_media[1, 1] / (matriz_media[1, 1] + matriz_media[1, 0])

    # Plotar matriz média de confusão com métricas
    plotar_matriz_confusao_com_metricas(
        matriz_media_int, 
        f"Matriz Média de Confusão - {modelo}", 
        acuracia_media, 
        sensibilidade_media, 
        especificidade_media
    )

bp = 1