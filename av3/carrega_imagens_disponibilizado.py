import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import algoritmosMultiCamadas as codigo
import cv2
import os

clear = lambda: os.system('cls' if os.name == 'nt' else 'clear')

def calcular_metricas_multiclasse(Y_true, Y_pred, C=20):
    N = Y_true.shape[1]
    confusion_matrix = np.zeros((C, C), dtype=int)

    for i in range(N):
        true_class = np.argmax(Y_true[:, i])
        pred_class = np.argmax(Y_pred[:, i])
        confusion_matrix[true_class, pred_class] += 1

    accuracy = np.trace(confusion_matrix) / np.sum(confusion_matrix)

    sensitivities = []
    specificities = []
    total = np.sum(confusion_matrix)
    for i in range(C):
        TP = confusion_matrix[i, i]
        FN = np.sum(confusion_matrix[i, :]) - TP
        FP = np.sum(confusion_matrix[:, i]) - TP
        TN = total - (TP + FP + FN)

        sens = TP / (TP + FN) if (TP+FN) > 0 else 0
        spec = TN / (TN + FP) if (TN+FP) > 0 else 0
        sensitivities.append(sens)
        specificities.append(spec)

    mean_sensitivity = np.mean(sensitivities)
    mean_specificity = np.mean(specificities)

    return accuracy, mean_specificity, mean_sensitivity, confusion_matrix

def plotar_matriz_confusao(matriz, titulo, acuracia, sensibilidade, especificidade, labels):
    plt.figure(figsize=(10,8))
    plt.title(f"{titulo}\nAcurácia: {acuracia:.4f} | Sensibilidade (Média): {sensibilidade:.4f} | Especificidade (Média): {especificidade:.4f}")
    sns.heatmap(matriz, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel("Previsão")
    plt.ylabel("Real")
    plt.show()

dimensao = 50
pasta_raiz = "RecFac"
caminho_pessoas = [x[0] for x in os.walk(pasta_raiz)]
caminho_pessoas.pop(0)

C = 20
X = np.empty((dimensao*dimensao,0))
Y_one_hot = np.empty((C,0))
Y_sign = np.empty((C,0))

for i,pessoa in enumerate(caminho_pessoas):
    imagens_pessoa = os.listdir(pessoa)
    for imagens in imagens_pessoa:
        caminho_imagem = os.path.join(pessoa,imagens)
        imagem_original = cv2.imread(caminho_imagem,cv2.IMREAD_GRAYSCALE)
        imagem_redimensionada = cv2.resize(imagem_original,(dimensao,dimensao))
        x = imagem_redimensionada.flatten().reshape(-1,1)

        # Para MLP: one-hot {0, 1}
        y_oh = np.zeros((C,1))
        y_oh[i,0] = 1

        # Para Perceptron/ADALINE: {-1, 1}
        y_s = -np.ones((C,1))
        y_s[i,0] = 1

        X = np.concatenate((X, x), axis=1)
        Y_one_hot = np.concatenate((Y_one_hot, y_oh), axis=1)
        Y_sign = np.concatenate((Y_sign, y_s), axis=1)

# Normalização z-score global
X = (X - np.mean(X)) / (np.std(X) + 1e-12)

rodadas = 3

acuracias = {"Perceptron": [], "ADALINE": [], "MLP - TANH": []}
especificidades = {"Perceptron": [], "ADALINE": [], "MLP - TANH": []}
sensibilidades = {"Perceptron": [], "ADALINE": [], "MLP - TANH": []}
matrizes = {"Perceptron": [], "ADALINE": [], "MLP - TANH": []}

learningR_Simple = 1e-2
epocas_max_Simple = 100

learningR_ADALINE = 1e-3
precisionR_ADALINE = 1e-4
epocas_max_ADALINE = 400

learningR_MLP = 1e-4
epocas_max_MLP = 5000
hidden_layers = [50,50,50,50]
activation_func = 'relu'

clear()

for i in range(rodadas):
    print(f"RODADA ATUAL: {i+1} DE {rodadas}")
    indices = np.random.permutation(X.shape[1])
    split = int(len(indices) * 0.8)
    train_idx, test_idx = indices[:split], indices[split:]

    X_train, X_test = X[:, train_idx], X[:, test_idx]
    y_train_oh, y_test_oh = Y_one_hot[:, train_idx], Y_one_hot[:, test_idx]
    y_train_s, y_test_s = Y_sign[:, train_idx], Y_sign[:, test_idx]

    # Perceptron (usa Y_sign)
    W_perceptron = codigo.simplePerceptron(X_train,y_train_s,epocas_max=epocas_max_Simple, lr=learningR_Simple)
    X_test_perceptron = np.concatenate((-np.ones((1, X_test.shape[1])), X_test), axis=0)
    u_perceptron = W_perceptron.T @ X_test_perceptron
    Y_pred_perceptron = np.zeros_like(y_test_s)
    for idx in range(u_perceptron.shape[1]):
        Y_pred_perceptron[np.argmax(u_perceptron[:, idx]), idx] = 1
    acc_p, esp_p, sens_p, matriz_p = calcular_metricas_multiclasse(y_test_oh, Y_pred_perceptron, C=C)
    acuracias["Perceptron"].append(acc_p)
    especificidades["Perceptron"].append(esp_p)
    sensibilidades["Perceptron"].append(sens_p)
    matrizes["Perceptron"].append(matriz_p)

    # ADALINE (usa Y_sign)
    W_adaline = codigo.ADALINE(X_train, y_train_s, epocas_max=epocas_max_ADALINE, lr=learningR_ADALINE, pr=precisionR_ADALINE)
    u_adaline = W_adaline.T @ np.concatenate((-np.ones((1, X_test.shape[1])), X_test), axis=0)
    Y_pred_adaline = np.zeros_like(y_test_s)
    for idx in range(u_adaline.shape[1]):
        Y_pred_adaline[np.argmax(u_adaline[:, idx]), idx] = 1
    acc_a, esp_a, sens_a, matriz_a = calcular_metricas_multiclasse(y_test_oh, Y_pred_adaline, C=C)
    acuracias["ADALINE"].append(acc_a)
    especificidades["ADALINE"].append(esp_a)
    sensibilidades["ADALINE"].append(sens_a)
    matrizes["ADALINE"].append(matriz_a)

    # MLP (usa Y_one_hot)
    model_mlp_tanh, precisions_mlp = codigo.MLP(X_train, y_train_oh,
                                                hidden_layers=hidden_layers,
                                                learning_rate=learningR_MLP,
                                                epocas_max=epocas_max_MLP,
                                                activation=activation_func)
    Y_pred_mlp_tanh = codigo.MLP_predict(model_mlp_tanh, X_test)
    Y_pred_mlp_tanh_final = np.zeros_like(y_test_oh)
    for idx in range(Y_pred_mlp_tanh.shape[1]):
        Y_pred_mlp_tanh_final[np.argmax(Y_pred_mlp_tanh[:, idx]), idx] = 1

    acc_mlp, esp_mlp, sens_mlp, matriz_mlp = calcular_metricas_multiclasse(y_test_oh, Y_pred_mlp_tanh_final, C=C)
    acuracias["MLP - TANH"].append(acc_mlp)
    especificidades["MLP - TANH"].append(esp_mlp)
    sensibilidades["MLP - TANH"].append(sens_mlp)
    matrizes["MLP - TANH"].append(matriz_mlp)

    clear()

labels = [f"Classe {i}" for i in range(C)]
for modelo in acuracias:
    melhor_idx = np.argmax(acuracias[modelo])
    matriz_melhor = matrizes[modelo][melhor_idx]
    acuracia_melhor = acuracias[modelo][melhor_idx]
    especificidade_melhor = especificidades[modelo][melhor_idx]
    sensibilidade_melhor = sensibilidades[modelo][melhor_idx]

    plotar_matriz_confusao(
        matriz_melhor, 
        f"Matriz de Confusão Melhor - {modelo}", 
        acuracia_melhor, 
        sensibilidade_melhor, 
        especificidade_melhor,
        labels
    )

    pior_idx = np.argmin(acuracias[modelo])
    matriz_pior = matrizes[modelo][pior_idx]
    acuracia_pior = acuracias[modelo][pior_idx]
    especificidade_pior = especificidades[modelo][pior_idx]
    sensibilidade_pior = sensibilidades[modelo][pior_idx]

    plotar_matriz_confusao(
        matriz_pior, 
        f"Matriz de Confusão Pior - {modelo}", 
        acuracia_pior, 
        sensibilidade_pior, 
        especificidade_pior,
        labels
    )

metrics = ['Acurácia', 'Sensibilidade', 'Especificidade']
table_data = []

for modelo in acuracias:
    acc = acuracias[modelo]
    acc_mean = np.mean(acc)
    acc_std = np.std(acc)
    acc_max = np.max(acc)
    acc_min = np.min(acc)
    table_data.append([modelo, 'Acurácia', f"{acc_mean:.4f}", f"{acc_std:.4f}", f"{acc_max:.4f}", f"{acc_min:.4f}"])

    sens = sensibilidades[modelo]
    sens_mean = np.mean(sens)
    sens_std = np.std(sens)
    sens_max = np.max(sens)
    sens_min = np.min(sens)
    table_data.append([modelo, 'Sensibilidade', f"{sens_mean:.4f}", f"{sens_std:.4f}", f"{sens_max:.4f}", f"{sens_min:.4f}"])

    esp = especificidades[modelo]
    esp_mean = np.mean(esp)
    esp_std = np.std(esp)
    esp_max = np.max(esp)
    esp_min = np.min(esp)
    table_data.append([modelo, 'Especificidade', f"{esp_mean:.4f}", f"{esp_std:.4f}", f"{esp_max:.4f}", f"{esp_min:.4f}"])

df_table = pd.DataFrame(table_data, columns=['Modelo', 'Métrica', 'Média', 'Desvio-Padrão', 'Maior Valor', 'Menor Valor'])

fig, ax = plt.subplots(figsize=(12, 8))
ax.axis('off')
table = ax.table(cellText=df_table.values,
                    colLabels=df_table.columns,
                    cellLoc='center',
                    loc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.5)
plt.title('Estatísticas das Métricas dos Modelos', fontsize=14, pad=20)
plt.show()

rodadas_ = np.arange(1, rodadas + 1)

def plotar_curva_acuracia():
    plt.figure(figsize=(10, 6))
    for modelo in acuracias:
        plt.plot(rodadas_, acuracias[modelo], marker='o', label=f'{modelo}')
    plt.title('Curva de Aprendizagem - Acurácia')
    plt.xlabel('Rodada')
    plt.ylabel('Acurácia')
    plt.xticks(rodadas_)
    plt.legend()
    plt.grid(True)
    plt.show()

plotar_curva_acuracia()
