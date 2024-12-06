import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import algoritmosMultiCamadas as codigo
import cv2
import os

def calcular_metricas_generica(y_true, y_pred, positivo=1, negativo=0):
    TP = np.sum((y_true == positivo) & (y_pred == positivo))    # Verdadeiros positivos
    TN = np.sum((y_true == negativo) & (y_pred == negativo))    # Verdadeiros negativos
    FP = np.sum((y_true == negativo) & (y_pred == positivo))    # Falsos positivos
    FN = np.sum((y_true == positivo) & (y_pred == negativo))    # Falsos negativos

    acuracia = (TP + TN) / (TP + TN + FP + FN)
    especificidade = TN / (TN + FP) if (TN + FP) > 0 else 0
    sensibilidade = TP / (TP + FN) if (TP + FN) > 0 else 0

    return acuracia, especificidade, sensibilidade, [[TN, FP], [FN, TP]]


def plotar_matriz_confusao(matriz, titulo, acuracia, sensibilidade, especificidade, labels):
    plt.figure()
    plt.title(f"{titulo}\nAcurácia: {acuracia:.4f} | Sensibilidade: {sensibilidade:.4f} | Especificidade: {especificidade:.4f}")
    plt.imshow(matriz, cmap='Blues', interpolation='nearest')
    plt.colorbar(label="Contagem")
    plt.xticks([0, 1], [f"{labels[0]} (Negativo)", f"{labels[1]} (Positivo)"])
    plt.yticks([0, 1], [f"{labels[0]} (Negativo)", f"{labels[1]} (Positivo)"])
    plt.xlabel("Previsão")
    plt.ylabel("Real")
    for i in range(2):
        for j in range(2):
            plt.text(j, i, f"{matriz[i][j]}", ha='center', va='center', color="black")
    plt.show()

clear = lambda: os.system('cls')


#ATENÇÃO:
#Salve este algoritmo no mesmo diretório no qual a pasta chamada RecFac está.


#A tarefa nessa etapa é realizar o reconhecimento facial de 20 pessoas

#Dimensões da imagem. Você deve explorar esse tamanho de acordo com o solicitado no pdf.
dimensao = 50 #50 signica que a imagem terá 50 x 50 pixels. ?No trabalho é solicitado para que se investigue dimensões diferentes:
# 50x50, 40x40, 30x30, 20x20, 10x10 .... (tua equipe pode tentar outros redimensionamentos.)

#Criando strings auxiliares para organizar o conjunto de dados:
pasta_raiz = "av3/RecFac"
caminho_pessoas = [x[0] for x in os.walk(pasta_raiz)]
caminho_pessoas.pop(0)

C = 20 #Esse é o total de classes 
X = np.empty((dimensao*dimensao,0)) # Essa variável X será a matriz de dados de dimensões p x N. 
Y = np.empty((C,0)) #Essa variável Y será a matriz de rótulos (Digo matriz, pois, é solicitado o one-hot-encoding).
for i,pessoa in enumerate(caminho_pessoas):
    imagens_pessoa = os.listdir(pessoa)
    for imagens in imagens_pessoa:

        caminho_imagem = os.path.join(pessoa,imagens)
        imagem_original = cv2.imread(caminho_imagem,cv2.IMREAD_GRAYSCALE)
        imagem_redimensionada = cv2.resize(imagem_original,(dimensao,dimensao))

        #A imagem pode ser visualizada com esse comando.
        # No entanto, o comando deve ser comentado quando o algoritmo for executado
        # cv2.imshow("eita",imagem_redimensionada)
        # cv2.waitKey(0)

        #vetorizando a imagem:
        x = imagem_redimensionada.flatten()

        #Empilhando amostra para criar a matriz X que terá dimensão p x N
        X = np.concatenate((
            X,
            x.reshape(dimensao*dimensao,1)
        ),axis=1)
        

        #one-hot-encoding (A EQUIPE DEVE DESENVOLVER)
        y = -np.ones((C,1))
        y[i,0] = 1

        Y = np.concatenate((
            Y,
            y
        ),axis=1)

X = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))

# Normalização dos dados (A EQUIPE DEVE ESCOLHER O TIPO E DESENVOLVER):

# Início das rodadas de monte carlo
#Aqui podem existir as definições dos hiperparâmetros de cada modelo.
acuracias = {"Perceptron": [], "ADALINE": [], "MLP - TANH": []}
matrizes = {"Perceptron": [], "ADALINE": [], "MLP - TANH": []}
acuracias_train = {"MLP - TANH": []}
especificidades_train = {"MLP - TANH": []}
sensibilidades_train = {"MLP - TANH": []}

learningR_Simple = 1e-2
epocas_max_Simple = 100

learningR_ADALINE = 1e-3
precisionR_ADALINE = 1e-4
epocas_max_ADALINE = 400

learningR_MLP = 1e-3
precisionR_MLP = 1e-5
epocas_max_MLP = 1500
hidden_layers = [10]
activation_tanh = 'tanh'

clear()

rodadas = 1

for i in range(rodadas):
    
    print(f"RODADA ATUAL: {i+1} DE {rodadas}")

    pass
    #Embaralhar X e Y
    indices = np.random.permutation(X.shape[1])  # Número de colunas em X (e também em Y)
    split = int(len(indices) * 0.8)
    train_idx, test_idx = indices[:split], indices[split:]

    #Particionar em Treino e Teste (80/20)
    X_train, X_test = X[:, train_idx], X[:, test_idx]
    y_train, y_test = Y[:, train_idx], Y[:, test_idx]


    # Ajustando y_test para {0, 1}
    y_test_bin = ((y_test + 1) // 2).astype(int)

    # Ajustando y_test para {-1, 1}
    y_test_bin_negativo = y_test

    #Treinameno Modelo Perceptron Simples
    w_perceptron = codigo.simplePerceptron(X_train,
                                           y_train,
                                           epocas_max=epocas_max_Simple,
                                           lr=learningR_Simple
                                           )[-1]

    #Teste Modelo Perceptron Simples
    X_test_perceptron = np.concatenate((-np.ones((1, X_test.shape[0])), X_test.T))
    y_pred_perceptron = np.array([codigo.sign((w_perceptron.T @ X_test_perceptron[:, i:i+1])[0, 0]) for i in range(X_test_perceptron.shape[1])])

    y_pred_perceptron_bin = ((y_pred_perceptron + 1) // 2).astype(int)

    acc_p, esp_p, sens_p, matriz_p = calcular_metricas_generica(y_test_bin, y_pred_perceptron_bin)
    acuracias["Perceptron"].append(acc_p)
    matrizes["Perceptron"].append(matriz_p)


    #Treinameno Modelo ADALINE
    w_adaline = codigo.ADALINE(X_train,
                               y_train,
                               epocas_max=epocas_max_ADALINE,
                               lr=learningR_ADALINE,
                               pr=precisionR_ADALINE
                               )[-1]
    
    #Teste Modelo Modelo ADALINE
    y_pred_adaline = np.array([w_adaline.T @ X_test_perceptron[:, i] for i in range(X_test.shape[0])]).flatten()
    y_pred_adaline_bin = np.array([codigo.sign_ajustavel(u, first=1, second=0, third=0) for u in y_pred_adaline])

    acc_a, esp_a, sens_a, matriz_a = calcular_metricas_generica(y_test_bin, y_pred_adaline_bin)
    acuracias["ADALINE"].append(acc_a)
    matrizes["ADALINE"].append(matriz_a)


    #Treinameno Modelo MLP Com topologia já definida
    model_mlp_tanh, precision_mlp_tanh = codigo.MLP(X_train,
                           y_train,
                           hidden_layers=hidden_layers,
                           learning_rate=learningR_MLP,
                           epocas_max=epocas_max_MLP,
                           activation=activation_tanh,
                           precision=precisionR_MLP
                           )
    
    #Teste Modelo MLP Com topologia já definida
    y_pred_mlp_train_tanh = codigo.MLP_predict(model_mlp_tanh, X_train)
    y_pred_mlp_bin_train_tanh = np.array([codigo.sign_ajustavel(u, first=1, second=0, third=-1) for u in y_pred_mlp_train_tanh.flatten()])

    acc_mlp_train_tanh, esp_mlp_train_tanh, sens_mlp_train_tanh, matrix_mlp = calcular_metricas_generica(y_train, y_pred_mlp_bin_train_tanh, negativo=-1)

    acuracias_train["MLP - TANH"].append(acc_mlp_train_tanh)
    especificidades_train["MLP - TANH"].append(esp_mlp_train_tanh)
    sensibilidades_train["MLP - TANH"].append(sens_mlp_train_tanh)
    acuracias["MLP - TANH"].append(acc_mlp_train_tanh)
    matrizes["MLP - TANH"].append(matrix_mlp)

    clear()


#MÉTRICAS DE DESEMPENHO para cada modelo:
#Tabela
#Matriz de confusão
#Curvas de aprendizagem
for modelo in acuracias:
        # Encontrar a melhor rodada
    melhor_idx = np.argmax(acuracias[modelo])
    matriz_melhor = matrizes[modelo][melhor_idx]
    acuracia_melhor = acuracias[modelo][melhor_idx]
    especificidade_melhor = matriz_melhor[0][0] / (matriz_melhor[0][0] + matriz_melhor[0][1]) if (matriz_melhor[0][0] + matriz_melhor[0][1]) > 0 else 0
    sensibilidade_melhor = matriz_melhor[1][1] / (matriz_melhor[1][1] + matriz_melhor[1][0]) if (matriz_melhor[1][1] + matriz_melhor[1][0]) > 0 else 0

    # Plotar a melhor matriz de confusão
    if modelo != "MLP - TAHN":
        plotar_matriz_confusao(
            matriz_melhor, 
            f"Matriz de Confusão Melhor - {modelo}", 
            acuracia_melhor, 
            sensibilidade_melhor, 
            especificidade_melhor,
            labels=["0", "1"]
        )
    else:
        plotar_matriz_confusao(
            matriz_melhor, 
            f"Matriz de Confusão Melhor - {modelo}", 
            acuracia_melhor, 
            sensibilidade_melhor, 
            especificidade_melhor,
            labels=["-1", "1"]
        )

    # Encontrar a pior rodada
    pior_idx = np.argmin(acuracias[modelo])
    matriz_pior = matrizes[modelo][pior_idx]
    acuracia_pior = acuracias[modelo][pior_idx]
    especificidade_pior = matriz_pior[0][0] / (matriz_pior[0][0] + matriz_pior[0][1]) if (matriz_pior[0][0] + matriz_pior[0][1]) > 0 else 0
    sensibilidade_pior = matriz_pior[1][1] / (matriz_pior[1][1] + matriz_pior[1][0]) if (matriz_pior[1][1] + matriz_pior[1][0]) > 0 else 0

    # Plotar a pior matriz de confusão
    if modelo != "MLP - TAHN":
        plotar_matriz_confusao(
            matriz_pior, 
            f"Matriz de Confusão Pior - {modelo}", 
            acuracia_pior, 
            sensibilidade_pior, 
            especificidade_pior,
            labels=["0", "1"]
        )
    else:
        plotar_matriz_confusao(
            matriz_pior, 
            f"Matriz de Confusão Pior - {modelo}", 
            acuracia_pior, 
            sensibilidade_pior, 
            especificidade_pior,
            labels=["-1", "1"]
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
rodadas_ = np.arange(1, rodadas + 1)

# Inicializar dicionários para armazenar as métricas de teste e treinamento
acuracias_test = {}
acuracias_train_plot = {}

for modelo in acuracias:
    acuracias_test[modelo] = acuracias[modelo]
    if modelo == "MLP - TANH":
        acuracias_train_plot[modelo] = acuracias_train[modelo]
    else:
        acuracias_train_plot[modelo] = [None] * rodadas  # Preencher com None para modelos sem treinamento

# Função para plotar a curva de aprendizado para Acurácia
def plotar_curva_acuracia():
    plt.figure(figsize=(10, 6))
    for modelo in acuracias:
        plt.plot(rodadas_, acuracias_test[modelo], marker='o', label=f'{modelo} - Teste')
    plt.title('Curva de Aprendizagem - Acurácia')
    plt.xlabel('Rodada')
    plt.ylabel('Acurácia')
    plt.xticks(rodadas_)
    plt.legend()
    plt.grid(True)
    plt.show()

# Chamar a função de plotagem
plotar_curva_acuracia()