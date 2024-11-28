import numpy as np
import matplotlib.pyplot as plt

# Carregar os dados
data = np.loadtxt("EMGDataset.csv", delimiter=',')

# Exibir formato e as primeiras linhas dos dados
print("Formato dos dados:", data.shape)
print("Primeiras linhas dos dados:")
print(data[:5])  # Mostra as primeiras 5 linhas

# Transpor os dados para que cada coluna represente uma amostra
data = data.T

# Definindo as classes
c1, c2, c3 = 2, 4, 5  # Classes: Sorriso, Surpreso, Rabugento

# Plotar os dados das classes selecionadas
plt.scatter(data[data[:, 2] == c1, 0], data[data[:, 2] == c1, 1], color='green', edgecolor='k', label='Sorriso')
plt.scatter(data[data[:, 2] == c2, 0], data[data[:, 2] == c2, 1], color='red', edgecolor='k', label='Surpreso')
plt.scatter(data[data[:, 2] == c3, 0], data[data[:, 2] == c3, 1], color='teal', edgecolor='k', label='Rabugento')

# Concatenar os dados para o modelo
X = np.concatenate((
    data[data[:, 2] == c1, :2],
    data[data[:, 2] == c2, :2],
    data[data[:, 2] == c3, :2],
))

# Verifique o formato de X
print("Formato de X:", X.shape)

N, p = X.shape

# Adicionando um termo de bias
X = np.concatenate((
    np.ones((N, 1)),
    X
), axis=1)

# Criar a matriz Y (one-hot encoding para 3 classes)
Y = np.concatenate((
    np.tile(np.array([[1, -1, -1]]), (10000, 1)),
    np.tile(np.array([[-1, 1, -1]]), (10000, 1)),
    np.tile(np.array([[-1, -1, 1]]), (10000, 1)),
))

# Cálculo dos pesos W
W = np.linalg.inv(X.T @ X) @ X.T @ Y
# Alternativa com pseudo-inversa
W2 = np.linalg.pinv(X.T @ X) @ X.T @ Y
# Usando a função de mínimos quadrados
W3 = np.linalg.lstsq(X, Y, rcond=None)[0]

# Plotando as linhas de decisão
x1 = np.linspace(-500, 5500, 1000)
for i in range(3):
    x2 = -W[0, i] / W[2, i] - W[1, i] / W[2, i] * x1  
    plt.plot(x1, x2, "--k")

# Novos dados para previsão
X_new = np.array([
    [1, 1047, 686],
    [1, 671, 2446],
])

# Previsão
Y_hat = np.argmax(X_new @ W, axis=1)
print("Previsões:", Y_hat)

# Plotando novos pontos
plt.scatter(X_new[:, 1], X_new[:, 2], marker='x')

# Criando a grade para o gráfico de contorno
xx, yy = np.meshgrid(x1, x1)
xx1 = np.ravel(xx)
xx2 = np.ravel(yy)
X_mapa = np.concatenate((
    np.ones((len(xx1), 1)),
    xx1.reshape(len(xx1), 1),
    xx2.reshape(len(xx2), 1),
), axis=1)

# Previsão para a grade
Y_mapa = np.argmax(X_mapa @ W, axis=1)
Y_mapa = Y_mapa.reshape(xx.shape)

# Contornos
plt.contourf(xx, yy, Y_mapa, alpha=.3)
plt.legend()
plt.xlabel("Corrugador do Supercílio")
plt.ylabel("Zigomático Maior")
plt.xlim(-150, 4200)
plt.ylim(-150, 4200)
plt.title("Classificação EMG")
plt.show()

bp = 1