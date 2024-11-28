import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import algoritmos as codigo

data = np.loadtxt("av3\spiral.csv", delimiter=",")
x_raw = data[:, :2]
y_raw = data[:, 2].astype(int)


plt.figure(figsize=(8, 6))
sns.scatterplot(x=x_raw[:, 1], y=x_raw[:, 0], hue=y_raw, palette="Set1", s=100)
plt.title("Initial Data Visualization")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

rodadas_max = 10
rodadas = 0
while(rodadas < rodadas_max):

    indices = np.random.permutation(len(x_raw))
    split = int(len(x_raw) * 0.8)
    train_idx, test_idx = indices[:split], indices[split:]
    X_train, X_test = x_raw[train_idx], x_raw[test_idx]
    y_train, y_test = y_raw[train_idx], y_raw[test_idx]

    codigo.simplePerceptron(X_train, y_train, epocas_max = 100)

    codigo.ADALINE(X_train, y_train, epocas_max = 100)

bp = 1