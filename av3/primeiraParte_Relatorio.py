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

codigo.simplePerceptron(x_raw, y_raw, rodadas_max=5, epocas_max = 100)

codigo.ADALINE(x_raw, y_raw, rodadas_max=5, epocas_max = 100)

bp = 1