import numpy as np
import matplotlib.pyplot as plt

def perturb(x,e):
    return np.random.uniform(low=x-e,high=x+e)

def f(x):
    return np.exp(-(x**2)) + 3*np.exp(-((x-3)**2))


x_axis = np.linspace(-2,5,1000)
plt.plot(x_axis,f(x_axis))
# plt.show()

x_opt = 1.27
f_opt = f(x_opt)

plt.scatter(x_opt,f_opt,color='r',marker='x')

e = .1

max_it = 10000000000
max_viz = 20
melhoria = True
i = 0

valores = [f_opt]
while i < max_it and melhoria:
    melhoria = False

    for j in range(max_viz):
        x_cand = perturb(x_opt,e)
        f_cand = f(x_cand)
        if f_cand > f_opt:
            x_opt = x_cand
            f_opt = f_cand
            valores.append(f_opt)
            melhoria = True
            # plt.pause(.1)
            plt.scatter(x_opt,f_opt,color='r',marker='x')
            break
    i+=1

plt.pause(.1)
plt.scatter(x_opt,f_opt,color='g',marker='x',s=100,linewidth=3)
plt.show()


# Vetor de valores
# plt.plot(valores)
# plt.show()