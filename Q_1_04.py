import numpy as np
import matplotlib.pyplot as plt

#DANDO BOM

def perturb(x, e):
    # return np.array([np.random.uniform(low=xi-e, high=xi+e) for xi in x])
    x_cand = np.random.uniform(low=x-e,high=x+e)

    x_cand = np.clip(x_cand, x_l, x_u)

    return x_cand


def f(x1, x2):
    termo1 = x1**2 - 10 * np.cos(2 * np.pi * x1) + 10
    termo2 = x2**2 - 10 * np.cos(2 * np.pi * x2) + 10
    
    return termo1 + termo2

max_int = 10000

x_l = [-5.12, -5.12]
x_u = [5.12, 5.12]

e = 0.1
max_viz = 20
melhoria = True
i = 0
rodadas = 0

x_opt = np.random.uniform(low=-5.12, high=5.13, size=2)
f_opt = f(x_opt[0], x_opt[1])

# 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x1_vals = np.linspace(x_l[0], x_u[0], 100)
x2_vals = np.linspace(x_l[1], x_u[1], 100)
X1, X2 = np.meshgrid(x1_vals, x2_vals)

ax.plot_surface(X1, X2, f(X1, X2), cmap='viridis', alpha=0.6)
# 3D

solucoes = []

while rodadas < 100:
    x_opt = np.random.uniform(low=-5.12, high=5.13, size=2)
    f_opt = f(x_opt[0], x_opt[1])

    while i < max_int and melhoria:
        melhoria = False

        for j in range(max_viz):
            x_cand = perturb(x_opt, e)
            f_cand = f(x_cand[0], x_cand[1])
            
            if f_cand < f_opt:
                x_opt = x_cand
                f_opt = f_cand
                melhoria = True
                ax.scatter(x_opt[0], x_opt[1], f_opt)
                break
        
        i += 1
    solucoes.append(x_opt)
    i = 0
    melhoria = True
    rodadas += 1

# Plot com marcacao
# ax.scatter(x_opt[0], x_opt[1], f_opt, color='r', marker='x', s=100, linewidth=3) 

ax.scatter(x_opt[0], x_opt[1], f_opt)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title('f(x1, x2)')

# plt.show()

values, counts = np.unique(solucoes, return_counts=True)

index = np.argmax(counts)

moda = values[index]

bp = 1