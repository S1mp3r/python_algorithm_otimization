import numpy as np
import matplotlib.pyplot as plt

#DANDO BOM

def perturb(x, e):
    # return np.array([np.random.uniform(low=xi-e, high=xi+e) for xi in x])

    x_cand = np.random.uniform(low=x-e,high=x+e)

    x_cand = np.clip(x_cand, x_l, x_u)

    return x_cand

def f(x1, x2):
    termo1 = x1 * np.sin(4 * np.pi * x1)
    termo2 = x2 * np.sin((4 * np.pi * x2) + np.pi)

    return termo1 - termo2 + 1

max_int = 10000

x_l = [-1, -1]
x_u = [3, 3]

e = 0.1
max_viz = 20
melhoria = True
i = 0

x_opt = np.random.uniform(low=-1, high=4, size=2)
f_opt = f(x_opt[0], x_opt[1])

# 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x1_vals = np.linspace(x_l[0], x_u[0], 100)
x2_vals = np.linspace(x_l[1], x_u[1], 100)
X1, X2 = np.meshgrid(x1_vals, x2_vals)

ax.plot_surface(X1, X2, f(X1, X2), cmap='viridis', alpha=0.6)
# 3D

while i < max_int and melhoria:
    melhoria = False

    for j in range(max_viz):
        x_cand = perturb(x_opt, e)
        f_cand = f(x_cand[0], x_cand[1])
        
        if f_cand > f_opt:
            x_opt = x_cand
            f_opt = f_cand
            melhoria = True
            ax.scatter(x_opt[0], x_opt[1], f_opt)
            break
    
    i += 1

# Plot com marcacao
# ax.scatter(x_opt[0], x_opt[1], f_opt, color='r', marker='x', s=100, linewidth=3) 

ax.scatter(x_opt[0], x_opt[1], f_opt)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title('f(x1, x2)')

plt.show()

bp = 1