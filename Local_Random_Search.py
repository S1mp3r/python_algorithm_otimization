import numpy as np
import matplotlib.pyplot as plt

def LRS_init_(rounds, limit_1, limit_2, limit_3, limit_4, f_apt, minimization):
    
    def f(x1, x2):
        return f_apt(x1, x2)

    limites = np.array(
            [[limit_1, limit_2],
            [limit_3, limit_4]]
        )

    x_opt = np.array([
                    np.random.uniform(limites[0,0], limites[0,1]),
                    np.random.uniform(limites[1,0], limites[1,1])
                ])
    f_opt = f(x_opt[0], x_opt[1])
    count = 0
    last_val = f_opt
    solucoes = []
    e = .1
    max_int = 10000

    def LRS(x,limites):
        l1, l2 = limites
        x_cand = np.clip(x + np.random.normal(0, e), l1, l2)
        return x_cand

    def perturb_LRS(x):
            return np.array([LRS(x, l) for x, l in zip(x, limites)])

    # 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x1_vals = np.linspace(limites[0,0], limites[0,1], 100)
    x2_vals = np.linspace(limites[0,0], limites[0,1], 100)
    X1, X2 = np.meshgrid(x1_vals, x2_vals)

    ax.plot_surface(X1, X2, f(X1, X2), rstride=10, cstride=10, cmap='viridis', alpha=0.6)
    # ax.plot_surface(X1, X2, f(X1, X2), cmap='viridis', alpha=0.6)
    # 3D

    rodadas = 0
    while rodadas < rounds:
        x_opt = np.array ([
                            np.random.uniform(limites[0,0], limites[0,1]),
                            np.random.uniform(limites[1,0], limites[1,1])
                            ])
        f_opt = f(x_opt[0], x_opt[1])
        
        i = 0
        count = 0
        last_val = f_opt
        while i < max_int:

            x_cand = perturb_LRS(x_opt)
            f_cand = f(x_cand[0], x_cand[1])
            
            if minimization:
                if f_cand < f_opt:
                    x_opt = x_cand
                    f_opt = f_cand
            else:
                if f_cand > f_opt:
                    x_opt = x_cand
                    f_opt = f_cand

            if count == 20:
                if np.abs(last_val - f_opt) < 0.000000001:
                    break
            else:
                last_val = f_opt
                count = 0
            
            count += 1
            i += 1
        solucoes.append(x_opt)
        ax.scatter(x_opt[0], x_opt[1], f_opt, color='r', s=100, edgecolor='k', zorder=5)
        rodadas += 1

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title('LRS f(x1, x2)')
   
    values, counts = np.unique(solucoes, return_counts=True)

    index = np.argmax(counts)

    moda = values[index]

    plt.show()

    bp = 1 