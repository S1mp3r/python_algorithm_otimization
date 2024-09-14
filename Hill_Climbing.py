import numpy as np
import matplotlib.pyplot as plt

def hill_Climbing_init_(rounds, limit_1, limit_2, limit_3, limit_4, f_apt, minimization):

    def f(x1, x2):
        return f_apt(x1, x2)

    max_int = 10000
    e = 0.1
    limites = np.array(
            [[limit_1, limit_2],
            [limit_3, limit_4]]
        )
    
    def HillClimbing(x, limites):
        l1, l2 = limites
        x_cand = np.clip(np.random.uniform(low=x-e, high=x+e), l1, l2)
        return x_cand
    
    def perturb(x):
        return np.array([HillClimbing(x, l) for x, l in zip(x, limites)])

    x_opt = np.array([limit_1, limit_1])
    f_opt = f(x_opt[0], x_opt[1])

    max_viz = 20
    melhoria = True
    i = 0
    rodadas = 0

    # 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x1_vals = np.linspace(limites[0,0], limites[0,1], 100)
    x2_vals = np.linspace(limites[0,0], limites[0,1], 100)
    X1, X2 = np.meshgrid(x1_vals, x2_vals)

    ax.plot_surface(X1, X2, f(X1, X2), rstride=10, cstride=10, cmap='viridis', alpha=0.6)
    # ax.plot_surface(X1, X2, f(X1, X2), cmap='viridis', alpha=0.6)
    # 3D

    count = 0
    solucoes = []
    last_val = f_opt

    #Hill Climbing
    while rodadas < rounds:
        x_opt = np.array([limit_1, limit_1])
        f_opt = f(x_opt[0], x_opt[1])

        melhoria = True
        i = 0
        while i < max_int and melhoria:
            melhoria = False

            for j in range(max_viz):
                x_cand = perturb(x_opt)
                f_cand = f(x_cand[0], x_cand[1])
                
                if minimization:
                    if f_cand < f_opt:
                        x_opt = x_cand
                        f_opt = f_cand
                        melhoria = True
                        break
                else:
                    if f_cand > f_opt:
                        x_opt = x_cand
                        f_opt = f_cand
                        melhoria = True
                        break
            
            if count == 20:
                if np.abs(last_val - f_opt) < 0.000000001:
                    break
            else:
                last_val = f_opt
                count = 0
            
            count += 1
            i += 1
        solucoes.append(x_opt)
        rodadas += 1

    # Plot com marcacao
    # ax.scatter(x_opt[0], x_opt[1], f_opt, color='r', marker='x', s=100, linewidth=3) 

    ax.scatter(x_opt[0], x_opt[1], f_opt, color='r',s=100, edgecolor='k', zorder=5)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title('Hill Climbing f(x1, x2)')

    values, counts = np.unique(solucoes, return_counts=True)

    index = np.argmax(counts)

    moda = values[index]

    plt.show()

    bp = 1