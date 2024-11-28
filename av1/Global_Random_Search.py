import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

#GRS
def GRS_init_(rounds, limit_1, limit_2, limit_3, limit_4, f_apt, minimization):
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
    tolerance = 1e-6

    # 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x1_vals = np.linspace(limites[0,0], limites[0,1], 100)
    x2_vals = np.linspace(limites[0,0], limites[0,1], 100)
    X1, X2 = np.meshgrid(x1_vals, x2_vals)

    ax.plot_surface(X1, X2, f(X1, X2), rstride=10, cstride=10, cmap='viridis', alpha=0.6)
    # ax.plot_surface(X1, X2, f(X1, X2), cmap='viridis', alpha=0.6)
    # 3D

    def GRS(limites):
        l1, l2 = limites
        x_cand = np.random.uniform(l1, l2)

        x_cand = np.clip(x_cand, l1, l2)
        
        return x_cand     

    def perturb_GRS():
        return np.array([GRS(l) for l in limites])

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

            x_cand = perturb_GRS()
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
                if np.abs(last_val - f_opt) < tolerance:
                    break
            else:
                last_val = f_opt
                count = 0
            
            count += 1
            i += 1
        solucoes.append(round(f_opt, 3))
        rodadas += 1
        ax.scatter(x_opt[0], x_opt[1], f_opt, color='r',s=100, edgecolor='k', zorder=5)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title('GRS f(x1, x2)')

    modal_f_opt, modal_count = Counter(solucoes).most_common(1)[0]

    plt.show()

    bp  =  1