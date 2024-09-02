# import numpy as np
# import matplotlib.pyplot as plt

# #GLOBAL RANDOM SEARCH

# def f(x1, x2):
#     x_cand = (x1**2) + (x2**2)

#     return x_cand

# #Maximo de interacoes
# max_int = 10000

# #Restricoess
# x_l = [-100, -100]
# x_u = [100, 100]

# #X otimo aleatorio uniforme
# x_opt = np.random.uniform(x_l, x_u)
# f_opt = f(x_opt[0], x_opt[1])

# plt.scatter(x_opt,f_opt,color='r',marker='x')

# #X candidato aleatorio uniforme
# x_cand = np.random.uniform(x_l, x_u)

# #Valor de abertura dos vizinhos
# sigma = 0.1

# x_axis = np.linspace(x_l,x_u,1000)
# plt.plot((x_axis), f(x_axis[0], x_axis[1]))

# i=0
# while i < max_int:
#     x_cand = np.random.uniform(x_l, x_u)
#     f_cand = f(x_cand[0], x_cand[1])
#     if f_cand > f_opt:
#         x_opt = x_cand
#         f_opt = f_cand
#         plt.scatter(x_opt,f_opt,color='r',marker='x')
#         break
#     i+=1






import numpy as np
import matplotlib.pyplot as plt

def perturb(x,e):
    return np.random.uniform(low=x-e,high=x+e)

def f(x1, x2):
    x_cand = (x1**2) + (x2**2)
    return x_cand


x_opt = 1
# f_opt = f(x_opt)
f_opt = [f(1, 1)]

plt.scatter(x_opt,f_opt,color='r',marker='x')

e = .1

#Maximo de interacoes
max_int = 10000

#Restricoess
x_l = [-100, -100]
x_u = [100, 100]

max_viz = 20

melhoria = True
i = 0

x_axis = np.linspace(x_l, x_u, 1000)
plt.plot(x_axis, np.reshape(f(x_axis[0], x_axis[1]), (2, 1000)))

# valores = [f_opt]
while i < max_int and melhoria:
    melhoria = False

    for j in range(max_viz):
        x_cand = perturb(x_opt,e)
        f_cand = f(x_cand[0], x_cand[1])
        if f_cand > f_opt:
            x_opt = x_cand
            f_opt = f_cand
            # valores.append(f_opt)
            melhoria = True
            plt.scatter(x_opt,f_opt,color='r',marker='x')
            break
    i+=1

plt.pause(.1)
plt.scatter(x_opt,f_opt,color='g',marker='x',s=100,linewidth=3)
plt.show()


# Vetor de valores
# plt.plot(valores)
# plt.show()