import numpy as np
import matplotlib.pyplot as plt

def f(x):
    pass

#Maximo de interacoes
max_int = 10000

#Restricoess
x_l = [-1, -1]
x_u = [1, 1]

#X otimo aleatorio uniforme
x_opt = np.random.uniform(x_l, x_u)
f_opt = f(x_opt)



#X candidato aleatorio uniforme
x_cand = np.random.uniform(x_l, x_u)

#Valor de abertura dos vizinhos
sigma = 0.1

i=0
while i < max_int:
    x_cand = np.random.uniform(x_l, x_u)
    f_cand = f(x_cand)
    if f_cand > f_opt:
        x_opt = x_cand
        f_opt = f_cand
        break
    i+=1
