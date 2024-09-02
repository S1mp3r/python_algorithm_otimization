import numpy as np
import matplotlib.pyplot as plt
import time

# def perturb(x, xl, xu, sig):
#     x_cand = x + np.random.normal(loc=0,scale=sig, size=x.shape)
#     x_cand = np.round(x_cand).astype(int)

#     for i in range(x.shape[0]):
#         if(x_cand[i] < xl):
#             x_cand[i] = xl
#         if(x_cand[i] > xu):
#             x_cand[i] = xu
#     return x_cand

# Ideia do Gui
def remRep(arr):
    rest = [i for i in range(len(arr)) if i not in arr]
    return rest

# Ideia do Gui
def isRep(arr):
    for i in range(len(arr)-1):
        for j in range(i+1, len(arr)):
            if arr[i] == arr[j]:
                return True, i
    return False, None

# Ideia do Gui
def perturb(x):
    rep, index = isRep(x)
    if rep:
        substitute = np.random.permutation(remRep(x))[0]
        x[index] = substitute
        return x
    else:
        i1, i2 = np.random.permutation(len(x))[0:2]
        x[i1], x[i2] = x[i2], x[i1]
        return x


# 2.1
def h(x):
    n = len(x)
    pares_atacantes = 0

    #Verifica os pares
    for i in range(n):
        for j in range(i + 1, n):
            if x[i] == x[j] or abs(x[i] - x[j]) == abs(i - j):
                pares_atacantes += 1

    return pares_atacantes

# 2.1
def f(x):
    return 28 - h(x)

def isNotInside(value, list):
    for optimal in list:
        if np.array_equal(optimal, value):
            return False
    return True


x_l = 0
x_u = 7

# 2.1
x_opt = np.random.uniform(low=x_l, high=x_u, size=8)
x_opt = np.round(x_opt).astype(int)

# 2.1
f_opt = f(x_opt)

# Ideia do Gui
# x_opt = np.random.permutation(8)
# f_opt = f(x_opt)

it_max  = 1000000

# 2.2
T = 100
sigma = 1

i = 0
xs_otimos = []

# 2.6
inicio = time.time()

# 2.5
while i < it_max and len(xs_otimos) < 92:
    # x_cand = perturb(x_opt,x_l,x_u,sigma)
    # f_cand = f(x_cand)

    # Ideia do Gui
    x_cand = perturb(x_opt)
    f_cand = f(x_cand)

    p_ij = np.exp(-(f_cand-f_opt)/T)
    testezin = np.random.uniform(0,1)

    if f_cand > f_opt or p_ij >= testezin:
        x_opt = x_cand
        f_opt = f_cand

    #2.3
    #Escalonamento
    i+=1
    T*=.79
    
    if f_opt == 28:
        if isNotInside(x_opt, xs_otimos):
            xs_otimos.append(x_opt)

        x_opt = np.random.uniform(low=x_l, high=x_u, size=8)
        x_opt = np.round(x_opt).astype(int)

        # Ideia do Gui
        # x_opt = np.random.permutation(8)
        
        # i = 0
        T = 100

# 2.6
fim = time.time()

# 2.6
tempo_execucao = fim - inicio
print(f"Tempo de execução: {tempo_execucao} segundos")

# COMO PLOTA???
plt.plot(xs_otimos)
plt.show()

bp = 1