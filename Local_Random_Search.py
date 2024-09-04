import numpy as np
import matplotlib.pyplot as plt


def perturb(x):
    i1,i2 = np.random.permutation(len(x))[0:2]
    x[i1],x[i2] = x[i2],x[i1]
    
    return x
    return np.random.permutation(len(x))

def f(cidades,x):
    s = 0
    for i in range(len(x)):
        p1 = cidades[x[i],:]
        p2 = cidades[x[(i+1)%len(x)],:]
        # s = np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
        s += np.linalg.norm(p1-p2)
    return s

def plot_inicial(cidades,x):
    _,ax = plt.subplots()
    ax.scatter(cidades[:,0],cidades[:,1])
    lines = []
    for i in range(len(x)):
        p1 = cidades[x[i],:]
        p2 = cidades[x[(i+1)%len(x)],:]
        if i == 0:
            l = ax.plot([p1[0],p2[0]],[p1[1],p2[1]],color='g')
        elif i == len(x)-1:
            l = ax.plot([p1[0],p2[0]],[p1[1],p2[1]],color='r')
        else:
            l = ax.plot([p1[0],p2[0]],[p1[1],p2[1]],color='k')
        lines.append(l[0])

    return ax,lines


def atualiza_plot(cidades,x,lines,ax):
    plt.pause(.5)
    for line in lines:
        line.remove()

    for i in range(len(x)):
        p1 = cidades[x[i],:]
        p2 = cidades[x[(i+1)%len(x)],:]
        if i == 0:
            l = ax.plot([p1[0],p2[0]],[p1[1],p2[1]],color='g')
        elif i == len(x)-1:
            l = ax.plot([p1[0],p2[0]],[p1[1],p2[1]],color='r')
        else:
            l = ax.plot([p1[0],p2[0]],[p1[1],p2[1]],color='k')
        lines[i] = l[0]

def fat(n):
    return 1 if n<=1 else n*fat(n-1)

    



p = 8

cidades = np.random.rand(p,2)

solucoes = np.random.permutation(p).reshape(1,p)

i = 1
max_sol = fat(p)
avaliacoes = [f(cidades,solucoes[0,:])]

while i < max_sol:

    x = np.random.permutation(p).reshape(1,p)

    if not np.any(np.all(x == solucoes,axis=1)):
        solucoes = np.concat((
            solucoes,x
        ))
        avaliacoes.append(f(cidades,x[0,:]))
        i+=1

# plt.stem(avaliacoes)
# plt.show()
avaliacoes = np.array(avaliacoes)

x_opt = np.random.permutation(p)

f_opt = f(cidades,x_opt)
ax,lines = plot_inicial(cidades,x_opt)

max_it = 100000

for i in range(max_it):
    x_cand = perturb(np.copy(x_opt))
    f_cand = f(cidades,x_cand)    
    if f_cand < f_opt:
        x_opt = x_cand
        f_opt = f_cand
        atualiza_plot(cidades,x_opt,lines,ax)


plt.show()









# import numpy as np
# import matplotlib.pyplot as plt

# #Pertubacao do otimo
# def pertub(x,xl,xu,sig):
#     x_cand = x + np.random.normal(loc=0,scale=sig)
#     for i in range(x.shape[0]):
#         if(x_cand[i]<xl[i]):
#             x_cand[i] = xl[i]
#         if(x_cand[i]> xu[i]):
#             x_cand[i] = xu[i]
#     return x_cand

# def f(x):
#     pass

# #Maximo de interacoes
# max_int = 10000

# #Restricoess
# x_l = [-1, -1]
# x_u = [1, 1]

# #X otimo aleatorio uniforme
# x_opt = np.random.uniform(x_l, x_u)

# f_opt = f(x_opt)

# #X candidato aleatorio uniforme
# x_cand = np.random.uniform(x_l, x_u)

# #Valor de abertura dos vizinhos
# sigma = 0.1

# # Valores para o teste
# x_axis = np.linspace(-2,5,1000)
# # plt.plot(x_axis,f(x_axis))

# i=0
# while i < max_int:
#     x_cand = pertub(x_opt, x_l, x_u, sigma)
#     f_cand = f(x_cand)
#     if f_cand > f_opt:
#         x_opt = x_cand
#         f_opt = f_cand
#         break
#     i+=1
