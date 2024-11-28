import numpy as np
import matplotlib.pyplot as plt

def perturb(x,xl,xu,sig):

    x_cand = x + np.random.normal(loc=0,scale=sig)
    for i in range(x.shape[0]):
        if(x_cand[i]<xl[i]):
            x_cand[i] = xl[i]
        if(x_cand[i]> xu[i]):
            x_cand[i] = xu[i]
    return x_cand
        

def f(x,y):
    return x**2*np.sin(4*np.pi*x) - y*np.sin(4*np.pi*y+np.pi) + 1


x_l = [-1,-1]
x_u = [2,2]


x_opt = np.random.uniform(low=x_l, high=x_u)
f_opt = f(x_opt[0],x_opt[1])

x_axis = np.linspace(-1,2,1000)
X,Y = np.meshgrid(x_axis,x_axis)
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot_surface(X,Y,f(X,Y),cmap='gray',alpha=.3,edgecolor='k',rstride=30,cstride=30)



it_max  = 1000
T = 100
sigma = .2

i = 0
f_otimos = []
while i < it_max:
    x_cand = perturb(x_opt,x_l,x_u,sigma)
    f_cand = f(x_cand[0],x_cand[1])

    p_ij = np.exp(-(f_cand-f_opt)/T)
    
    if f_cand < f_opt or p_ij >= np.random.uniform(0,1):
        x_opt = x_cand
        f_opt = f_cand
    i+=1
    f_otimos.append(f_opt)
    T*=.79

plt.show()


plt.plot(f_otimos)
plt.show()

bp = 1