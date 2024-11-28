import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def sign(u):
    return 1 if u>=0 else -1

def simplePerceptron(x_raw, y_raw,  epocas_max = 200):

    x_raw = x_raw.T
    y_raw = y_raw.T

    x_normalized = (x_raw - np.min(x_raw)) / (np.max(x_raw) - np.min(x_raw))

    c = 2
    p, N = x_raw.shape
    R = 2
    accuracy = []
    sensibility = []
    specificty = []

    #Adicionando BIAS
    x_normalized = np.concatenate((
        -np.ones((1,N)),
        x_normalized)
    )

    lr = .05

    w = np.zeros((p+1,1))
    w = np.random.random_sample((p+1,1))-.5

    erro = True
    epoca = 0

    while(erro and epoca < epocas_max):
        erro = False

        for t in range(N):
            x_t = x_normalized[:,t].reshape(p+1,1)
            u_t = (w.T@x_t)[0,0]
            y_t = sign(u_t)
            d_t = float(y_raw[t])
            e_t = d_t - y_t
            w = w + (lr*e_t*x_t)/2

            if(y_t!=d_t):
                erro = True

        epoca+=1
    x2 = -w[1,0]/w[2,0]*x_normalized + w[0,0]/w[2,0]
    x2 = np.nan_to_num(x2)
    plt.title("Perceptron Simples")
    line = plt.plot(x_normalized, x2, color='green', linewidth=3)
    plt.show()


def EQM(X,Y,w):
    p_1,N = X.shape
    eq = 0
    for t in range(N):
        x_t = X[:,t].reshape(p_1,1)
        u_t = w.T@x_t
        d_t = Y[t]
        eq += (d_t-u_t[0,0])**2
    return eq/(2*N)

def ADALINE(x_raw, y_raw, epocas_max = 200):

    x_raw = x_raw.T
    y_raw = y_raw.T

    x_normalized = (x_raw - np.min(x_raw)) / (np.max(x_raw) - np.min(x_raw))
    # x_normalized = x_raw

    c = 2
    p, N = x_raw.shape
    R = 2
    accuracy = []
    sensibility = []
    specificty = []

    #Adicionando BIAS
    x_normalized = np.concatenate((
        -np.ones((1,N)),
        x_normalized)
    )

    lr = .05
    pr = .05

    w = np.zeros((p+1,1))
    w = np.random.random_sample((p+1,1))-.5

    epoca = 0

    EQM1 = 1
    EQM2 = 0
    while(epoca < epocas_max and abs(EQM1-EQM2) > pr):
        EQM1 = EQM(x_normalized, y_raw, w)

        for t in range(N):
            x_t = x_normalized[:,t].reshape(p+1,1)
            u_t = w.T @ x_t
            d_t = y_raw[t]
            e_t = d_t - u_t
            w = w + lr*e_t*x_t
        epoca+=1
        EQM2 = EQM(x_normalized, y_raw, w)

    x2 = -w[1,0]/w[2,0]*x_normalized + w[0,0]/w[2,0]
    x2 = np.nan_to_num(x2)
    plt.title("Adelinda")
    plt.plot(x_normalized,x2,color='k',alpha=.2)
    plt.show()