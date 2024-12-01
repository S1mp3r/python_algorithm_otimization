import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def sign(u):
    return 1 if u>=0 else -1

def simplePerceptron(x_raw, y_raw,  epocas_max = 200, lr=0.01):

    x_raw = x_raw.T
    y_raw = y_raw.T

    x_normalized = x_raw
    # x_normalized = (x_raw - np.min(x_raw)) / (np.max(x_raw) - np.min(x_raw))

    c = 2
    p, N = x_raw.shape

    #Adicionando BIAS
    x_normalized = np.concatenate((
        -np.ones((1,N)),
        x_normalized)
    )

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

def ADALINE(x_raw, y_raw, epocas_max = 200, lr=0.01, pr=0.01):

    x_raw = x_raw.T
    y_raw = y_raw.T

    # x_normalized = x_raw
    x_normalized = (x_raw - np.min(x_raw)) / (np.max(x_raw) - np.min(x_raw))

    c = 2
    p, N = x_raw.shape

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





def forward(x, w_list, i_list, y_list, y, signs):
    j = 0

    for w in w_list:
        if j == 0:
            i_list[j] = w.T@x
            y_list[j] = signs(i_list[j]) #FAZER O SIGMOIDE
        else:
            y_bias = y[j - 1]
            i_list[j] = w.T@y_bias
            y_list[j] = signs(i_list[j]) #FAZER O SIGMOIDE
        j += 1

    return x

def backward(x, d, w_list, erro_list, y, lr, signs):
    j = len(w_list) - 1
    
    while(j >= 0):
        if j + 1 == len(w_list):
            erro_list[j] = signs(x) * (d - y[j])
            y_bias = y[j - 1]
            w_list[j] = w_list[j] + lr*erro_list[j]*y_bias
        elif j == 0:
            w_list[j + 1] = w_list[j + 1].T
            erro_list[j] = signs(x) * (w_list[j + 1] * erro_list[j + 1])
            w_list[j] = w_list[j] + lr*erro_list[j]*x
        else:
            w_list[j + 1] = w_list[j + 1].T
            erro_list[j] = signs(x) * (w_list[j + 1] * erro_list[j + 1])
            y_bias = y[j - 1]
            w_list[j] = w_list[j] + lr*erro_list[j]*y_bias
        j -= 1

    return x,d

def EQM_MLP(X, Y, w_list, i_list, y_list, m, L, signs):
    p_1,N = X.shape
    eq = 0
    D = []

    for t in range(N):
        x_t = X[:,t].reshape(p_1,1)
        forward(x_t, w_list, i_list, y_list, Y, signs)
        D.append(Y[t])
        eqi = 0
        j = 0
        for _ in range(m):
            eqi += (D[j] - Y[len(L) - 1][j])**2
            j += 1
        
        eq += eqi
    
    return eq/(2*N)

def sigmoid_log(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid_log(x) * (1 - sigmoid_log(x))

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x)**2

def MLP(x_raw, y_raw, epocas_max = 200, lr=0.01, pr=0.01, L=1, Qn=1, m=1, c=2, activation="sigmoid"):

    x_raw = x_raw.T
    y_raw = y_raw.T

    x_normalized_positivo = (x_raw - np.min(x_raw)) / (np.max(x_raw) - np.min(x_raw))

    x_normalized_negativo = 2*((x_raw - np.min(x_raw)) / (np.max(x_raw) - np.min(x_raw))) - 1

    p, N = x_raw.shape

    #Adicionando BIAS
    x_normalized_positivo = np.concatenate((
        -np.ones((1,N)),
        x_normalized_positivo)
    )

    x_normalized_negativo = np.concatenate((
        -np.ones((1,N)),
        x_normalized_negativo)
    )

    y = np.concatenate((
        -np.ones((1, N)),
        y_raw),
    axis=0)

    L = [p] + Qn + [1]
    
    i_list = []
    y_list = []
    w_list = []
    erro_list = []
    results_ta = []


    for _ in range(L):
        w = np.zeros((L+1,1))
        w = np.random.random_sample((L+1,1))-.5
        w_list.append(w)

    epoca = 0

    if activation == 'sigmoid':
        activation_derivative = sigmoid_derivative
    elif activation == 'tanh':
        activation_derivative = tanh_derivative

    EQM1 = 1
    EQM2 = 0
    while(epoca < epocas_max and abs(EQM1-EQM2) > pr):
        EQM1 = EQM_MLP(x_normalized_positivo, y_raw, w_list, i_list, y_list, m, L, activation_derivative)


        for t in range(N):
            x_t = x_normalized_positivo[:,t].reshape(p+1,1)
            forward(x_t, w_list, i_list, y_list, y, activation_derivative)
            d_t = y_raw[t]
            backward(x_t, d_t, w_list, erro_list, y_raw, lr, activation_derivative)
        
        EQM2 = EQM_MLP(x_normalized_positivo, y_raw, w_list, i_list, y_list, m, L, activation_derivative)

    x2 = -w[1,0]/w[2,0]*x_normalized_positivo + w[0,0]/w[2,0]
    x2 = np.nan_to_num(x2)
    plt.title("MLP")
    plt.plot(x_normalized_positivo, x2,color='k',alpha=.2)
    plt.show()
    

