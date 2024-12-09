import numpy as np

precisions = []

def sign(u):
    return 1 if u >= 0 else -1

def EQM(X, Y, w):
    p_1, N = X.shape
    eq = 0
    for t in range(N):
        x_t = X[:, t].reshape(p_1, 1)
        u_t = w.T @ x_t
        d_t = Y[:, t].reshape(-1, 1)
        eq += np.sum((d_t - u_t) ** 2)
    return eq / (2 * N)

def simplePerceptron(X, Y, epocas_max=200, lr=0.05):
    X = X.T  # (N, p)
    N, p = X.shape  
    C = Y.shape[0]

    X_normalized = np.concatenate((-np.ones((N, 1)), X), axis=1)  # (N, p+1)
    W = np.random.random_sample((p + 1, C)) - 0.5

    erro = True
    epoca = 0

    while erro and epoca < epocas_max:
        erro = False
        for t in range(N):
            x_t = X_normalized[t, :].reshape(-1, 1)  
            u_t = W.T @ x_t  
            y_t = np.argmax(u_t)    
            d_t = np.argmax(Y[:, t]) 
            if y_t != d_t:
                W[:, d_t] += lr * x_t.flatten()
                W[:, y_t] -= lr * x_t.flatten()
                erro = True
        epoca += 1

    return W

def ADALINE(X, Y, epocas_max=200, lr=0.05, pr=0.05):
    p, N = X.shape
    C = Y.shape[0]

    X_normalized = np.concatenate((-np.ones((1, N)), X), axis=0)  # (p+1, N)
    W = np.random.random_sample((p + 1, C)) - 0.5

    EQM1 = 1
    EQM2 = 0
    epoca = 0

    while epoca < epocas_max and abs(EQM1 - EQM2) > pr:
        EQM1 = EQM(X_normalized, Y, W)
        for t in range(N):
            x_t = X_normalized[:, t].reshape(p + 1, 1)
            u_t = W.T @ x_t
            d_t = Y[:, t].reshape(-1, 1)
            e_t = d_t - u_t
            W += lr * x_t @ e_t.T

        epoca += 1
        EQM2 = EQM(X_normalized, Y, W)

    return W  

def softmax(Z):
    expZ = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    return expZ / np.sum(expZ, axis=0, keepdims=True)

def MLP(X, Y, hidden_layers=[100, 100], learning_rate=1e-4, epocas_max=5000, activation='relu'):

    p, N = X.shape
    C = Y.shape[0]  # Número de classes
    layers = [p] + hidden_layers + [C]

    weights = []
    biases = []

    # Inicialização He
    for i in range(len(layers) - 1):
        w = np.random.randn(layers[i+1], layers[i]) * np.sqrt(2 / layers[i])
        b = np.zeros((layers[i+1], 1))
        weights.append(w)
        biases.append(b)

    def relu(x):
        return np.maximum(0, x)
    def relu_deriv(a):
        return (a > 0).astype(float)

    losses = []
    precisions.clear()

    for epoch in range(epocas_max):
        activations = [X]
        zs = []
        for l in range(len(weights)):
            z = np.dot(weights[l], activations[-1]) + biases[l]
            zs.append(z)
            if l < len(weights)-1:
                a = relu(z)
            else:
                a = softmax(z)  # Última camada: softmax
            activations.append(a)

        y_pred = activations[-1]

        loss = -np.mean(np.sum(Y * np.log(y_pred+1e-12), axis=0))
        losses.append(loss)

        # Backpropagation
        deltas = [y_pred - Y] 
        for l in range(len(weights)-2, -1, -1):
            delta = np.dot(weights[l+1].T, deltas[0]) * relu_deriv(activations[l+1])
            deltas.insert(0, delta)

        for l in range(len(weights)):
            dw = np.dot(deltas[l], activations[l].T) / N
            db = np.mean(deltas[l], axis=1, keepdims=True)
            weights[l] -= learning_rate * dw
            biases[l] -= learning_rate * db

    model = {'weights': weights, 'biases': biases, 'activation':'relu'}
    return model, precisions

def MLP_predict(model, X):
    def relu(x):
        return np.maximum(0, x)

    weights = model['weights']
    biases = model['biases']

    a = X
    for l in range(len(weights)-1):
        z = np.dot(weights[l], a) + biases[l]
        a = relu(z)

    # Ultima camada softmax
    z = np.dot(weights[-1], a) + biases[-1]
    y_pred = np.exp(z - np.max(z, axis=0, keepdims=True))
    y_pred = y_pred / np.sum(y_pred, axis=0, keepdims=True)
    return y_pred