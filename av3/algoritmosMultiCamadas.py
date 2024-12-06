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
        d_t = Y[t, :].reshape(-1, 1)
        eq += np.sum((d_t - u_t) ** 2)
    return eq / (2 * N)

def simplePerceptron(X, Y, epocas_max=200, lr=0.05):
    X = X.T  # (p, N)
    Y = Y.T  # (C, N)

    p, N = X.shape
    C = Y.shape[1]  # Número de classes

    X_normalized = np.concatenate((-np.ones((1, N)), X))  # Adiciona bias
    W = np.random.random_sample((p + 1, C)) - 0.5  # Peso para cada classe

    erro = True
    epoca = 0

    while erro and epoca < epocas_max:
        erro = False
        for t in range(p):
            x_t = X_normalized[:, t].reshape(p + 1, 1)
            u_t = W.T @ x_t
            y_t = np.argmax(u_t)
            d_t = np.argmax(Y[t, :])
            if y_t != d_t:
                W[:, d_t] += lr * x_t.flatten()
                W[:, y_t] -= lr * x_t.flatten()
                erro = True
        epoca += 1

    return W

def ADALINE(X, Y, epocas_max=200, lr=0.05, pr=0.05):
    X = X.T  # (p, N)
    Y = Y.T  # (C, N)

    p, N = X.shape
    C = Y.shape[0]  # Número de classes

    X_normalized = np.concatenate((-np.ones((1, N)), X))  # Adiciona bias
    W = np.random.random_sample((p + 1, C)) - 0.5  # Peso para cada classe

    EQM1 = 1
    EQM2 = 0
    epoca = 0

    while epoca < epocas_max and abs(EQM1 - EQM2) > pr:
        EQM1 = EQM(X_normalized, Y, W)

        for t in range(N):
            x_t = X_normalized[:, t].reshape(p + 1, 1)
            u_t = W.T @ x_t
            d_t = Y[t, :].reshape(-1, 1)
            e_t = d_t - u_t
            W += lr * x_t @ e_t.T

        epoca += 1
        EQM2 = EQM(X_normalized, Y, W)

    return W

def MLP(X, Y, hidden_layers=[25], learning_rate=0.1, epocas_max=200, activation='tanh', precision=1e-6):
    X = X.T
    Y = Y.T

    p, N = X.shape
    C = Y.shape[0]  # Número de classes

    layers = [p] + hidden_layers + [C]

    weights = []
    biases = []
    for i in range(len(layers) - 1):
        if activation == 'tanh':
            w = np.random.randn(layers[i + 1], layers[i]) * np.sqrt(2 / layers[i])

        b = np.zeros((layers[i + 1], 1))
        weights.append(w)
        biases.append(b)

    if activation == 'tanh':
        activation_func = lambda x: np.tanh(x)
        activation_derivative = lambda a: 1 - a ** 2

    losses = []

    for epoch in range(epocas_max):
        activations = [X]
        zs = []

        # Forward pass
        for w, b in zip(weights, biases):
            z = np.dot(w, activations[-1]) + b
            zs.append(z)
            a = activation_func(z)
            activations.append(a)

        y_pred = activations[-1]
        loss = np.mean((y_pred - Y) ** 2) / 2
        losses.append(loss)
    
        current_precision = abs(losses[-1] - losses[-2]) if len(losses) > 1 else np.inf

        precisions.append(current_precision)

        if epoch > 0 and current_precision < precision:
            break

        # Backpropagation
        deltas = [None] * len(weights)
        delta = (y_pred - Y) * activation_derivative(y_pred)
        deltas[-1] = delta

        for l in range(len(weights) - 2, -1, -1):
            delta = np.dot(weights[l + 1].T, deltas[l + 1]) * activation_derivative(activations[l + 1])
            deltas[l] = delta

        for l in range(len(weights)):
            weights[l] -= learning_rate * np.dot(deltas[l], activations[l].T) / N
            biases[l] -= learning_rate * np.mean(deltas[l], axis=1, keepdims=True)

    model = {'weights': weights, 'biases': biases, 'activation_func': activation_func}
    return model, precisions

def MLP_predict(model, X):
    X = X.T

    activations = [X]
    weights = model['weights']
    biases = model['biases']
    activation_func = model['activation_func']

    for w, b in zip(weights, biases):
        z = np.dot(w, activations[-1]) + b
        a = activation_func(z)
        activations.append(a)

    y_pred = activations[-1]

    return y_pred.T  # (N, C)