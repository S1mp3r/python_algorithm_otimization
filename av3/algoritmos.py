import numpy as np

precisions = []

def sign(u):
    return 1 if u >= 0 else -1

def sign_ajustavel(u, first, second, third):
    return first if u >= second else third

def EQM(X, Y, w):
    p_1, N = X.shape
    eq = 0
    for t in range(N):
        x_t = X[:, t].reshape(p_1, 1)
        u_t = w.T @ x_t
        d_t = Y[t]
        eq += (d_t - u_t[0, 0]) ** 2
    return eq / (2 * N)


def simplePerceptron(x_raw, y_raw, epocas_max = 200, lr = 0.05):
    x_raw = x_raw.T  # (p, N)
    y_raw = y_raw.T  # (1, N)

    x_normalized = x_raw

    p, N = x_raw.shape

    x_normalized = np.concatenate((-np.ones((1, N)), x_normalized))

    w = np.random.random_sample((p + 1, 1)) - 0.5

    w_list = []
    erro = True
    epoca = 0

    while erro and epoca < epocas_max:
        erro = False
        for t in range(N):
            x_t = x_normalized[:, t].reshape(p + 1, 1)
            u_t = (w.T @ x_t)[0, 0]
            y_t = sign(u_t)
            d_t = float(y_raw[t])
            e_t = d_t - y_t
            w += (lr * e_t * x_t) / 2

            if y_t != d_t:
                erro = True
        w_list.append(w)
        epoca += 1

    return w_list

def ADALINE(x_raw, y_raw, epocas_max = 200, lr = 0.05, pr = 0.05):
    x_raw = x_raw.T  # (p, N)
    y_raw = y_raw.T  # (1, N)

    x_normalized = x_raw

    p, N = x_raw.shape

    x_normalized = np.concatenate((-np.ones((1, N)), x_normalized))
    
    w = np.random.random_sample((p + 1, 1)) - 0.5
    w_list = []

    EQM1 = 1
    EQM2 = 0

    epoca = 0
    while epoca < epocas_max and abs(EQM1 - EQM2) > pr:
        EQM1 = EQM(x_normalized, y_raw, w)

        for t in range(N):
            x_t = x_normalized[:, t].reshape(p + 1, 1)
            u_t = w.T @ x_t
            d_t = y_raw[t]
            e_t = d_t - u_t
            w += lr * e_t * x_t
        
        epoca += 1
        w_list.append(w)
        EQM2 = EQM(x_normalized, y_raw, w)

    return w_list

def MLP(x_raw, y_raw, hidden_layers=[25], learning_rate=0.1, epocas_max=200, activation='tanh', precision=1e-6):
    x_raw = x_raw.T
    y_raw = y_raw.reshape(1, -1)

    x_normalized = x_raw

    p, N = x_raw.shape
    n_output = 1

    layers = [p] + hidden_layers + [n_output]

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
        activations = [x_normalized]
        zs = []

        # Forward pass
        for w, b in zip(weights, biases):
            z = np.dot(w, activations[-1]) + b
            zs.append(z)
            a = activation_func(z)
            activations.append(a)

        y_pred = activations[-1]
        loss = np.mean((y_pred - y_raw) ** 2) / 2
        losses.append(loss)
    
        current_precision = abs(losses[-1] - losses[-2]) if len(losses) > 1 else np.inf

        precisions.append(current_precision)

        if epoch > 0 and current_precision < precision:
            break

        # Backpropagation
        deltas = [None] * len(weights)
        delta = (y_pred - y_raw) * activation_derivative(y_pred)
        deltas[-1] = delta

        for l in range(len(weights) - 2, -1, -1):
            delta = np.dot(weights[l + 1].T, deltas[l + 1]) * activation_derivative(activations[l + 1])
            deltas[l] = delta

        for l in range(len(weights)):
            weights[l] -= learning_rate * np.dot(deltas[l], activations[l].T) / N
            biases[l] -= learning_rate * np.mean(deltas[l], axis=1, keepdims=True)

    model = {'weights': weights, 'biases': biases, 'activation_func': activation_func}
    return model, precisions


def MLP_predict(model, x_raw):
    x_raw = x_raw.T
    
    x_normalized = x_raw

    activations = [x_normalized]
    weights = model['weights']
    biases = model['biases']
    activation_func = model['activation_func']

    for w, b in zip(weights, biases):
        z = np.dot(w, activations[-1]) + b
        a = activation_func(z)
        activations.append(a)

    y_pred = activations[-1]

    return y_pred.T # (N,1)


