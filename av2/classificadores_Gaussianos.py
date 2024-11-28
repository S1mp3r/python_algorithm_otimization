import numpy as np

def MQO(X, y):
    # Cálculo dos coeficientes do MQO
    return np.linalg.pinv(X.T @ X) @ X.T @ y

def gaussian_density(x, mean, inv_cov, log_det_cov, case):
    if case == 1:
        diff = x - mean
        return - 0.5 * log_det_cov - 0.5 * diff @ inv_cov @ diff
    if case == 2:
        diff = x - mean
        return diff @ inv_cov @ diff

def classificador_Gaussiano_Trad(X_test, X_train, Y_train, c):
    N, p = X_train.shape
    means = []
    inv_covs = []
    log_det_covs = []
    classes = [1,2,4,5]

    # Pré-computação das médias e covariâncias por classe
    for i in classes:
        class_data = X_train[Y_train == i]
        n_i = class_data.shape[0]
        if n_i == 0:
            continue
        mean = np.mean(class_data, axis=0)
        cov = np.cov(class_data, rowvar=False)
        inv_cov = np.linalg.pinv(cov)
        log_det_cov = np.log(np.linalg.det(cov))

        means.append(mean)
        inv_covs.append(inv_cov)
        log_det_covs.append(log_det_cov)

    # Cálculo das probabilidades para cada teste
    predictions = []
    for x in X_test:
        probabilities = np.array([
            gaussian_density(x, mean, inv_cov, log_det_cov, case=1)
            for mean, inv_cov, log_det_cov in zip(
                means, inv_covs, log_det_covs)
        ])
        predictions.append(np.argmax(probabilities) + 1)
    return np.array(predictions)

def classificador_Gaussiano_Cov_Iguais(X_test, X_train, Y_train, c):
    N, p = X_train.shape
    # Covariância comum para todas as classes
    cov = np.cov(X_train, rowvar=False)
    inv_cov = np.linalg.pinv(cov)

    # Pré-computação das médias
    means = []
    for i in range(1, c + 1):
        class_data = X_train[Y_train == i]
        if class_data.shape[0] == 0:
            continue
        mean = np.mean(class_data, axis=0)
        means.append(mean)

    # Cálculo das distâncias de Mahalanobis
    predictions = []
    for x in X_test:
        probabilities = np.array([
            gaussian_density(x, mean, inv_cov, 0, case=2)
            for mean in means
        ])
        prediction = np.argmin(probabilities) + 1
        predictions.append(prediction)
    return np.array(predictions)

def classificador_Gaussiano_Matriz_Agregada(X_test, X_train, Y_train, c):
    N, p = X_train.shape
    means = []
    covariances = []
    n_samples_per_class = []

    # Pré-computação das médias e covariâncias por classe
    for i in range(1, c + 1):
        class_data = X_train[Y_train == i]
        n_i = class_data.shape[0]
        if n_i == 0:
            continue
        mean = np.mean(class_data, axis=0)
        cov = np.cov(class_data, rowvar=False)
        means.append(mean)
        covariances.append(cov)
        n_samples_per_class.append(n_i)

    # Covariância agregada
    cov_agregada = sum((n_i / N) * cov for cov, n_i in zip(covariances, n_samples_per_class))
    inv_cov_agregada = np.linalg.pinv(cov_agregada)

    predictions = []
    for x in X_test:
        probabilities = np.array([
            gaussian_density(x, mean, inv_cov_agregada, 0, case=2)
            for mean in means
        ])
        prediction = np.argmin(probabilities) + 1
        predictions.append(prediction)
    return np.array(predictions)

def classificador_Gaussiano_Friedman(X_test, X_train, Y_train, c, lamb):
    N, p = X_train.shape
    means = []
    covariances = []
    n_samples_per_class = []

    # Pré-computação das médias e covariâncias por classe
    for i in range(1, c + 1):
        class_data = X_train[Y_train == i]
        n_i = class_data.shape[0]
        if n_i == 0:
            continue
        mean = np.mean(class_data, axis=0)
        cov = np.cov(class_data, rowvar=False)
        means.append(mean)
        covariances.append(cov)
        n_samples_per_class.append(n_i)

    # Regularização de Friedman
    cov_agregada = sum((n_i / N) * cov for cov, n_i in zip(covariances, n_samples_per_class))
    covariances_regs = []
    inv_cov_regs = []
    log_det_cov_regs = []
    for cov, n_i in zip(covariances, n_samples_per_class):
        numerator = (1 - lamb) * n_i * cov + lamb * N * cov_agregada
        denominator = (1 - lamb) * n_i + lamb * N
        cov_reg = numerator / denominator
        inv_cov_reg = np.linalg.pinv(cov_reg)
        log_det_cov_reg = np.log(np.linalg.det(cov_reg))
        covariances_regs.append(inv_cov_reg)
        inv_cov_regs.append(inv_cov_reg)
        log_det_cov_regs.append(log_det_cov_reg)

    predictions = []
    if lamb != 1:
        for x in X_test:
            probabilities = np.array([
                gaussian_density(x, mean, inv_cov_reg, log_det_cov_reg, case=1)
                for mean, inv_cov_reg, log_det_cov_reg in zip(
                    means, inv_cov_regs, log_det_cov_regs)
            ])
            predictions.append(np.argmax(probabilities) + 1)
    else:
        inv_cov_agregada = np.linalg.pinv(cov_agregada)
        for x in X_test:
            probabilities = np.array([
                gaussian_density(x, mean, inv_cov_agregada, 0, case=2)
                for mean in means
            ])
            prediction = np.argmin(probabilities) + 1
            predictions.append(prediction)
    return np.array(predictions)

def classificador_Gaussiano_Naive_Bayes(X_test, X_train, Y_train, c):
    N, p = X_train.shape
    means = []
    inv_vars = []
    classes = [1,2,4,5]

    # Pré-computação das médias e variâncias por classe
    for i in classes:
        class_data = X_train[Y_train == i]
        n_i = class_data.shape[0]
        if n_i == 0:
            continue
        mean = np.mean(class_data, axis=0)
        var = np.var(class_data, axis=0)
        inv_var = 1 / var
        means.append(mean)
        inv_vars.append(inv_var)

    # Cálculo das probabilidades de classificação
    predictions = []
    for x in X_test:
        probabilities = np.array([
            - 0.5 * np.sum(np.log(2 * np.pi * 1 / inv_var)) -
            0.5 * np.sum((x - mean) ** 2 * inv_var)
            for mean, inv_var in zip(means, inv_vars)
        ])
        predictions.append(np.argmax(probabilities) + 1)
    return np.array(predictions)
