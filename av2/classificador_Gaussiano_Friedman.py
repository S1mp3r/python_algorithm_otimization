import numpy as np
import matplotlib.pyplot as plt

def gaussian_density(x, mean, cov, inv_cov, reg=False):

    if reg:
        return 
    return -1/2*np.log(np.linalg.det(cov)) - 1/2*(mean - x).T@inv_cov@(mean - x)

def gaussian_density_lambdaOne(x, mean, cov, inv_cov):
    return (mean - x).T@inv_cov@(mean - x)


def calcular_covariancia_agregada(X_treino, y_treino, classes):

    cov_agregado = np.zeros((X_treino.shape[1], X_treino.shape[1]))
    for classe in classes:
        X_classe = X_treino[y_treino == classe]
        cov_classe = np.cov(X_classe.T)
        p_classe = len(X_classe) / len(X_treino)
        cov_agregado += p_classe * cov_classe
    return cov_agregado

def classificador_gaussiano_Friedman(X, means, covariances, lambdas):
    predictions = []
    inv_cov = []
    
    for covar in covariances:
        inv_cov.append(np.linalg.inv(covar))

    if lambdas == 0:
        pass
    elif lambdas == 1:
        for x in X:
            probabilities = [gaussian_density(x, mean, cov, inv_cov) for mean, cov, inv_cov in zip(means, covariances, inv_cov)]
            predictions.append(np.argmax(probabilities) + 1)
        return np.array(predictions)

    for x in X:
        probabilities = [gaussian_density(x, mean, cov, inv_cov) for mean, cov, inv_cov in zip(means, covariances, inv_cov)]
        predictions.append(np.argmax(probabilities) + 1)
    
    return np.array(predictions)
