import copy
import math
import numpy as np
import pandas as pd

df = pd.read_csv('breast_cancer.csv')
X = df[['Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape', 'Marginal Adhesion',
        'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli', 'Mitoses']].to_numpy()
mu = X.mean()
sigma = X.std()
X = (X - mu) / sigma
Y = df['Class'].to_numpy()
Y = Y / 2 - 1

global w_in, w_in
w_in = np.zeros_like(X[0])
b_in = 0.0

learning_rate = 0.001
iterations = 1000
regularization = 0.5
k = 10


def model(x, w, b):
    z = np.dot(x, w) + b
    return 1 / (1 + np.exp(-z))


def cost(x, y, w, b, lambda_):
    m, n = x.shape
    c = 0.0
    for i in range(m):
        f_wb_i = model(x[i], w, b)
        c += (-y[i] * np.log(f_wb_i)) - ((1 - y[i]) * np.log(1 - f_wb_i))
    reg_cost = 0
    for j in range(n):
        reg_cost += w[j] ** 2

    return c / m + (lambda_ / (2 * m)) * reg_cost


def gradient(x, y, w, b, lambda_):
    m, n = x.shape
    dj_dw = np.zeros(n, )
    dj_db = 0
    for i in range(m):
        err = model(x[i], w, b) - y[i]
        dj_db += err
        for j in range(n):
            dj_dw_aux = err * x[i, j]
            dj_dw[j] += dj_dw_aux
    dj_dw /= m
    dj_db /= m
    for j in range(n):
        dj_dw[j] += (lambda_ / m) * w[j]
    return dj_db, dj_dw


def gradient_descent(x, y, w, b, alpha, iters, lambda_):
    J_wb = []
    w_aux = copy.deepcopy(w)
    b_aux = b
    for i in range(iters):
        dj_db, dj_dw = gradient(x, y, w_aux, b_aux, lambda_)
        w_aux = w_aux - alpha * dj_dw
        b_aux = b_aux - alpha * dj_db
        if i < 100000:  # prevent resource exhaustion
            J_wb.append(cost(x, y, w_aux, b_aux, lambda_))
        # Print cost every at intervals 10 times or as many iterations if < 10
        if i % math.ceil(iters / 10) == 0:
            print(f"Iteration {i:4d}: Cost {J_wb[-1]}   ")
    return w_aux, b_aux


def predict(x, w, b):
    m, n = x.shape
    p = np.zeros(m)
    for i in range(m):
        f_wb = model(x[i], w, b)
        if f_wb >= 0.5:
            p[i] = 1
        else:
            p[i] = 0
    return p


def cross_validate(x, y, k, learning_rate, iterations, regularization):
    n_samples = x.shape[0]
    fold_size = n_samples // k
    accuracy = 0
    for i in range(k):
        x_test = x[i * fold_size:(i + 1) * fold_size, :]
        y_test = y[i * fold_size:(i + 1) * fold_size]
        x_train = np.concatenate((x[:i * fold_size, :], x[(i + 1) * fold_size:, :]), axis=0)
        y_train = np.concatenate((y[:i * fold_size], y[(i + 1) * fold_size:]), axis=0)
        w_out, b_out = gradient_descent(x_train, y_train, w_in, b_in, learning_rate, iterations, regularization)
        pr = predict(x_test, w_out, b_out)
        accuracy += np.mean(pr == y_test) * 100 / k
    return accuracy


print('Train Accuracy: ', cross_validate(X, Y, k, learning_rate, iterations, regularization))
