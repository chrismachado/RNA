import copy as cp

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from perceptron_nneuronios.perceptron_nneuronios import PerceptronNN
from util.utilidadeSG import UtilidadeSG


def main():
    stream = open('./configuracoes/config_execucao.yml', 'r', encoding='utf-8').read()

    config_exec = yaml.load(stream=stream)

    iris_classe = config_exec['iris_classe']
    realizacoes = config_exec['realizacoes']
    eta = config_exec['eta']
    epocas = config_exec['epocas']
    base_treino = config_exec['base_treino']
    atributos = config_exec['atributos']
    verb = config_exec['verb']
    log = config_exec['log']
    curva = config_exec['curva_aprendizado']
    type_y = config_exec['type_y']

    ppn = PerceptronNN(epocas=epocas, eta=eta, base_treino=base_treino, n=3, type_y=type_y)

    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'

    df = pd.read_csv(url, header=None)
    y = df.iloc[:, 4].values

    y_ = np.zeros((y.shape[0], 3))
    # print(y_.shape)
    # print(y_)

    if type_y == 'tanh':
        k = -1
    else:
        k = 0

    for i in range(y.shape[0]):
        if y[i] == 'Iris-setosa':
            y_[i] = [1, k, k]
        elif y[i] == 'Iris-versicolor':
            y_[i] = [k, 1, k]
            # y[i] = [0, 1, 0]
        elif y[i] == 'Iris-virginica':
            y_[i] = [k, k, 1]
            # y[i] = [0, 0, 1]
    # print(y_)
    # y = np.reshape(y, (150, 1))

    # atributos de treinamento
    X = df.iloc[:, atributos].values

    xx = cp.deepcopy(X)
    UtilidadeSG().normalize_(xx)

    if xx.shape[1] == 2:
        plt.plot(xx[:50, 0], xx[:50, 1], 'bo', mec='k', markersize=5)
        plt.plot(xx[50:100, 0], xx[50:100, 1], 'r*', mec='k')
        plt.plot(xx[100:, 0], xx[100:, 1], 'g^', mec='k')

    UtilidadeSG().normalize_(X)

    # # shuffle_
    ppn.shuffle_(X, y_)
    #
    # ppn = ppn.fit(X, y_)
    UtilidadeSG().execution(X=X, y=y_, clf=ppn, num=realizacoes)

    # print(ppn.w_)


if __name__ == '__main__':
    main()
