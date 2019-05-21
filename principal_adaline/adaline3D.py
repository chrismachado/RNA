
import copy

import matplotlib.pyplot as plt
import numpy as np
import yaml

from perceptron_adaline.adaline import Adaline
from util.artificial_dataset import Artificial
from util.realiza_adaline import executar
from util.utilidade import Utilidade


def main():
    stream = open('./configuracoes/config_execucao.yml', 'r', encoding='utf-8').read()
    config_exec = yaml.load(stream=stream)

    a, b, c = config_exec['abc']
    realizacoes = config_exec['realizacoes']
    eta = config_exec['eta']
    epocas = config_exec['epocas']
    ruido = config_exec['ruido']

    ada = Adaline(eta=eta, epochs=epocas, base_treino=0.8)

    X, y, N = Artificial().function3d(a=a, b=b, c=c, ruido=ruido)
    y = np.reshape(y, (y.shape[0], 1))

    Utilidade().normalize_(X[:, 0:2])
    Utilidade().normalize_(y)

    xx = copy.deepcopy(X[:, 0])
    yy = copy.deepcopy(X[:, 1])

    ada.shuffle_(X, y)

    ada = executar(X, y, clf=ada, num=realizacoes)
    w = ada.w_

    plt.plot(range(len(ada.custo)), ada.custo, ls='-')
    plt.title('Curva de aprendizado')
    plt.xlabel('Época')
    plt.ylabel('Custo')
    plt.show()

    zz = w[1] * xx + w[2] * yy + w[0]

    fig = plt.figure()
    plt3d = fig.gca(projection='3d')
    plt3d.cla()

    # plt3d.scatter(X_train[:, 0], X_train[:, 1], y_train, color='red', alpha=1.0, facecolors='none')
    plt3d.scatter(X[:, 0], X[:, 1], y, color='red', alpha=1.0)

    xx = np.reshape(xx, (N, N))
    yy = np.reshape(yy, (N, N))
    zz = np.reshape(zz, (N, N))

    plt3d.plot_surface(xx, yy, zz, rstride=10, cstride=10, antialiased=True,
                       color='blue')
    plt3d.set_xlabel('x1')
    plt3d.set_ylabel('x2')
    plt3d.set_zlabel('y')

    plt.show()

main()