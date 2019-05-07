#encoding:utf-8

import pandas as pd

from perceptron_simples.perceptron import Perceptron
from util.utilidade import *


def main():
    ppn = Perceptron(epochs=100, eta=0.1)

    df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)

    y = df.iloc[0:150, 4].values

    y = np.where(y == 'Iris-versicolor', 0, 1)

    # sepal length and sepal weight
    X = df.iloc[0:150, [0, 1, 2, 3]].values

    #shuffle_
    ppn.shuffle_(X, y)

    X_new, y_new, accuracy, weights, dmm, erros, imax_, imin_ = execution(X, y, clf=ppn, num=20)
    #
    # ppn.w_ = weights[imin_]
    # plot_decision(X=X_new[imin_][1], y=y_new[imin_][1], clf=ppn, title="Superficie de decisao [%d]  Versicolor x Outras" % (imin_ + 1),
    #               xlabel="Comprimento Sépala", ylabel="Largura Sépala")
    # plot_learning_bend(erros[imin_], title="Curva de aprendizado no pior caso [%d]" % (imin_ + 1), xlabel="Épocas",
    #                    ylabel="Erros")
    #
    # ppn.w_ = weights[imax_]
    # plot_decision(X=X_new[imax_][1], y=y_new[imax_][1], clf=ppn, title="Superficie de decisao [%d]  Versicolor x Outras" % (imax_ + 1),
    #               xlabel="Comprimento Sépala", ylabel="Largura Sépala")
    # plot_learning_bend(erros[imax_], title="Curva de aprendizado no melhor caso[%d]" % (imax_ + 1), xlabel="Épocas",
    #                    ylabel="Erros")
    #
    #
    # plot_decision(X=X_new[imax_][0], y=y_new[imax_][0], clf=ppn,
    #               title="Superficie de decisao [%d]  Versicolor x Outras" % (imax_ + 1),
    #               xlabel="Comprimento Sépala", ylabel="Largura Sépala")


main()