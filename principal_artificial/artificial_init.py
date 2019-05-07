#encoding:utf-8

from perceptron_simples.perceptron import Perceptron
from util.artificial_dataset import Artificial
from util.utilidade import *

def main():
    ppn = Perceptron(epochs=100, eta=0.1)
    artificial = Artificial()

    X, y = artificial.and_problem(npc=10, noise=0.2)

    #Shuffle
    ppn.shuffle_(X, y)

    X_new, y_new, accuracy, weights, dmm, erros, imax_, imin_ = execution(X, y, clf=ppn, num=20)

    ppn.w_ = weights[imin_]
    plot_decision(X=X_new[imin_][1], y=y_new[imin_][1], clf=ppn, title="Superficie de decisao [%d] Artificial" % (imin_ + 1),
                  xlabel="Entrada 1", ylabel="Entrada 2")
    plot_learning_bend(erros[imin_], title="Curva de aprendizado no pior caso[%d]" % (imin_ + 1), xlabel="Épocas", ylabel="Erros")

    ppn.w_ = weights[imax_]
    plot_decision(X=X_new[imax_][1], y=y_new[imax_][1], clf=ppn, title="Superficie de decisao [%d] Artificial Teste" % (imax_ + 1),
                  xlabel="Entrada 1", ylabel="Entrada 2")
    plot_learning_bend(erros[imax_], title="Curva de aprendizado no melhor caso[%d]" % (imax_ + 1), xlabel="Épocas", ylabel="Erros")

    plot_decision(X=X_new[imax_][0], y=y_new[imax_][0], clf=ppn, title="Superficie de decisao [%d] Artificial Treinamento" % (imax_ + 1),
                  xlabel="Entrada 1", ylabel="Entrada 2")

main()