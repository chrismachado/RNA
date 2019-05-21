
import numpy as np
import pandas as pd
import yaml

from perceptron_simples.perceptron import Perceptron
from util.utilidade import Utilidade


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

    ppn = Perceptron(epochs=epocas, eta=eta, base_treino=base_treino)
    util = Utilidade(verb=verb, log=log)

    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'

    df = pd.read_csv(url, header=None)
    y = df.iloc[:, 4].values

    y = np.where(y == iris_classe, 0, 1)

    # atributos de treinamento
    X = df.iloc[:, atributos].values

    # shuffle_
    ppn.shuffle_(X, y)

    X_new, y_new, accuracy, weights, dmm, erros, imax_, imin_ = util.execution(X, y, clf=ppn, num=realizacoes)

    if curva == 's':
        util.plot_learning_bend(erros[imin_],
                                title="Curva de aprendizado no pior caso [%d] %s" % (imin_ + 1, iris_classe),
                                xlabel="Épocas",
                                ylabel="Erros")
        util.plot_learning_bend(erros[imax_],
                                title="Curva de aprendizado no melhor caso [%d] %s" % (imax_ + 1, iris_classe),
                                xlabel="Épocas",
                                ylabel="Erros")

    try:

        ppn.w_ = weights[imin_]
        util.plot_decision(X=X_new[imin_][1], y=y_new[imin_][1], clf=ppn, title="Superficie de decisao [%d] %s" % (imin_ + 1, iris_classe.upper()),
                      xlabel="Comprimento Sépala", ylabel="Largura Sépala")

        ppn.w_ = weights[imax_]
        util.plot_decision(X=X_new[imax_][1], y=y_new[imax_][1], clf=ppn,
                      title="Superficie de decisao [%d] %s " % (imax_ + 1, iris_classe),
                      xlabel="Comprimento Sépala", ylabel="Largura Sépala")


        X = np.concatenate((X_new[imax_][0], X_new[imax_][1]))
        y = np.concatenate((y_new[imax_][0], y_new[imax_][1]))

        util.plot_decision(X=X, y=y, clf=ppn,
                           title="Superficie de decisao [%d] - Base completa %s" % (imax_ + 1, iris_classe.upper()),
                           xlabel="Entrada 1", ylabel="Entrada 2", X_highlight=X_new[imax_][1])

    except Exception as e:
        print('\n\n################################################################################')
        print('# Para que o gráfico seja plotado, é necessário que hajam apenas 2 atributos.  #')
        print('################################################################################')


if __name__ == '__main__':
    main()