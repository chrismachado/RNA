#encoding:utf-8

import numpy as np
import yaml

from perceptron_simples.perceptron import Perceptron
from util.artificial_dataset import Artificial
from util.utilidade import Utilidade


def main():
    stream = open('./configuracoes/config_execucao.yml', 'r', encoding='utf-8').read()

    config_exec = yaml.load(stream=stream)

    classe = config_exec['classe']
    realizacoes = config_exec['realizacoes']
    eta = config_exec['eta']
    epocas = config_exec['epocas']
    base_treino = config_exec['base_treino']
    verb = config_exec['verb']
    log = config_exec['log']
    npc = config_exec['num_por_classe']
    ruido = config_exec['ruido']

    ppn = Perceptron(epochs=epocas, eta=eta, base_treino=base_treino)
    util = Utilidade(verb=verb, log=log)
    artificial = Artificial()

    if classe == 'and':
        X, y = artificial.and_problem(npc=npc, ruido=ruido)
    elif classe == 'or':
        X, y = artificial.or_problem(npc=npc, ruido=ruido)
    else:
        X, y = artificial.and_problem(npc=npc, ruido=ruido)

    #Shuffle
    ppn.shuffle_(X, y)

    X_new, y_new, accuracy, weights, dmm, erros, imax_, imin_ = util.execution(X, y, clf=ppn, num=realizacoes)

    try:
        ppn.w_ = weights[imin_]
        util.plot_decision(X=X_new[imin_][1], y=y_new[imin_][1], clf=ppn,
                           title="Superficie de decisao [%d] %s" % (imin_ + 1, classe),
                           xlabel="Entrada 1", ylabel="Entrada 2")
        util.plot_learning_bend(erros[imin_],
                                title="Curva de aprendizado no pior caso [%d] %s" % (imin_ + 1, classe),
                                xlabel="Épocas",
                                ylabel="Erros")

        ppn.w_ = weights[imax_]
        util.plot_decision(X=X_new[imax_][1], y=y_new[imax_][1], clf=ppn,
                           title="Superficie de decisao [%d] %s " % (imax_ + 1, classe),
                           xlabel="Entrada 1", ylabel="Entrada 2")

        util.plot_learning_bend(erros[imax_],
                                title="Curva de aprendizado no melhor caso [%d] %s" % (imax_ + 1, classe),
                                xlabel="Épocas",
                                ylabel="Erros")

        X = np.concatenate((X_new[imax_][0], X_new[imax_][1]))
        y = np.concatenate((y_new[imax_][0], y_new[imax_][1]))

        util.plot_decision(X=X, y=y, clf=ppn,
                           title="Superficie de decisao [%d] - Base completa %s" % (imax_ + 1, classe.upper()),
                           xlabel="Entrada 1", ylabel="Entrada 2", X_highlight=X_new[imax_][1])

    except Exception as e:
        print('\n\n################################################################################')
        print('# Para que o gráfico seja plotado, é necessário que hajam apenas 2 atributos.  #')
        print('################################################################################')


if __name__ == '__main__':
    main()