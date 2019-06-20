#encoding:utf-8

import copy

import matplotlib.pyplot as plt
import yaml

from perceptron_nneuronios.perceptron_nneuronios import PerceptronNN
from util.artificial_dataset import Artificial
from util.utilidadeSG import UtilidadeSG


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
    type_d = config_exec['type_y']

    ppn = PerceptronNN(epocas=epocas, eta=eta, base_treino=base_treino, n=3, type_y=type_d)
    util = UtilidadeSG()
    artificial = Artificial()
    X, y = artificial.artificial_sigmoide(npc=npc, ruido=ruido, type_d=type_d)

    UtilidadeSG().normalize_(X)
    xx = copy.deepcopy(X)
    yy = copy.deepcopy(y)

    ppn.shuffle_(X, y)

    #Plot
    plt.plot(xx[:npc, 0], xx[:npc, 1], 'bo', mec='k')
    plt.plot(xx[npc:2*npc, 0], xx[npc:2*npc, 1], 'r*', mec='k')
    plt.plot(xx[2*npc:, 0], xx[2*npc:, 1], 'g^', mec='k')
    # plt.show()

    UtilidadeSG().execution(X, y, clf=ppn, num=realizacoes)
    # X_test, X_train = ppn.get_test_sample(X), ppn.get_train_sample(X)
    # y_test, y_train = ppn.get_test_sample(y), ppn.get_train_sample(y)
    #
    # ppn = ppn.fit(X_train, y_train)
    #
    # ppn.test(X_test, y_test)

    # print(ppn.w_)
    # y = np.reshape(y, (3 * npc, 3))
    # print(ppn.w_.T)
    # # print(y[1] == [0, 0, 1])
    # yn = []
    # for i in range(y.shape[0]):
    #     if y[i].tolist() == [1, 0, 0]:
    #         yn.append(0)
    #     elif y[i].tolist() == [0, 1, 0]:
    #         yn.append(1)
    #     elif y[i].tolist() == [0, 0, 1]:
    #         yn.append(2)

    # print(yn)
    # y = np.array(yn)

    # ppn.plotData2d()


    # plt.axis([0, 1, 0, 1])

    # plt.plot(xx, zz1, ls='-')
    # plt.show()

    # UtilidadeSG().plot_decision(xx, ppn)
    # print(xx.shape)
    # print(y.shape)
    # plot_decision_regions(X=X, y=y, clf=ppn)

if __name__ == '__main__':
    main()