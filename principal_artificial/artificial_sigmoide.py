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

    realizacoes = config_exec['realizacoes']
    eta = config_exec['eta']
    epocas = config_exec['epocas']
    base_treino = config_exec['base_treino']
    npc = config_exec['num_por_classe']
    ruido = config_exec['ruido']
    type_d = config_exec['type_y']

    ppn = PerceptronNN(epocas=epocas, eta=eta, base_treino=base_treino, n=3, type_y=type_d)
    util = UtilidadeSG(ptype='Artificial')
    artificial = Artificial()
    X, y = artificial.artificial_sigmoide(npc=npc, ruido=ruido, type_d=type_d)

    util.normalize_(X)
    xx = copy.deepcopy(X)
    yy = copy.deepcopy(y)

    ppn.shuffle_(X, y)

    #Plot
    plt.plot(xx[:npc, 0], xx[:npc, 1], 'bo', mec='k', markersize=5, label='Padrão 1')
    plt.plot(xx[npc:2*npc, 0], xx[npc:2*npc, 1], 'r*', mec='k', label='Padrão 2')
    plt.plot(xx[2*npc:, 0], xx[2*npc:, 1], 'g^', mec='k', label='Padrão 3')
    plt.legend()

    util.execution(X, y, clf=ppn, num=realizacoes)

if __name__ == '__main__':
    main()