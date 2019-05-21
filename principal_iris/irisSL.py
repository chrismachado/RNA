
import numpy as np
import pandas as pd
import yaml

from perceptron_sigmoide.perceptron_sigmoide import PerceptronSG


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

    ppn = PerceptronSG(epocas=epocas, eta=eta, base_treino=base_treino, neuronios=3)

    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'

    df = pd.read_csv(url, header=None)
    y = df.iloc[:, 4].values

    # y = np.where(y == 'Iris-setosa', [1, 0, 0], y)
    # y = np.where(y == 'Iris-setosa', '1, 0, 0', y)
    # y = np.where(y == 'Iris-versicolor', [0, 1, 0], y)
    # y = np.where(y == 'Iris-virginica', [0, 0, 1], y)

    for i in range(y.shape[0]):
        if y[i] == 'Iris-setosa':
            y[i] = np.array([1, 0, 0])
        elif y[i] == 'Iris-versicolor':
            y[i] = np.array([0, 1, 0])
        elif y[i] == 'Iris-virginica':
            y[i] = np.array([0, 0, 1])

    y = np.reshape(y, (150, 1))

    # atributos de treinamento
    X = df.iloc[:, atributos].values

    # shuffle_
    ppn.shuffle_(X, y)

    ppn.fit(X, y)


if __name__ == '__main__':
    main()