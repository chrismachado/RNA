
import matplotlib.pyplot as plt
import numpy as np
import yaml

from perceptron_adaline.adaline import Adaline
from util.realiza_adaline import executar
from util.utilidade import Utilidade


def main():
    stream = open('./configuracoes/config_execucao.yml', 'r', encoding='utf-8').read()
    config_exec = yaml.load(stream=stream)

    a, b, c = config_exec['abc']
    realizacoes = config_exec['realizacoes']
    eta = config_exec['eta']
    epocas = config_exec['epocas2D']
    ruido = config_exec['ruido']

    ada = Adaline(eta=eta, epochs=epocas)

    x = np.arange(0, 10, 0.5, dtype=float)
    X = np.reshape(x, (x.shape[0], 1))
    Utilidade().normalize_(X)

    ruido = np.reshape(np.random.uniform(-ruido, ruido, x.size), (X.shape[0], 1))

    # y = a * x + b
    y = a * X + b + ruido
    y = np.reshape(y, (y.shape[0], 1))

    ada = executar(X=X, y=y, clf=ada, num=realizacoes)
    z = ada.w_[1] * x + ada.w_[0]

    plt.plot(range(len(ada.custo)), ada.custo, ls='-')
    plt.title('Curva de aprendizado')
    plt.xlabel('Época')
    plt.ylabel('Custo')
    plt.savefig("../figuras/%s.png" % 'ca2d', format='png')
    plt.show()

    plt.plot(X, y, 'ro', label='f(x) = a * x + b ± ruido', mfc='none')
    plt.title('Reta aprendida pelo Adaline 2D')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig("../figuras/%s.png" % 'datara2d', format='png')
    plt.plot(X, z, label='f(x) = w[1] * x + w[0]', )
    plt.legend()
    plt.savefig("../figuras/%s.png" % 'ra2d', format='png')
    plt.show()

if __name__ == '__main__':
    main()