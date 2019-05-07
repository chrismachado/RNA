import matplotlib.pyplot as plt
import numpy as np
from mlxtend.plotting import plot_decision_regions


class Utilidade(object):
    def __init__(self, verb='n', log='n'):
        self.verb = verb
        self.log = log

    def execution(self, X, y, clf, num=20):
        # Salva a configuração randômica das entradas
        X_new = []
        y_new = []
        accuracy = []
        weights = []
        dmm = []
        erros = []

        X.transpose()
        self.normalize_(X)

        for _ in range(num):
            if self.verb == 's':
                print("Executando realização %s..." % (_+1), end='')

            X_train, X_test = clf.get_train_sample(X), clf.get_test_sample(X)
            y_train, y_test = clf.get_train_sample(y), clf.get_test_sample(y)

            X_new.append([X_train, X_test])
            y_new.append([y_train, y_test])

            clf.train(X_train, y_train)
            erros.append(clf.errors_)
            weights.append(clf.w_)

            hit, y_predict, cm = clf.test(X_test, y_test)

            if self.verb == 's':
                print("\tTaxa de acerto: ", hit)

            accuracy.append(hit)
            # dmm.append(confusion_matrix(y_test, y_predict))
            dmm.append(cm)

        imax_, imin_ = self.evaluate_exec(accuracy=accuracy)

        self.inf_log(accuracy, dmm, weights, imax_, imin_)

        return X_new, y_new, accuracy, weights, dmm, erros, imax_, imin_

    def plot_decision(self, X, y, clf, title='Title', xlabel='xlabel', ylabel='ylabel'):
        plot_decision_regions(X, y, clf=clf)

        plt.axis([0, 1, 0, 1])

        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        if self.log == 's':
            plt.savefig("../figuras/%s.png" % (title), format='png')

        plt.show()

    def plot_learning_bend(self, erros, title='Title', xlabel='xlabel', ylabel='ylabel'):

        plt.plot(range(1, len(erros) + 1), erros, marker='o')
        plt.axis([1, len(erros), min(erros), max(erros)])
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        if self.log == 's':
            plt.savefig("../figuras/%s.png" % title, format='png')

        plt.show()

    def normalize_(self, X):
        for i in range(X.shape[1]):
            max_ = max(X[:, i])
            min_ = min(X[:, i])

            for j in range(X.shape[0]):
                X[j, i] = (X[j, i] - min_) / (max_ - min_)

    def evaluate_exec(self, accuracy):
        max_acc_value = accuracy[0]
        min_acc_value = accuracy[0]
        imax_ = 0
        imin_ = 0

        for index in range(1, len(accuracy)):
            if max_acc_value <= accuracy[index]:
                imax_ = index
                max_acc_value = accuracy[index]
            if min_acc_value >= accuracy[index]:
                imin_ = index
                min_acc_value = accuracy[index]
        return imax_, imin_

    def inf_log(self, accuracy, dmm, weights, imax_, imin_):
        print("+=================================================+")
        print("+========       RESULTADO GERAL      =============+")
        print("+==  Acurácia : ", np.mean(accuracy))
        print("+==  Desvio Padrão : %s" % np.std(accuracy))

        if self.verb == 's':
            print("\n+========    PIOR RESULTADO OBTIDO   =============+")
            print("+==  Iteração com pior resultado: ", imin_ + 1)
            print("+==  Taxa de acerto: ", accuracy[imin_])
            print("+==  Pesos desta iteração: ", weights[imin_])
            print("+==  Matriz de confusao :  \n\t\t\t%s\n\t\t\t%s" % (dmm[imin_][0], dmm[imin_][1]))

        print("\n+========    MELHOR RESULTADO OBTIDO   =============+")
        print("+==  Iteração com melhor resultado: ", imax_ + 1)
        print("+==  Taxa de acerto: ", accuracy[imax_])
        print("+==  Pesos desta iteração: ", weights[imax_])
        print("+==  Matriz de confusao :  \n\t\t\t%s\n\t\t\t%s" % (dmm[imax_][0], dmm[imax_][1]))
