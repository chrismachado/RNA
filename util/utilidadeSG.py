import warnings

import matplotlib.pyplot as plt
import numpy as np


class UtilidadeSG(object):
    def __init__(self, verb='n', log='n', ptype='unknwon-base'):
        self.verb = verb
        self.log = log
        self.ptype = ptype

    def execution(self, X, y, clf, num=20):
        X_new = []
        y_new = []
        accuracy = []
        weights = []
        dmm = []
        erros = []

        X.transpose()

        self.normalize_(X)

        for _ in range(num):
            clf.shuffle_(X, y)
            print("Executando realização %s..." % (_+1), end='')

            X_train, X_test = clf.get_train_sample(X), clf.get_test_sample(X)
            y_train, y_test = clf.get_train_sample(y), clf.get_test_sample(y)

            X_new.append([X_train, X_test])
            y_new.append([y_train, y_test])

            clf = clf.fit(X_train, y_train)
            erros.append(clf.errors)
            weights.append(clf.w_)

            # print("\n", clf.w_)
            hit = clf.test(X_test, y_test)

            print("\tTaxa de acerto: ", hit)

            accuracy.append(hit)

        print("+=================================================+")
        print("+========       RESULTADO GERAL      =============+")
        print("+==  Acurácia : ", np.mean(accuracy))
        print("+==  Desvio Padrão : %s" % np.std(accuracy))

        imin, imax = self.evaluate_exec(accuracy=accuracy)
        # print(X.shape)
        if X.shape[1] == 2:
            self.plot_decision(X=X, clf=clf, X_highlights=X_new[imax][1])
        else:
            warnings.warn("X possui mais de 2 dimensões => %s" % X.shape)

        with open("../log/slp-%s" % self.ptype, 'w') as f:
            f.write("+=================================================+\n")
            f.write("+========       RESULTADO GERAL      =============+\n")
            f.write("+==  Acurácia : %s\n" % np.mean(accuracy))
            f.write("+==  Desvio Padrão : %s\n" % np.std(accuracy))

        return

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

    def plot_decision(self, X, clf, X_highlights=None):
        xx1_max, xx1_min = X[:, 0].max() + 0.2, X[:, 0].min() - 0.2
        xx2_max, xx2_min = X[:, 1].max() + 0.2, X[:, 1].min() - 0.2

        xx1, xx2 = np.meshgrid(np.arange(xx1_min, xx1_max, 0.035), np.arange(xx2_min, xx2_max, 0.035))
        Z = np.array([xx1.ravel(), xx2.ravel()]).T

        aux = 0 if clf.type_y != 'tanh' else -1
        s = 25

        marker = 's'

        print("\nCreating colormap...")
        for x1, x2 in Z:
            predict = clf.around(clf.predict([-1, x1, x2]))
            if np.array_equal(predict, np.array([1, aux, aux])):
                plt.scatter(x1, x2, c='#800D9F', s=s, marker=marker)
            elif np.array_equal(predict, np.array([aux, 1, aux])):
                plt.scatter(x1, x2, c='#0083FF', s=s, marker=marker)
            elif np.array_equal(predict, np.array([aux, aux, 1])):
                plt.scatter(x1, x2, c='#AFE31A', s=s, marker=marker)

        print()

        for xx1, xx2 in X_highlights:
            plt.plot(xx1, xx2, 'ko', fillstyle='none', markersize=8)

        #ColorMap Perceptron
        plt.savefig("../figuras/%s-%s.png" % ('cmp', self.ptype), format='png')
        plt.show()


