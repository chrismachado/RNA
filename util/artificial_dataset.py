import itertools
import random

import numpy as np


class Artificial(object):

    def and_problem(self, npc=10, ruido=0.015):
        X = []
        for i, j in zip([0, 0, 1, 1], [0, 1, 0, 1]):
            for _ in range(npc):
                X.append([random.uniform( i + ((-1) ** (_+1)) * ruido, i + ((-1) ** _) * ruido),
                          random.uniform(j + ((-1) ** (_+1)) * ruido, j + ((-1) ** _) * ruido)])

        X = np.array(X)
        y = np.concatenate((np.zeros((3*npc, 1), dtype=int), np.ones((npc, 1), dtype=int)))

        y = np.reshape(y, (4*npc))
        return X, y

    def or_problem(self, npc=10, ruido=0.015):
        X = []

        for i, j in zip([0, 0, 1, 1], [0, 1, 0, 1]):
            for _ in range(npc):
                X.append([random.uniform(i + ((-1) ** (_ + 1)) * ruido, i + ((-1) ** _) * ruido),
                          random.uniform(j + ((-1) ** (_ + 1)) * ruido, j + ((-1) ** _) * ruido)])

        X = np.array(X)
        y = np.concatenate((np.zeros((npc, 1), dtype=int), np.ones((3*npc, 1), dtype=int)))
        y = np.reshape(y, (4*npc))

        return X, y

    def function(self, a, b, ruido):
        x = np.arange(0, 1, 0.2)
        # X = np.reshape(x, (x.shape[0], 1))

        N = x.size
        X = np.ones(shape=(N * N, 2))
        X[:, 0] = X[:, 0] * np.random.uniform(0, 5, N * N)
        y = a*X[:, 0] + b + np.random.uniform(-ruido, ruido, N * N)

        return X, y, N

    def function3d(self, a, b, c, ruido):
        x = np.arange(0, 10, 1, dtype=float)
        N = x.size

        X = np.ones(shape=(N * N, 2))
        ruido = np.random.uniform(-ruido, ruido, N * N)

        i = 0
        for aa, bb in itertools.product(x, x):
            X[i][0] = aa
            X[i][1] = bb
            i += 1
        y = a * X[:, 0] + b * X[:, 1] + c + ruido

        return X, y, N