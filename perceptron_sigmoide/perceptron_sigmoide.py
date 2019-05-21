import numpy as np


class PerceptronSG(object):

    def __init__(self, eta, epocas, base_treino, neuronios):
        self.eta = eta
        self.epocas = epocas
        self.base_treino = base_treino
        self.neuronios = neuronios

    def fit(self, X, y):

        self.w_ = np.ones(shape=(X.shape[1] + 1, self.neuronios))
        for _ in range(self.epocas):
            self.shuffle_(X, y)
            u = self.net_input(X)

            e = y - self.predict(u[:, :-1])


    def net_input(self, X):
        #xTw
        return np.dot(X, self.w_[1:, :]) - self.w_[0, :]

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, 0)

    def shuffle_(self, X, y):
        state = np.random.get_state()
        np.random.shuffle(X)
        np.random.set_state(state)
        np.random.shuffle(y)

    # dar somente 1 shuffle_ antes de usar estas funcoes
    def get_train_sample(self, K):
        return K[:int(len(K) * self.base_treino)]

    def get_test_sample(self, K):
        return K[int(len(K) * self.base_treino):]

    def to_predict(self, e):
        # for j in range(e.shape[1]):
        for i in range(e.shape[0]):
            p = max(e[i, :])
