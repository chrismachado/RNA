import numpy as np


class Adaline(object):

    def __init__(self, eta, epochs, base_treino=0.8):
        self.eta = eta
        self.epochs = epochs
        self.base_treino = base_treino

    def fit(self, X, y):

        self.w_ = np.ones(shape=(1 + X.shape[1], 1))
        # self.w_ = np.zeros(shape=(1 + X.shape[1], 1))
        self.custo = []

        for _ in range(self.epochs):

            self.shuffle_(X, y)
            saida = self.net_input(X)
            erros = y - saida

            self.w_[1:] += self.eta * X.T.dot(erros)
            self.w_[0] += self.eta * sum(erros)

            custo_ = sum(erros ** 2) / 2.0

            self.custo.append(custo_)
        return self

    def test(self, X, y):
        if X.shape[1] == 1:
            ynew = self.w_[1] * X + self.w_[0]
        else:
            ynew = self.w_[1] * X[:, 0] + self.w_[2] * X[:, 1] + self.w_[0]

        ynew = np.reshape(ynew, (ynew.shape[0], 1))

        mse = y - ynew
        mse = sum(mse ** 2.0) / mse.shape[0]
        rmse = np.sqrt(mse)

        return mse, rmse

    def net_input(self, X):
        #xTw
        return np.dot(X, self.w_[1:]) + self.w_[0]

    #nao usa
    # def predict(self, X):
    #     return np.where(self.net_input(X) >= 0.0, 1, 0)

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
