import numpy as np


class PerceptronSIG(object):

    def __init__(self, eta=0.01, epochs=50, base_treino=0.8, type_y='none'):
        self.eta = eta
        self.epochs = epochs
        self.base_treino = base_treino
        self.type_y = type_y

    def yl(self, u):
        if self.type_y == 'none':
            return 1
        elif self.type_y == 'logistic':
            y = 1.0 / (1.0 + np.exp(-u))
            return y * (1.0 + y)
        elif self.type_y == 'tanh':
            y = np.tanh(-u)
            return 1.0 - y ** 2.0

    def fit(self, X, y):
        self.w_ = np.zeros((1 + X.shape[1]))
        self.errors_ = []
        # XX = np.c_[-np.ones(shape=(X.shape[0], 1)), X]
        # yy = cp.deepcopy(y)
        for _ in range(self.epochs):
            errors = 0
            self.shuffle_(X, y) #Embaralha os dois vetores na mesma proporcao

            for xi, target in zip(X, y):
                dy = self.yl(self.net_input(xi))

                e = (target - self.predict(xi))
                # self.w_ += self.eta * e * dy * xi
                self.w_[1:] += self.eta * dy * e * xi
                self.w_[0] += self.eta * e * dy
                errors += abs(e != 0.0)

            self.errors_.append(errors)
            if errors == 0:
                break

        return self

    def test(self, X, y):
        hitrate = 0
        y_test = []
        cm = np.zeros((2, 2))
        # XX = np.c_[-np.ones(shape=(X.shape[0], 1)), X]
        # yy = cp.deepcopy(y)
        for xi, target in zip(X, y):
            dy = target - self.predict(xi)
            y_test.append(dy)

           #Matriz de confusao
            if dy == 0:
                hitrate += 1
                if self.predict(xi) == 0:
                    cm[0][0] += 1
                else:
                    cm[1][1] += 1
            else:
                if self.predict(xi) == 0:
                    cm[0][1] += 1
                else:
                    cm[1][0] += 1

        hitrate = hitrate / len(y)
        return hitrate, y, cm

    def net_input(self, X):
        #xTw
        # return np.dot(X, self.w_)
        return np.dot(X, self.w_[1:]) + self.w_[0]

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
