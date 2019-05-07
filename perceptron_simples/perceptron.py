import numpy as np

class Perceptron(object):

    def __init__(self, eta=0.01, epochs=50):
        self.eta = eta
        self.epochs = epochs

    def train(self, X, y):
        self.w_ = np.zeros(1 + X.shape[1])
        self.w_[0] = -1
        self.errors_ = []

        for _ in range(self.epochs):
            errors = 0
            self.shuffle_(X, y) #Embaralha os dois vetores na mesma proporcao

            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                # errors += int(update != 0.0)
                errors += abs(update != 0.0)

            self.errors_.append(errors)
            if errors == 0:
                break

        return self

    def test(self, X, y):
        hitrate = 0
        y_test = []
        cm = np.zeros((2,2))
        for xi, target in zip(X, y):
            dy = self.predict(xi) - target
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
        return K[:int(len(K) * 0.7)]

    def get_test_sample(self, K):
        return K[int(len(K) * 0.7):]
