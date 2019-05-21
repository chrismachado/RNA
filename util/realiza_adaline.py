import numpy as np


def executar(X, y, clf, num=20):
    # Salva a configuração randômica das entradas
    rmse = []
    mse = []
    weights = []
    custo = []

    X.transpose()

    for _ in range(num):
        print("Executando realização %s..." % (_ + 1))

        clf.shuffle_(X, y)
        X_train, X_test = clf.get_train_sample(X), clf.get_test_sample(X)
        y_train, y_test = clf.get_train_sample(y), clf.get_test_sample(y)

        clf.shuffle_(X_train, y_train)
        clf = clf.fit(X_train, y_train)
        custo.append(clf.custo)

        clf.shuffle_(X_test, y_test)
        mse_, rmse_ = clf.test(X_test, y_test)

        mse.append(mse_)
        rmse.append(rmse_)

        print("\t→ MSE: ", mse_)
        print("\t→ RMSE: ", rmse_)

        weights.append(clf.w_)

    imax, imin = melhor_pior_iteracao(rmse)

    clf.custo = custo[imin]
    clf.w_ = weights[imin]

    print("\n#######################################################")
    print("### Desvio padrão do MSE: ", np.std(mse))
    print("### Desvio padrão do RMSE: ", np.std(rmse))

    print("\n#######################################################")
    print("## Iteração com menor erro quadrático médio: ", imin)
    print("## Erro quadrático médio: ", mse[imin])
    print("## Raiz quadrada erro quadrático médio: ", rmse[imin])

    return clf

def melhor_pior_iteracao(rmse):
    max_acc_value = rmse[0]
    min_acc_value = rmse[0]
    imax_ = 0
    imin_ = 0

    for index in range(1, len(rmse)):
        if max_acc_value <= rmse[index]:
            imax_ = index
            max_acc_value = rmse[index]
        if min_acc_value >= rmse[index]:
            imin_ = index
            min_acc_value = rmse[index]
    return imax_, imin_
