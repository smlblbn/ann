import numpy as np


def read_data(filename='./set1.dat'):
    dataset = []
    with open(filename, 'r') as f:
        for line in f:
            line = (line.strip()).split('\t')
            line = [float(t) for t in line]
            dataset.append(line)

    dataset = np.array(dataset, dtype=np.float32)
    np.random.shuffle(dataset)
    X = dataset[:, :-1]
    X = scale_data(np.array(X))
    Y = dataset[:, -1]

    N = X.shape[0]
    tr = int(N * 0.8)

    X_train = X[:tr, :]
    X_test = X[tr:, :]
    Y_train = Y[:tr]
    Y_test = Y[tr:]

    return X_train, Y_train, X_test, Y_test


def scale_data(X):
    ############################################################################
    #                            START OF YOUR CODE                            #
    ############################################################################
    min_values = np.amin(X, axis=0)
    max_values = np.amax(X, axis=0)
    return (X - min_values) / (max_values - min_values)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
