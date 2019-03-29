import numpy as np


def cholesky_fact(a):
    dim = a.shape[0]

    d = np.eye(dim)
    l = np.zeros((dim, dim))
    for j in range(dim):
        d[j][j] = a[j][j] - np.array([d[s][s] * l[j][s] ** 2 for s in range(j - 1)]).sum()
        for i in range(j + 1, dim):
            l[i][j] = (a[i][j] - np.array([d[s][s] * l[i][s] * l[j][s] for s in range(j - 1)]).sum()) / d[j][j]
    return l, d


if __name__ == "__main__":
    A = np.array([[1, 2, 1, 1, 1],
                  [2, 5, 1, 3, 1],
                  [1, 1, 6, 0, 6],
                  [1, 3, 0, 3, 0],
                  [1, 1, 6, 0, 7]])
    L, D = cholesky_fact(A)

    print(L, D)
