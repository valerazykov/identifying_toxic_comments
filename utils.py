import numpy as np


def grad_finite_diff(function, w, eps=1e-8):
    """
    Возвращает численное значение градиента, подсчитанное по следующей формуле:
        result_i := (f(w + eps * e_i) - f(w)) / eps,
        где e_i - следующий вектор:
        e_i = (0, 0, ..., 0, 1, 0, ..., 0)
                          >> i <<
    """
    I_matr = np.eye(w.shape[0])
    return np.array([(function(w + eps * I_matr[i]) - function(w)) / eps
                    for i in range(w.shape[0])])
