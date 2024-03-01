import numpy as np
import scipy
from scipy.special import expit


class BaseSmoothOracle:
    """
    Базовый класс для реализации оракулов.
    """
    def func(self, w):
        """
        Вычислить значение функции в точке w.
        """
        raise NotImplementedError('Func oracle is not implemented.')

    def grad(self, w):
        """
        Вычислить значение градиента функции в точке w.
        """
        raise NotImplementedError('Grad oracle is not implemented.')


class BinaryLogistic(BaseSmoothOracle):
    """
    Оракул для задачи двухклассовой логистической регрессии.

    Оракул должен поддерживать l2 регуляризацию.
    """

    def __init__(self, l2_coef):
        """
        Задание параметров оракула.

        l2_coef - коэффициент l2 регуляризации
        """
        self.l2_coef = l2_coef

    def func(self, X, y, w):
        """
        Вычислить значение функционала в точке w на выборке X с ответами y.

        X - scipy.sparse.csr_matrix или двумерный numpy.array

        y - одномерный numpy array

        w - одномерный numpy array
        """
        return (np.logaddexp(
                    np.zeros((y.shape[0], 1)),
                    np.multiply(- y.reshape(-1, 1), (X @ w.reshape(-1, 1)))
                ).sum() / y.shape[0] +
                self.l2_coef / 2 * np.square(np.linalg.norm(w)))

    def grad(self, X, y, w):
        """
        Вычислить градиент функционала в точке w на выборке X с ответами y.

        X - scipy.sparse.csr_matrix или двумерный numpy.array

        y - одномерный numpy array

        w - одномерный numpy array
        """
        y = y.reshape(-1, 1)
        if isinstance(X, np.ndarray):
            res = - (
                y * expit(- y * (X @ w.reshape(-1, 1))) * X
            ).sum(axis=0) / y.shape[0] + self.l2_coef * w
        elif isinstance(X, scipy.sparse.csr_matrix):
            res = - (
                X.multiply(np.multiply(
                    y,
                    expit(np.multiply(-y, (X @ w.reshape(-1, 1)))))
                )
            ).sum(axis=0) / y.shape[0] + self.l2_coef * w
        else:
            raise NotImplementedError
        return np.array(res).reshape(-1)
