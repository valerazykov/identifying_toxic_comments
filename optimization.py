import numpy as np
from time import time
from oracles import BinaryLogistic
import scipy
from scipy.special import expit

EPS = 1e-9
DEFAULT_L2_COEF = 0.0001


def accuracy(x, y):
    return np.sum(x == y) / y.shape[0]


def get_default_w_0(X, y):
    if isinstance(X, np.ndarray):
        return (y.reshape(-1, 1) * X).sum(axis=0) / (
                    (X * X).sum(axis=0) + EPS
                ).reshape(-1)
    elif isinstance(X, scipy.sparse.csr_matrix):
        return (X.multiply(y.reshape(-1, 1))).sum(axis=0) / (
                    (X.multiply(X)).sum(axis=0) + EPS
                ).reshape(-1)


class GDClassifier:
    """
    Реализация метода градиентного спуска для произвольного
    оракула, соответствующего спецификации оракулов из модуля oracles.py
    """

    def __init__(
            self, loss_function='binary_logistic', step_alpha=1, step_beta=0,
            tolerance=1e-5, max_iter=1000, **kwargs
    ):
        """
        loss_function - строка, отвечающая за функцию потерь классификатора.
        Может принимать значения:
        - 'binary_logistic' - бинарная логистическая регрессия

        step_alpha - float, параметр выбора шага из текста задания

        step_beta- float, параметр выбора шага из текста задания

        tolerance - точность, по достижении которой, необходимо прекратить
        оптимизацию. Необходимо использовать критерий выхода по модулю разности
        соседних значений функции:
        если |f(x_{k+1}) - f(x_{k})| < tolerance: то выход

        max_iter - максимальное число итераций

        **kwargs - аргументы, необходимые для инициализации
        """
        if loss_function == 'binary_logistic':
            self.oracle = BinaryLogistic(
                kwargs.get("l2_coef", DEFAULT_L2_COEF)
            )
        else:
            raise NotImplementedError
        self.step_alpha = step_alpha
        self.step_beta = step_beta
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.w = None
        self.kwargs = kwargs

    def fit(self, X, y, w_0=None, trace=False):
        """
        Обучение метода по выборке X с ответами y

        X - scipy.sparse.csr_matrix или двумерный numpy.array

        y - одномерный numpy array

        w_0 - начальное приближение в методе

        trace - переменная типа bool

        Если trace = True, то метод должен вернуть словарь history, содержащий
        информацию о поведении метода.
        Длина словаря history = количество итераций + 1 (начальное приближение)

        history['time']: list of floats, содержит интервалы времени между
            двумя итерациями метода (0 для самой первой точки)
        history['func']: list of floats, содержит значения функции на
        каждой итерации
        history['accuracy']: list of floats, содержит значения точности на
            обучающей выборке для каждой итерации
        """
        if w_0 is None:
            self.w = get_default_w_0(X, y)
        else:
            self.w = w_0

        Q_prev = None
        Q = self.oracle.func(X, y, self.w)

        if trace:
            history = {
                "time": [0.0],
                "func": [Q],
                "accuracy": [accuracy(y, self.predict(X))]
            }
            time_start = time()

        for k in range(1, self.max_iter + 1):
            gw = self.oracle.grad(X, y, self.w)
            eta = self.step_alpha / k ** self.step_beta
            self.w -= eta * gw
            Q_prev = Q
            Q = self.oracle.func(X, y, self.w)

            if trace:
                history["time"].append(time() - time_start)
                history["func"].append(Q)
                history["accuracy"].append(
                    accuracy(y, self.predict(X)))

            if abs(Q - Q_prev) < self.tolerance:
                break

        if trace:
            return history

    def predict(self, X):
        """
        Получение меток ответов на выборке X

        X - scipy.sparse.csr_matrix или двумерный numpy.array

        return: одномерный numpy array с предсказаниями
        """
        # Значения меток y: -1, 1
        pred = np.sign((np.array(X @ self.w.reshape(-1, 1))).reshape(-1))
        zeros_count = (pred == 0).sum()
        # Замена 0 на 1, -1
        pred[pred == 0] = np.hstack([np.ones(zeros_count // 2),
                                     np.full(((zeros_count + 1) // 2), -1)])
        return pred

    def predict_proba(self, X):
        """
        Получение вероятностей принадлежности X к классу k

        X - scipy.sparse.csr_matrix или двумерный numpy.array

        return: двумерной numpy array, [i, k] значение соответствует
        вероятности принадлежности i-го объекта к классу k
        """
        p_plus = expit((X @ self.w.reshape(-1, 1)))
        p_minus = 1 - p_plus
        return np.hstack([p_minus, p_plus])

    def get_objective(self, X, y):
        """
        Получение значения целевой функции на выборке X с ответами y

        X - scipy.sparse.csr_matrix или двумерный numpy.array
        y - одномерный numpy array

        return: float
        """
        return self.oracle.func(X, y, self.w)

    def get_gradient(self, X, y):
        """
        Получение значения градиента функции на выборке X с ответами y

        X - scipy.sparse.csr_matrix или двумерный numpy.array
        y - одномерный numpy array

        return: numpy array, размерность зависит от задачи
        """
        self.oracle.grad(X, y, self.w)

    def get_weights(self):
        """
        Получение значения весов функционала
        """
        return self.w


class SGDClassifier(GDClassifier):
    """
    Реализация метода стохастического градиентного спуска для произвольного
    оракула, соответствующего спецификации оракулов из модуля oracles.py
    """

    def __init__(
            self, loss_function='binary_logistic', batch_size=1000,
            step_alpha=1, step_beta=0,
            tolerance=1e-5, max_iter=1000, random_seed=153, **kwargs
    ):
        """
        loss_function - строка, отвечающая за функцию потерь классификатора.
        Может принимать значения:
        - 'binary_logistic' - бинарная логистическая регрессия

        batch_size - размер подвыборки, по которой считается градиент

        step_alpha - float, параметр выбора шага из текста задания

        step_beta- float, параметр выбора шага из текста задания

        tolerance - точность, по достижении которой, необходимо прекратить
        оптимизацию. Необходимо использовать критерий выхода по модулю разности
        соседних значений функции:
        если |f(x_{k+1}) - f(x_{k})| < tolerance: то выход

        max_iter - максимальное число итераций (эпох)

        random_seed - в начале метода fit необходимо вызвать
        np.random.seed(random_seed). Этот параметр нужен для воспроизводимости
        результатов на разных машинах.

        **kwargs - аргументы, необходимые для инициализации
        """
        super().__init__(
            loss_function, step_alpha, step_beta, tolerance, max_iter, **kwargs
        )
        self.batch_size = batch_size
        self.random_seed = random_seed

    def fit(self, X, y, w_0=None, trace=False, log_freq=1):
        """
        Обучение метода по выборке X с ответами y

        X - scipy.sparse.csr_matrix или двумерный numpy.array

        y - одномерный numpy array

        w_0 - начальное приближение в методе

        Если trace = True, то метод должен вернуть словарь history, содержащий
        информацию о поведении метода. Если обновлять history после каждой
        итерации, метод перестанет превосходить в скорости метод GD. Поэтому,
        необходимо обновлять историю метода лишь после некоторого числа
        обработанных объектов в зависимости от приближённого номера эпохи.
        Приближённый номер эпохи:
            {количество объектов, обработанных методом SGD} /
            {количество объектов в выборке}

        log_freq - float от 0 до 1, параметр, отвечающий за частоту обновления.
        Обновление должно происходить каждый раз, когда разница между двумя
        значениями приближённого номера эпохи будет превосходить log_freq.

        history['epoch_num']: list of floats, в каждом элементе списка будет
            записан приближённый номер эпохи (0 для самой первой точки)
        history['time']: list of floats, содержит интервалы времени между
            двумя соседними замерами (0 для самой первой точки)
        history['func']: list of floats, содержит значения функции после
            текущего приближённого номера эпохи
        history['weights_diff']: list of floats, содержит квадрат нормы
        разности векторов весов с соседних замеров (0 для самой первой точки)
        history['accuracy']: list of floats, содержит значения точности на
            обучающей выборке для текущего приближённого номера эпохи
        """
        np.random.seed(self.random_seed)
        if w_0 is None:
            self.w = get_default_w_0(X, y)
        else:
            self.w = w_0

        Q_prev = None
        Q = self.oracle.func(X, y, self.w)

        # for log
        if trace:
            history = {
                "epoch_num": [0],
                "time": [0.0],
                "func": [Q],
                "weights_diff": [0.0],
                "accuracy": [accuracy(y, self.predict(X))]
            }
            appr_epoch_num = 0
            time_start = time()
            w_prev = self.w.copy()

        for k in range(1, self.max_iter + 1):
            obj_idx = np.arange(X.shape[0])
            np.random.shuffle(obj_idx)
            X_sh = X[obj_idx, :]
            y_sh = y[obj_idx]

            for i in range(0, X.shape[0], self.batch_size):
                X_batch = X_sh[i: i + self.batch_size, :]
                y_batch = y_sh[i: i + self.batch_size]

                gw = self.oracle.grad(X_batch, y_batch, self.w)
                eta = self.step_alpha / k ** self.step_beta
                self.w -= eta * gw

                if trace:
                    appr_epoch_num += self.batch_size / X.shape[0]
                    if appr_epoch_num - history["epoch_num"][-1] >= log_freq:
                        history["epoch_num"].append(appr_epoch_num)
                        history["time"].append(time() - time_start)
                        history["func"].append(self.oracle.func(X, y, self.w))
                        history["weights_diff"] = (
                                np.linalg.norm(self.w - w_prev) ** 2
                        )
                        w_prev = self.w.copy()
                        history["accuracy"].append(
                            accuracy(y, self.predict(X)))
            Q_prev = Q
            Q = self.oracle.func(X, y, self.w)

            if abs(Q - Q_prev) < self.tolerance:
                break

        if trace:
            return history
