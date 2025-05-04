import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from scipy.integrate import solve_bvp


def solver_bvp(x, f):
    """
    Решает краевую задачу -y'' = f(x) с использованием solve_bvp и возвращает y(x) на исходной сетке x.

    Args:
        x: Массив точек, в которых нужно найти решение.
        f: Функция f(x).

    Returns:
        x: Исходный массив x.
        y: Значения y(x), интерполированные на исходную сетку x.
    """

    def fun(x, y):
        return np.vstack((y[1], -f(x)))

    def bc(ya, yb):
        return np.array([ya[0], yb[0]])  # y(a) = y(b) = 0

    y_init = np.zeros((2, x.size))
    sol = solve_bvp(fun, bc, x, y_init)

    # Интерполируем решение на исходную сетку x
    y = sol.sol(x)[0]  #  возвращает массив, где первая строка - y(x), вторая - y'(x)

    return x, y


def FDM(f, n, a, b):
    """
    Решает краевую задачу y'' = f(x), y(0) = y(1) = 0 методом конечных разностей
    с использованием scipy.sparse.linalg.spsolve. Но работает пока что только для
    гладких функций

    Args:
        f: Функция f(x) в уравнении y'' = f(x).
        n: Количество внутренних точек.

    Returns:
        Массивы x и y, представляющие решение.
    """

    h = 1 / (n + 1)
    x = np.linspace(a, b, n + 2)  # Включаем граничные точки
    x_internal = x[1:-1]          # Только внутренние точки

    # Создаем трехдиагональную матрицу A
    diagonals = [[1] * (n - 1), [-2] * n, [1] * (n - 1)]
    A = diags(diagonals, [-1, 0, 1]).tocsr()  # Создаем разреженную матрицу

    # Создаем вектор b
    b = h**2 * f(x_internal)

    # Решаем систему Ay = b
    y_internal = spsolve(A, b)

    # Добавляем граничные условия
    y = np.zeros(n + 2)
    y[1:-1] = y_internal

    return x, y
