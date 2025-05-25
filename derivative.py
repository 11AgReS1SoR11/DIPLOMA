import tensorflow as tf
import numpy as np
from scipy.interpolate import CubicSpline


def calculate_auto_derivative_2(x, y):

    if not callable(y):
        raise ValueError("y must be callable")

    x_tf = []
    if isinstance(x, tf.Tensor):
        x_tf = x
    else:
        x_tf = tf.convert_to_tensor(x.reshape(-1, 1), dtype=tf.float32)

    with tf.GradientTape() as tape_exact2:
        tape_exact2.watch(x_tf)
        with tf.GradientTape() as tape_exact1:
            tape_exact1.watch(x_tf)
            y_exact = y(x_tf)
        dy_exact_dx = tape_exact1.gradient(y_exact, x_tf) # y'
    d2y_exact_dx2 = tape_exact2.gradient(dy_exact_dx, x_tf) # y''

    return d2y_exact_dx2


def calculate_auto_derivative(x, y):

    if not callable(y):
        raise ValueError("y must be callable")

    x_tf = []
    if isinstance(x, tf.Tensor):
        x_tf = x
    else:
        x_tf = tf.convert_to_tensor(x.reshape(-1, 1), dtype=tf.float32)

    with tf.GradientTape() as tape_exact1:
        tape_exact1.watch(x_tf)
        y_exact = y(x_tf)
    dy_exact_dx = tape_exact1.gradient(y_exact, x_tf) # y'
    
    if dy_exact_dx is None:
        return 0
 
    return dy_exact_dx


def get_cubic_interpolation(x, y, derivative = None):

    if len(x) != len(y):
        raise ValueError("Длина массивов x и y должна совпадать")

    if not np.all(np.diff(x) > 0):
        raise ValueError("Массив x должен быть строго возрастающим.")

    f_spline = CubicSpline(x, y)

    if derivative:
        f_spline = f_spline(x, nu=derivative)

    return f_spline
