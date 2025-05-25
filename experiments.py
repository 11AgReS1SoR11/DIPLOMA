import numpy as np
from varnn import VariationalNeuralNetwork, l2_norm_square
import time
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from derivative import calculate_auto_derivative, get_cubic_interpolation
from Approx import solver_bvp


def create_table(data, filename):
    """
    Создает таблицу из данных и сохраняет ее как изображение PNG.

    Args:
        data (dict): Словарь с данными для таблицы.
        filename (str): Имя файла для сохранения таблицы (например, "table.png").
    """

    # Округляем числовые значения в данных
    for col in data:
        if isinstance(data[col], list):
            data[col] = [round(x, 7) if isinstance(x, float) else x for x in data[col]]

    df = pd.DataFrame(data)

    plt.figure(figsize=(15, len(df) * 0.5))  # Adjust figure size based on number of rows
    ax = plt.subplot(111, frame_on=False)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)

    table = ax.table(cellText=df.values, colLabels=df.columns, loc='center', cellLoc='center', fontsize=10)
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)  # Adjust scale for better appearance

    plt.tight_layout()  # Adjust layout to prevent labels from overlapping
    plt.savefig(filename, bbox_inches='tight', pad_inches=0.05, transparent=True)
    plt.close()  # Close the figure to release memory


# @deprecated("Вычисляет квадрат нормы и использует точное решение, поэтому используй run_experiment, который использует только нейросетевое решение")
def run_experiment_2(f, U, num_points, spatial_range, num_hidden, num_layers, num_iters, learning_rate, batch_size = 128, optimizer = 'adam'):
    """
    Проводит эксперимент с заданными параметрами и возвращает результаты.

    Args:
        num_hidden (int): Количество нейронов в слое.
        num_layers (int): Количество слоев в сети.
        num_iters (int): Количество итераций обучения.
        learning_rate (float): Скорость обучения.

    Returns:
        dict: Словарь с результатами эксперимента.
    """

    # Параметры для нашей задачи
    x_test = np.linspace(spatial_range[0], spatial_range[1], num_points)  # Тестовые данные
    u_exact = U(x_test) if callable(U) else U # Точное решение
    du_dx_exact = get_cubic_interpolation(x_test, u_exact, derivative=1) # the same np.pi * np.cos(np.pi * x_test)

    # Создаем и обучаем модель
    varnn = VariationalNeuralNetwork(right_hand_side_function=f, spatial_range=spatial_range, num_hidden=num_hidden,
                                      batch_size=batch_size, num_iters=num_iters,
                                      lr_rate=learning_rate, num_layers=num_layers, optimizer=optimizer)
    start_time = time.time()
    varnn.train()
    training_time = time.time() - start_time

    # Получаем предсказания
    u_predicted = varnn.predict(x_test)

    # Вычисляем апостериорную оценку ошибки и реальную ошибку
    aposterrori_error_estimate = varnn.compute_aposterrori_error_estimate(x_test, U)

    du_dx_predicted = varnn.predict_derivative(x_test)
    real_error_norm_dx = l2_norm_square(du_dx_exact.reshape(-1, 1) - du_dx_predicted, spatial_range[0], spatial_range[1])

    # Вычисляем отношение
    ratio_dx = (aposterrori_error_estimate / real_error_norm_dx) if real_error_norm_dx != 0 else np.nan

    results = {
        "batch size": batch_size,
        "Optimizer": optimizer,
        "Neurons": num_hidden,
        "Layers": num_layers,
        "Iterations": num_iters,
        "Learning Rate": learning_rate,
        "||M||^2": aposterrori_error_estimate,
        "||U'-V'||^2": real_error_norm_dx,
        "||M||^2/||U'-V'||^2": ratio_dx,
        "Training Time": training_time
    }

    return results


def get_avarage(results_list):
    """
    Фунция для усреднее результата

    Args:
        results_list: список результатов
    Returns:
        dict: Словарь с усредненными результатами эксперимента.
    """
    averaged_results = {}
    for key in results_list[0].keys():
        if key in ["batch size", "Optimizer", "Neurons", "Layers", "Iterations", "Learning Rate"]: # const params
            averaged_results[key] = results_list[0][key]
        else:
            averaged_results[key] = np.mean([result[key] for result in results_list])

    return averaged_results


def run_experiment(f, U, num_points, spatial_range, num_hidden, num_layers, num_iters, learning_rate, batch_size=128, optimizer='adam', p=None, q=None, num_runs=15):
    """
    Проводит эксперимент с заданными параметрами, усредняя результаты по нескольким запускам, и возвращает усредненные результаты.

    Args:
        f: Правая часть дифференциального уравнения.
        U: Функция точного решения.
        num_points (int): Количество точек для оценки решения.
        spatial_range (tuple): Диапазон пространственных координат (a, b).
        num_hidden (int): Количество нейронов в слое.
        num_layers (int): Количество слоев в сети.
        num_iters (int): Количество итераций обучения.
        learning_rate (float): Скорость обучения.
        batch_size (int): Размер батча (по умолчанию 256).
        optimizer (str): Оптимизатор ('adam' или другой) (по умолчанию 'adam').
        p: Функция p(x) в дифференциальном уравнении (по умолчанию None).
        q: Функция q(x) в дифференциальном уравнении (по умолчанию None).
        num_runs (int): Количество запусков эксперимента для усреднения (по умолчанию 15).

    Returns:
        dict: Словарь с усредненными результатами эксперимента.
    """

    results_list = []  # Список для хранения результатов каждого запуска

    x_test = np.linspace(spatial_range[0], spatial_range[1], num_points)  # Тестовые данные
    u_exact = U(x_test)  # Точное решение
    du_dx_exact = get_cubic_interpolation(x_test, u_exact, derivative=1)

    for _ in range(num_runs):
        # Создаем и обучаем модель
        varnn = VariationalNeuralNetwork(right_hand_side_function=f, spatial_range=spatial_range,
                                          num_hidden=num_hidden, batch_size=batch_size, num_iters=num_iters,
                                          lr_rate=learning_rate, num_layers=num_layers, optimizer=optimizer,
                                          p_function=p, q_function=q)
        start_time = time.time()
        varnn.train()
        training_time = time.time() - start_time

        # Получаем предсказания
        u_predicted = varnn.predict(x_test)

        # Вычисляем апостериорную оценку ошибки и реальную ошибку
        aposterrori_error_estimate = varnn.compute_aposterrori_error_estimate(x_test)

        du_dx_predicted = varnn.predict_derivative(x_test)
        real_error_norm_dx = tf.cast(tf.sqrt(l2_norm_square(du_dx_exact.reshape(-1, 1) - du_dx_predicted,
                                                             spatial_range[0], spatial_range[1])), dtype=tf.float32).numpy()
        derive_norm = tf.cast(tf.sqrt(l2_norm_square(du_dx_predicted, spatial_range[0], spatial_range[1])), dtype=tf.float32).numpy()

        # Вычисляем отношение
        ratio_dx = (aposterrori_error_estimate / real_error_norm_dx) if real_error_norm_dx != 0 else np.nan
        ratio_derive = aposterrori_error_estimate / derive_norm * 100 if derive_norm != 0 else np.nan

        results = {
            "batch size": batch_size,
            "Optimizer": optimizer,
            "Neurons": num_hidden,
            "Layers": num_layers,
            "Iterations": num_iters,
            "Learning Rate": learning_rate,
            "||M||": aposterrori_error_estimate,
            "||U'-V'||": real_error_norm_dx,
            "||M||/||U'-V'||": ratio_dx,
            "||M||/||V'||,%": ratio_derive,
            "Training Time": training_time
        }
        results_list.append(results)

    return get_avarage(results_list)


def experiments(f, U, num_points, spatial_range, postfix, p = None, q = None):
    # 1) Эксперименты с количеством нейронов
    neuron_counts = [10, 16, 32, 64]
    neuron_data = []
    for num_hidden in neuron_counts:
        results = run_experiment(f=f, U=U, num_points=num_points, spatial_range=spatial_range, num_hidden=num_hidden, num_layers=3, num_iters=3000, learning_rate=1e-3, p=p, q=q)
        neuron_data.append(results)

    neuron_df = pd.DataFrame(neuron_data)
    create_table(neuron_df.to_dict('list'), f"results_neurons_{postfix}.png")

    # 2) Эксперименты с количеством слоев
    layer_counts = [2, 3, 4, 5]
    layer_data = []
    for num_layers in layer_counts:
        results = run_experiment(f=f, U=U, num_points=num_points, spatial_range=spatial_range, num_hidden=32, num_layers=num_layers, num_iters=3000, learning_rate=1e-3, p=p, q=q)
        layer_data.append(results)

    layer_df = pd.DataFrame(layer_data)
    create_table(layer_df.to_dict('list'), f"results_layers_{postfix}.png")

    # 3) Эксперименты с количеством итераций
    iterations_counts = [1000, 2000, 3000, 4000, 5000]
    iters_data = []
    for num_iters in iterations_counts:
        results = run_experiment(f=f, U=U, num_points=num_points, spatial_range=spatial_range, num_hidden=32, num_layers=3, num_iters=num_iters, learning_rate=1e-3, p=p, q=q)
        iters_data.append(results)

    iters_df = pd.DataFrame(iters_data)
    create_table(iters_df.to_dict('list'), f"results_iters_{postfix}.png")

    # 4) Эксперименты со скоростью обучения
    learning_rates = [1e-1, 1e-2, 1e-3, 1e-4]
    lr_data = []
    for learning_rate in learning_rates:
        results = run_experiment(f=f, U=U, num_points=num_points, spatial_range=spatial_range, num_hidden=32, num_layers=3, num_iters=3000, learning_rate=learning_rate, p=p, q=q)
        lr_data.append(results)

    lr_df = pd.DataFrame(lr_data)
    create_table(lr_df.to_dict('list'), f"results_lr_{postfix}.png")

    # 5) Эксперименты с разными оптимизаторами
    optimizers = ['adam', 'sgd', 'rmsprop']
    optimizer_data = []
    for optimizer in optimizers:
        results = run_experiment(f=f, U=U, num_points=num_points, spatial_range=spatial_range, num_hidden=32, num_layers=3, num_iters=3000, learning_rate=1e-3, optimizer=optimizer, p=p, q=q)
        optimizer_data.append(results)

    optimizer_df = pd.DataFrame(optimizer_data)
    create_table(optimizer_df.to_dict('list'), f"results_optimizers_{postfix}.png")

    # 6) Эксперименты с разными оптимизаторами
    batch_sizes = [64, 128, 256, 512, 1024, 2048]
    batch_size_data = []
    for batch_size in batch_sizes:
        results = run_experiment(f=f, U=U, num_points=num_points, spatial_range=spatial_range, num_hidden=32, num_layers=3, num_iters=3000, learning_rate=1e-3, batch_size=batch_size, p=p, q=q)
        batch_size_data.append(results)

    optimizer_df = pd.DataFrame(batch_size_data)
    create_table(optimizer_df.to_dict('list'), f"results_batchSize_{postfix}.png")


if __name__ == "__main__":

    num_points = 300

    ########################### TEST №1 ###########################
    spatial_range = [0, 1]

    def f(x):
        return (np.pi**2) * tf.sin(np.pi * x)

    def U(x):
        return tf.sin(np.pi * x)

    postfix = "ppsin(px)" # set for different filenames
    experiments(f, U, num_points, spatial_range, postfix)

    # postfix = "ppsin(px)_approx"
    # x = np.linspace(spatial_range[0], spatial_range[1], num_points)
    # x, u_data = solver_bvp(x, f)
    # experiments(f, u_data, num_points, spatial_range, postfix)
    ########################### TEST №1 ###########################


    ########################### TEST №2 ###########################
    spatial_range2 = [-1, 1]

    def f2(x):
        return tf.where(x < 0, -1.0, 1.0)

    def U2(x):
        return tf.where(x < 0, 0.5 * x * (x + 1), -0.5 * x * (x - 1))

    postfix = "-1+1"
    experiments(f2, U2, num_points, spatial_range2, postfix)

    # postfix = "-1+1_approx"
    # x = np.linspace(spatial_range2[0], spatial_range2[1], num_points)
    # x, u_data2 = solver_bvp(x, f2)

    # experiments(f2, u_data2, num_points, spatial_range2, postfix)
    ########################### TEST №2 ###########################


    ########################### TEST №3 ###########################
    spatial_range3 = [0, 1]

    def f3(x):
        return 100 * (np.pi**2) * tf.sin(10 * np.pi * x)

    def U3(x):
        return tf.sin(10 * np.pi * x)

    postfix = "oscillating"
    experiments(f3, U3, num_points, spatial_range3, postfix)

    # postfix = "oscillating_approx"
    # x = np.linspace(spatial_range3[0], spatial_range3[1], num_points)
    # x, u_data3 = solver_bvp(x, f3)

    # experiments(f3, u_data3, num_points, spatial_range3, postfix)
    ########################### TEST №3 ###########################

    ########################### TEST №4: Gaussian Peak ###########################
    spatial_range4 = [-5, 5]
    peak_center = 0
    peak_width = 0.5
    peak_height = 10

    def f4(x):
        return peak_height * tf.exp(-((x - peak_center)**2) / (2 * peak_width**2))

    def U4(x):
        return -6.26657*x*tf.math.erf(np.sqrt(2)*x) - 2.5*tf.exp(-2*x*x) + 31.3329


    postfix = "gaussian_peak"
    experiments(f4, U4, num_points, spatial_range4, postfix)

    # postfix = "gaussian_peak_approx"
    # x = np.linspace(spatial_range4[0], spatial_range4[1], num_points)
    # x, u_data4 = solver_bvp(x, f4)

    # experiments(f4, u_data4, num_points, spatial_range4, postfix)
    ########################### TEST №4 ###########################

    ########################### TEST №5 ###########################
    spatial_range5 = [0, np.pi]

    def f5(x):
        return x * tf.sin(x**2)

    def U5(x):
        c1 = tf.math.special.fresnel_cos(tf.sqrt(2 / np.pi) * x)
        c2 = tf.math.special.fresnel_cos(tf.sqrt(2 * np.pi)) * x

        u = (np.pi * c1 - c2) / (2 * tf.sqrt(2 * np.pi))

        return u

    postfix = "xsin(xx)"
    experiments(f5, U5, num_points, spatial_range5, postfix)

    # postfix = "xsin(xx)_approx"
    # x = np.linspace(spatial_range5[0], spatial_range5[1], num_points)
    # x, u_data6 = solver_bvp(x, f5)

    # experiments(f5, u_data6, num_points, spatial_range5, postfix)
    ########################### TEST №5 ###########################

    ########################### TEST №6 ###########################
    spatial_range6 = [0, np.pi]

    def p(x):
        return 1 * tf.ones_like(x, dtype=tf.float32)
    
    def q(x):
        return 4 * tf.ones_like(x, dtype=tf.float32)

    def f6(x):
        return -tf.cos(x)

    def U6(x):
        return (tf.exp(2*np.pi - 2*x) - tf.exp(2*x) - np.exp(2 * np.pi)*tf.cos(x) + tf.cos(x)) / (5*np.exp(2*np.pi) - 5)

    postfix = "-y'' + 4*y = -cos(x)"
    experiments(f6, U6, num_points, spatial_range6, postfix, p, q)
    ########################### TEST №6 ###########################

    # ########################### TEST №7 ###########################
    # spatial_range7 = [-np.pi/2, np.pi/2]

    # def p2(x):
    #     return tf.exp(x)
    
    # def q2(x):
    #     return tf.zeros_like(x, dtype=tf.float32)

    # def f7(x):
    #     return tf.sin(x)

    # def U7(x):
    #     return tf.exp(-x) / (2*np.exp(np.pi) - 1)*(-2*tf.exp(x + np.pi/2) + (np.exp(np.pi) - 1)*tf.sin(x) - ((np.exp(np.pi) - 1)*tf.cos(x)) + np.exp(np.pi) + 1)

    # postfix = "-exp(x)*y'' - exp(x)*y = sin(x)"
    # experiments(f7, U7, num_points, spatial_range7, postfix, p2, q2)
    # ########################### TEST №7 ###########################

    print("All experiments finished!")
