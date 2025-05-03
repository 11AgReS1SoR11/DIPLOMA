import numpy as np
from varnn import VariationalNeuralNetwork, l2_norm_square
import time
import pandas as pd
import matplotlib.pyplot as plt


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


def run_experiment(num_hidden, num_layers, num_iters, learning_rate, batch_size = 128, optimizer = 'adam'):
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
    spatial_range = [0, 1]
    num_points = 100  # Количество точек для вычисления ошибки
    x_test = np.linspace(spatial_range[0], spatial_range[1], num_points)  # Тестовые данные
    u_exact = np.sin(np.pi * x_test)  # Точное решение
    du_dx_exact = np.pi * np.cos(np.pi * x_test)

    # Создаем и обучаем модель
    varnn = VariationalNeuralNetwork(spatial_range=spatial_range, num_hidden=num_hidden,
                                      batch_size=batch_size, num_iters=num_iters,
                                      lr_rate=learning_rate, num_layers=num_layers, optimizer=optimizer)
    start_time = time.time()
    varnn.train()
    training_time = time.time() - start_time

    # Получаем предсказания
    u_predicted = varnn.predict(x_test)

    # Вычисляем апостериорную оценку ошибки и реальную ошибку
    aposterrori_error_estimate = varnn.compute_aposterrori_error_estimate(x_test)

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

if __name__ == "__main__":

    postfix = "ppsin(px)" # set for different filenames

    # 1) Эксперименты с количеством нейронов
    neuron_counts = [10, 16, 32, 64]
    neuron_data = []
    for num_hidden in neuron_counts:
        results = run_experiment(num_hidden=num_hidden, num_layers=3, num_iters=2000, learning_rate=1e-3)
        neuron_data.append(results)

    neuron_df = pd.DataFrame(neuron_data)
    create_table(neuron_df.to_dict('list'), f"results_neurons_{postfix}.png")

    # 2) Эксперименты с количеством слоев
    layer_counts = [2, 3, 4, 5]
    layer_data = []
    for num_layers in layer_counts:
        results = run_experiment(num_hidden=32, num_layers=num_layers, num_iters=2000, learning_rate=1e-3)
        layer_data.append(results)

    layer_df = pd.DataFrame(layer_data)
    create_table(layer_df.to_dict('list'), f"results_layers_{postfix}.png")

    # 3) Эксперименты с количеством итераций
    iterations_counts = [1000, 2000, 3000, 4000]
    iters_data = []
    for num_iters in iterations_counts:
        results = run_experiment(num_hidden=32, num_layers=3, num_iters=num_iters, learning_rate=1e-3)
        iters_data.append(results)

    iters_df = pd.DataFrame(iters_data)
    create_table(iters_df.to_dict('list'), f"results_iters_{postfix}.png")

    # 4) Эксперименты со скоростью обучения
    learning_rates = [1e-2, 1e-3, 1e-4, 1e-5]
    lr_data = []
    for learning_rate in learning_rates:
        results = run_experiment(num_hidden=32, num_layers=3, num_iters=2000, learning_rate=learning_rate)
        lr_data.append(results)

    lr_df = pd.DataFrame(lr_data)
    create_table(lr_df.to_dict('list'), f"results_lr_{postfix}.png")

    # 5) Эксперименты с разными оптимизаторами
    optimizers = ['adam', 'sgd', 'rmsprop']
    optimizer_data = []
    for optimizer in optimizers:
        results = run_experiment(num_hidden=32, num_layers=3, num_iters=2000, learning_rate=1e-3, optimizer=optimizer)
        optimizer_data.append(results)

    optimizer_df = pd.DataFrame(optimizer_data)
    create_table(optimizer_df.to_dict('list'), f"results_optimizers_{postfix}.png")

    # 6) Эксперименты с разными оптимизаторами
    batch_sizes = [64, 128, 256, 512, 1024]
    batch_size_data = []
    for batch_size in batch_sizes:
        results = run_experiment(num_hidden=32, num_layers=3, num_iters=2000, learning_rate=1e-3, batch_size=batch_size)
        batch_size_data.append(results)

    optimizer_df = pd.DataFrame(batch_size_data)
    create_table(optimizer_df.to_dict('list'), f"results_batchSize_{postfix}.png")

    # # 7) Best parametrs
    # optimizers = ['adam', 'sgd', 'rmsprop']
    # best_data = []
    # for optimizer in optimizers:
    #     results = run_experiment(num_hidden=128, num_layers=4, num_iters=2000, learning_rate=1e-4, optimizer=optimizer)
    #     best_data.append(results)

    # best_df = pd.DataFrame(best_data)
    # create_table(best_df.to_dict('list'), f"best_{postfix}.png")

    print("All experiments finished!")



























































# Попробовал через производные
# def run_experiment(num_hidden, num_layers, num_iters, learning_rate, optimizer = 'adam'):
#     """
#     Проводит эксперимент с заданными параметрами и возвращает результаты.
#     Использует tf.GradientTape() для вычисления производной.

#     Args:
#         num_hidden (int): Количество нейронов в слое.
#         num_layers (int): Количество слоев в сети.
#         num_iters (int): Количество итераций обучения.
#         learning_rate (float): Скорость обучения.

#     Returns:
#         dict: Словарь с результатами эксперимента.
#     """

#     # Параметры для нашей задачи
#     spatial_range = [0, 1]
#     batch_size = 1000
#     num_points = 1000  # Количество точек для вычисления ошибки
#     x_test = np.linspace(spatial_range[0], spatial_range[1], num_points)  # Тестовые данные
#     u_exact = np.sin(np.pi * x_test)  # Точное решение

#     # Создаем и обучаем модель
#     varnn = VariationalNeuralNetwork(spatial_range=spatial_range, num_hidden=num_hidden,
#                                       batch_size=batch_size, num_iters=num_iters,
#                                       lr_rate=learning_rate, num_layers=num_layers, optimizer=optimizer)
#     start_time = time.time()
#     varnn.train()
#     training_time = time.time() - start_time

#     x_test_tf = tf.convert_to_tensor(x_test, dtype=tf.float32)
#     with tf.GradientTape() as tape:
#         tape.watch(x_test_tf)  # Начинаем отслеживать операции с x_test_tf
#         u_predicted_tf = varnn.predict(x_test_tf)  # Получаем предсказания (теперь в виде тензора TensorFlow)
#     u_predicted_prime_tf = tape.gradient(u_predicted_tf, x_test_tf)  # Вычисляем производную

#     # Преобразуем результаты обратно в массивы NumPy
#     u_predicted = u_predicted_tf.numpy().flatten()  # Преобразуем в NumPy и делаем одномерным
#     u_predicted_prime = u_predicted_prime_tf.numpy().flatten() # Преобразуем производную в NumPy

#     # Вычисляем производную точного решения (аналитически)
#     u_exact_prime = np.pi * np.cos(np.pi * x_test)

#     # Вычисляем апостериорную оценку ошибки и реальную ошибку
#     aposterrori_error_estimate = varnn.compute_aposterrori_error_estimate(x_test)
#     real_error = l2_norm_square(u_exact_prime - u_predicted_prime, spatial_range[0], spatial_range[1])

#     # Вычисляем отношение
#     ratio = (aposterrori_error_estimate / real_error) if real_error != 0 else np.nan

#     results = {
#         "Optimizer": optimizer,
#         "Neurons": num_hidden,
#         "Layers": num_layers,
#         "Iterations": num_iters,
#         "Learning Rate": learning_rate,
#         "||M||^2": aposterrori_error_estimate,
#         "||U'-V'||^2": real_error,  # Обновлено название
#         "||M||^2/||U'-V'||^2": ratio, # Обновлено название
#         "Training Time": training_time
#     }

#     return results
