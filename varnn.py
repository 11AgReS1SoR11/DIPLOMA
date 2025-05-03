import numpy as np
import tensorflow as tf
import os
import shutil
from derivative import calculate_auto_derivative_2, calculate_auto_derivative, get_cubic_interpolation

def l2_norm_square(x, a, b):
    return (tf.reduce_mean(tf.square(x)) * (b - a)).numpy()

class VariationalNeuralNetwork:
    """
    Реализация метода Ритца с использованием нейронных сетей для решения дифференциальных уравнений.
    В данном примере решается уравнение Пуассона: -u''(x) = f(x) с граничными условиями u(0) = u(1) = 0.
    """

    def __init__(self, right_hand_side_function, spatial_range=[0, 1], num_hidden=20, batch_size=200, num_iters=1000, lr_rate=1e-3, num_layers=3, optimizer='adam', output_path='Varnn_modern'):
        """
        Инициализация параметров модели.

        Args:
            right_hand_side_function (callable): Функция правой части уравнения f(x). Принимает tf.Tensor x и возвращает tf.Tensor f(x).
            spatial_range (list): Пространственная область (например, [0, 1]).
            num_hidden (int): Количество нейронов в каждом скрытом слое.
            batch_size (int): Размер батча для обучения.
            num_iters (int): Количество итераций обучения.
            lr_rate (float): Скорость обучения.
            num_layers (int): Количество слоев в сети.
            output_path (str): Путь для сохранения обученной модели.
            optimizer (str): Выбор оптимизатора ('adam', 'sgd', 'rmsprop').
        """
        self.spatial_range = spatial_range
        self.num_hidden = num_hidden
        self.batch_size = batch_size
        self.num_iters = num_iters
        self.lr_rate = lr_rate
        self.num_layers = num_layers
        self.output_path = output_path
        self.loss_history = []  # История изменения функции потерь
        self.optimizer_name = optimizer # Сохраняем название оптимизатора
        self.right_hand_side_function = right_hand_side_function  # Сохраняем функцию правой части
        self.model = self.build_model() # Создание модели
        self.optimizer = self.get_optimizer()  # Выбираем оптимизатор  <----


    def get_optimizer(self):
        """
        Выбор оптимизатора на основе имени.

        Returns:
            tf.keras.optimizers.Optimizer: Оптимизатор Keras.
        """
        if self.optimizer_name == 'adam':
            return tf.keras.optimizers.Adam(learning_rate=self.lr_rate)
        elif self.optimizer_name == 'sgd':
            return tf.keras.optimizers.SGD(learning_rate=self.lr_rate)
        elif self.optimizer_name == 'rmsprop':
            return tf.keras.optimizers.RMSprop(learning_rate=self.lr_rate)
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer_name}.  Choose 'adam', 'sgd', or 'rmsprop'.")

    def build_model(self):
        """
        Создание полносвязной нейронной сети (Fully Connected Network).

        Returns:
            tf.keras.Model: Скомпилированная модель Keras.
        """
        layers = [tf.keras.layers.Dense(self.num_hidden, activation='tanh', input_shape=(1,))]  # Входной слой
        for _ in range(self.num_layers - 1): # Добавляем скрытые слои
            layers.append(tf.keras.layers.Dense(self.num_hidden, activation='tanh'))
        layers.append(tf.keras.layers.Dense(1))  # Выходной слой (одно значение)

        model = tf.keras.Sequential(layers)
        return model

    def bubble_function(self, x):
        """
        Создание "bubble function" для обеспечения граничных условий Дирихле u(0) = u(1) = 0.

        Args:
            x (tf.Tensor): Входные данные (значения x).

        Returns:
            tf.Tensor: Значение "bubble function" в точке x.
        """
        a = self.spatial_range[0]
        b = self.spatial_range[1]
        return (x - a) * (b - x)

    def compute_loss(self, x):
        """
        Вычисление функции потерь на основе вариационного принципа.

        Args:
            x (tf.Tensor): Входные данные (значения x).

        Returns:
            tf.Tensor: Значение функции потерь.
        """
        with tf.GradientTape() as tape:
            tape.watch(x)  # Отслеживаем градиенты x
            u_nn = self.model(x)  # Выход нейронной сети
            u = self.bubble_function(x) * u_nn  # Приближенное решение с учетом граничных условий
        
        du_dx = tape.gradient(u, x) # du/dx

        loss = tf.reduce_mean(0.5 * tf.square(du_dx) - self.right_hand_side_function(x) * u) * (self.spatial_range[1] - self.spatial_range[0])
        return loss

    @tf.function
    def train_step(self, x):
        """
        Шаг обучения модели.

        Args:
            x (tf.Tensor): Входные данные (значения x).
        """
        with tf.GradientTape() as tape:
            loss = self.compute_loss(x) # Вычисляем loss
        gradients = tape.gradient(loss, self.model.trainable_variables) # Градиенты loss по обучаемым переменным
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables)) # Применяем градиенты

        return loss


    def train(self):
        """
        Обучение нейронной сети.
        """
        if os.path.exists(self.output_path):
            shutil.rmtree(self.output_path)
        os.makedirs(self.output_path)

        for iteration in range(self.num_iters):
            # Генерация случайных точек в пространственной области
            batch_data = self.spatial_range[0] + (self.spatial_range[1] - self.spatial_range[0]) * np.random.rand(self.batch_size, 1).astype(np.float32)
            batch_data_tf = tf.convert_to_tensor(batch_data, dtype=tf.float32)

            # Шаг обучения
            loss = self.train_step(batch_data_tf)
            self.loss_history.append(loss.numpy()) # Сохраняем loss

            # if (iteration + 1) % 100 == 0:
            #     print(f'Iteration: {iteration + 1}, Loss: {loss.numpy():.4f}')

        # Сохранение обученной модели
        self.model.save(os.path.join(self.output_path, 'model.keras'))

    def predict(self, x):
        """
        Предсказание значения u(x) с использованием обученной нейронной сети.

        Args:
            x (np.ndarray or tf.Tensor): Входные значения x, для которых нужно сделать предсказание.

        Returns:
            np.ndarray or tf.Tensor: Предсказанные значения u(x).  Возвращает NumPy array, если на вход был NumPy array, и tf.Tensor, если на вход был tf.Tensor.
        """
        if isinstance(x, tf.Tensor):
            x_tf = x
        else:
            x_tf = tf.convert_to_tensor(x.reshape(-1, 1), dtype=tf.float32) # Преобразуем в тензор

        u_nn = self.model(x_tf) # Прогоняем через модель
        u = self.bubble_function(x_tf) * u_nn # Учитываем граничные условия

        if isinstance(x, tf.Tensor):
            return u
        else:
            return u.numpy()

    def predict_derivative(self, x):
        """
        Предсказание производной u'(x) с использованием обученной нейронной сети.

        Args:
            x (np.ndarray): Входные значения x, для которых нужно сделать предсказание.

        Returns:
            np.ndarray: Предсказанные значения u'(x).
        """
        x_tf = tf.convert_to_tensor(x.reshape(-1, 1), dtype=tf.float32)

        with tf.GradientTape() as tape:
            tape.watch(x_tf)
            u_nn = self.model(x_tf)
            u = self.bubble_function(x_tf) * u_nn

        du_dx = tape.gradient(u, x_tf)
        return du_dx.numpy()

    def compute_aposterrori_error_estimate(self, x, u_exact, beta=1.0):
        """
        Вычисление апостериорной оценки ошибки.

        Args:
            x (np.ndarray): Входные значения x.
            u_exact (callable or tf.Tensor): Точного решения u(x).
            beta (float): Параметр beta в функционале.

        Returns:
            float: Апостериорная оценка ошибки.
        """
        # 1) Находим производную точного решения
        du_exact_dx = calculate_auto_derivative(x, u_exact) if callable(u_exact) else get_cubic_interpolation(x, u_exact, derivative=1) # U'
        d2u_exact_dx2 = calculate_auto_derivative_2(x, u_exact) if callable(u_exact) else get_cubic_interpolation(x, u_exact, derivative=2) # U''

        # 2) Находим производную решения, полученного нейронной сетью
        du_approx_dx = self.predict_derivative(x) # Производная приближенного решения (V')

        # 3) Находим норму разности производных ||U' - V'||
        diff = du_exact_dx - du_approx_dx
        diff = diff.numpy().reshape(-1, 1)
        norm_diff_derivs_squared = l2_norm_square(diff, self.spatial_range[0], self.spatial_range[1])

        # 4) Считаем константу Фридгерца
        a = self.spatial_range[0]
        b = self.spatial_range[1]
        C_Omega = (b - a) / np.pi

        # 5) Находим норму невязки исходного уравнения ||U'' + f(x)||
        x_tf = tf.convert_to_tensor(x.reshape(-1, 1), dtype=tf.float32)
        f_x = self.right_hand_side_function(x_tf)
        residual = d2u_exact_dx2 + f_x  # U'' + f(x)
        residual = residual.numpy().reshape(-1, 1)
        norm_residual_squared = l2_norm_square(residual, self.spatial_range[0], self.spatial_range[1])

        # 6) Считаем оценку M^2
        M = (1 + beta) * (norm_diff_derivs_squared) + (1 + 1 / beta) * (C_Omega**2) * (norm_residual_squared)

        return M
