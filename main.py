import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time
import os
import shutil

class VariationalNeuralNetwork:
    """
    Реализация метода Ритца с использованием нейронных сетей для решения дифференциальных уравнений.
    В данном примере решается уравнение Пуассона: -u''(x) = f(x) с граничными условиями u(0) = u(1) = 0.
    """

    def __init__(self, spatial_range=[0, 1], num_hidden=20, batch_size=200, num_iters=1000, lr_rate=1e-3, output_path='Varnn_modern'):
        """
        Инициализация параметров модели.

        Args:
            spatial_range (list): Пространственная область (например, [0, 1]).
            num_hidden (int): Количество нейронов в каждом скрытом слое.
            batch_size (int): Размер батча для обучения.
            num_iters (int): Количество итераций обучения.
            lr_rate (float): Скорость обучения.
            output_path (str): Путь для сохранения обученной модели.
        """
        self.spatial_range = spatial_range
        self.num_hidden = num_hidden
        self.batch_size = batch_size
        self.num_iters = num_iters
        self.lr_rate = lr_rate
        self.output_path = output_path
        self.loss_history = []  # История изменения функции потерь
        self.model = self.build_model() # Создание модели
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr_rate)  # Оптимизатор Adam


    def build_model(self):
        """
        Создание полносвязной нейронной сети (Fully Connected Network).

        Returns:
            tf.keras.Model: Скомпилированная модель Keras.
        """
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(self.num_hidden, activation='tanh', input_shape=(1,)),  # Входной слой
            tf.keras.layers.Dense(self.num_hidden, activation='tanh'),  # Скрытый слой 1
            tf.keras.layers.Dense(self.num_hidden, activation='tanh'),  # Скрытый слой 2
            tf.keras.layers.Dense(1)  # Выходной слой (одно значение)
        ])
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

    def right_hand_side(self, x):
        """
        Правая часть уравнения Пуассона: f(x) = (pi**2)*sin(pi*x).

        Args:
            x (tf.Tensor): Входные данные (значения x).

        Returns:
            tf.Tensor: Значение правой части уравнения в точке x.
        """
        return (np.pi**2) * tf.sin(np.pi * x)

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

        loss = tf.reduce_mean(0.5 * tf.square(du_dx) - self.right_hand_side(x) * u)
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

            if (iteration + 1) % 100 == 0:
                print(f'Iteration: {iteration + 1}, Loss: {loss.numpy():.4f}')

        # Сохранение обученной модели
        self.model.save(os.path.join(self.output_path, 'model.keras'))

    def predict(self, x):
        """
        Предсказание значения u(x) с использованием обученной нейронной сети.

        Args:
            x (np.ndarray): Входные значения x, для которых нужно сделать предсказание.

        Returns:
            np.ndarray: Предсказанные значения u(x).
        """
        x_tf = tf.convert_to_tensor(x.reshape(-1, 1), dtype=tf.float32) # Преобразуем в тензор
        u_nn = self.model(x_tf) # Прогоняем через модель
        u = self.bubble_function(x_tf) * u_nn # Учитываем граничные условия
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

# Main execution block
if __name__ == "__main__":
    # Define parameters
    spatial_range = [0, 1]
    num_hidden = 32
    batch_size = 128
    num_iters = 2000
    learning_rate = 1e-3

    # Create and train the model
    varnn = VariationalNeuralNetwork(spatial_range, num_hidden, batch_size, num_iters, learning_rate)
    start_time = time.time()
    varnn.train()
    end_time = time.time()
    training_time = end_time - start_time
    print(f"Training time: {training_time:.2f} seconds")

    # Generate test data
    num_points = 100
    x_test = np.linspace(spatial_range[0], spatial_range[1], num_points)

    # Make predictions
    u_predicted = varnn.predict(x_test)
    du_dx_predicted = varnn.predict_derivative(x_test)

    # Compute exact solution
    u_exact = np.sin(np.pi * x_test)
    du_dx_exact = np.pi * np.cos(np.pi * x_test)

    # Plot results
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(x_test, u_predicted, label='Predicted')
    plt.plot(x_test, u_exact, label='Exact')
    plt.xlabel('x')
    plt.ylabel('u(x)')
    plt.title('Solution u(x)')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(x_test, du_dx_predicted, label='Predicted')
    plt.plot(x_test, du_dx_exact, label='Exact')
    plt.xlabel('x')
    plt.ylabel("u'(x)")
    plt.title("Derivative u'(x)")
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Plot loss history
    plt.figure(figsize=(8, 6))
    plt.plot(varnn.loss_history)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Loss History')
    plt.show()
