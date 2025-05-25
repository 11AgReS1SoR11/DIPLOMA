import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time
# from warnings import deprecated
from varnn import VariationalNeuralNetwork, l2_norm_square
from derivative import calculate_auto_derivative, get_cubic_interpolation
from Approx import FDM, solver_bvp


# @deprecated("Вычисляет квадрат нормы и использует точное решение, поэтому используй solver, который использует только нейросетевое решение")
def solver_2(f, U, num_points, spatial_range, num_hidden, batch_size, num_iters, learning_rate, num_layers, optimizer, filename):
    """
    вычисляет квадрат нормы
    """
    # Create and train the model
    varnn = VariationalNeuralNetwork(right_hand_side_function=f, spatial_range=spatial_range, num_hidden=num_hidden,
                                      batch_size=batch_size, num_iters=num_iters,
                                      lr_rate=learning_rate, num_layers=num_layers, optimizer=optimizer)
    start_time = time.time()
    varnn.train()
    end_time = time.time()
    training_time = end_time - start_time
    print(f"Training time: {training_time:.2f} seconds")

    # Generate test data
    x_test = np.linspace(spatial_range[0], spatial_range[1], num_points)

    # Make predictions
    u_predicted = varnn.predict(x_test)
    du_dx_predicted = varnn.predict_derivative(x_test)

    # Compute exact solution
    u_exact = U(x_test) if callable(U) else U
    du_dx_exact = get_cubic_interpolation(x_test, u_exact, derivative=1) # the same np.pi * np.cos(np.pi * x_test) or calculate_auto_derivative(x_test, U).numpy()

    ### ERRORS ###

    aposterrori_error_norm_estimate = varnn.compute_aposterrori_error_estimate_2(x_test, U)
    print(f"A Posteriori Error Estimate: {aposterrori_error_norm_estimate:.4f}")

    real_error_norm_dx = l2_norm_square(du_dx_exact.reshape(-1, 1) - du_dx_predicted, spatial_range[0], spatial_range[1])
    print(f"A Real Error dx: {real_error_norm_dx:.4f}")

    print(f"Compare: aposterrori_error_norm_estimate/real_error_norm_dx = { (aposterrori_error_norm_estimate/real_error_norm_dx):.4f}")

    ### ERRORS ###

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
    plt.savefig(f"{filename}.png")

    # Plot loss history
    plt.figure(figsize=(8, 6))
    plt.plot(varnn.loss_history)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Loss History')
    plt.savefig(f"{filename}_Loss_History.png")


def tryExact(num_points, num_hidden, batch_size, num_iters, learning_rate, num_layers, optimizer):

    spatial_range = [0, 1]

    def f(x):
        return (np.pi**2) * tf.sin(np.pi * x)

    def U(x):
        return tf.sin(np.pi * x)

    solver_2(f, U, num_points, spatial_range, num_hidden, batch_size, num_iters, learning_rate, num_layers, optimizer, "exact_glad")

    spatial_range2 = [-1, 1]

    def f2(x):
        return tf.where(x < 0, -1.0, 1.0)

    def U2(x):
        return tf.where(x < 0, 0.5 * x * (x + 1), -0.5 * x * (x - 1))

    solver_2(f2, U2, num_points, spatial_range2, num_hidden, batch_size, num_iters, learning_rate, num_layers, optimizer, "exact_not_glad")


def tryApprox(num_points, num_hidden, batch_size, num_iters, learning_rate, num_layers, optimizer):

    spatial_range = [0, 1]

    def f(x):
        return (np.pi**2) * tf.sin(np.pi * x)

    x = np.linspace(spatial_range[0], spatial_range[1], num_points)
    x, u_data = solver_bvp(x, f)

    solver_2(f, u_data, num_points, spatial_range, num_hidden, batch_size, num_iters, learning_rate, num_layers, optimizer, "approx_glad")

    spatial_range2 = [-1, 1]

    def f2(x):
        return tf.where(x < 0, -1.0, 1.0)

    x = np.linspace(spatial_range2[0], spatial_range2[1], num_points)
    x, u_data2 = solver_bvp(x, f2)

    solver_2(f2, u_data2, num_points, spatial_range2, num_hidden, batch_size, num_iters, learning_rate, num_layers, optimizer, "approx_not_glad")


def solver(f, U, num_points, spatial_range, num_hidden, batch_size, num_iters, learning_rate, num_layers, optimizer, filename, p = None, q = None):
    """
    вычисляет нормы (не квадрат). В качестве приближения, используется решение нейронной сети (y = v')
    """
    # Create and train the model
    varnn = VariationalNeuralNetwork(right_hand_side_function=f, spatial_range=spatial_range, num_hidden=num_hidden, batch_size=batch_size,num_iters=num_iters,
                                      lr_rate=learning_rate, num_layers=num_layers, optimizer=optimizer, p_function=p, q_function=q)
    start_time = time.time()
    varnn.train()
    end_time = time.time()
    training_time = end_time - start_time
    print(f"Training time: {training_time:.2f} seconds")

    # Generate test data
    x_test = np.linspace(spatial_range[0], spatial_range[1], num_points)

    # Make predictions
    u_predicted = varnn.predict(x_test)
    du_dx_predicted = varnn.predict_derivative(x_test)

    # Compute exact solution
    u_exact = U(x_test)
    du_dx_exact = get_cubic_interpolation(x_test, u_exact, derivative=1)

    ### ERRORS ###

    aposterrori_error_norm_estimate = varnn.compute_aposterrori_error_estimate(x_test)
    print(f"A Posteriori Error Estimate: {aposterrori_error_norm_estimate:.4f}")

    real_error_norm_dx = tf.cast(tf.sqrt(l2_norm_square(du_dx_exact.reshape(-1, 1) - du_dx_predicted, spatial_range[0], spatial_range[1])), dtype=tf.float32)
    print(f"A Real Error dx: {real_error_norm_dx:.4f}")

    print(f"Compare: aposterrori_error_norm_estimate/real_error_norm_dx = { (aposterrori_error_norm_estimate/real_error_norm_dx):.4f}")

    ### ERRORS ###

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
    plt.savefig(f"{filename}.png")

    # Plot loss history
    plt.figure(figsize=(8, 6))
    plt.plot(varnn.loss_history)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Loss History')
    plt.savefig(f"{filename}_Loss_History.png")


if __name__ == "__main__":
    # Define parameters
    num_hidden = 20
    batch_size = 128
    num_iters = 2000
    learning_rate = 1e-3
    num_layers = 3
    optimizer = 'adam'
    num_points = 100

    #### This code use exact solution = deprecated ####
    # tryExact(num_points, num_hidden, batch_size, num_iters, learning_rate, num_layers, optimizer)

    # tryApprox(num_points, num_hidden, batch_size, num_iters, learning_rate, num_layers, optimizer)
    ###################################################


    # ########################## TEST №1 ###########################
    # filename = "(pi**2)*sin(pi*x)"

    # spatial_range = [0, 1]

    # def f(x):
    #     return (np.pi**2) * tf.sin(np.pi * x)

    # def U(x):
    #     return tf.sin(np.pi * x)
    # ########################## TEST №1 ###########################

    # ########################## TEST №2 ###########################
    # filename = "-1+1"

    # spatial_range = [-1, 1]

    # def f(x):
    #     return tf.where(x < 0, -1.0, 1.0)

    # def U(x):
    #     return tf.where(x < 0, 0.5 * x * (x + 1), -0.5 * x * (x - 1))
    # ########################## TEST №2 ###########################

    # ########################## TEST №3 ###########################
    # filename = "oscillating"

    # spatial_range = [0, 1]

    # def f(x):
    #     return 100 * (np.pi**2) * tf.sin(10 * np.pi * x)

    # def U(x):
    #     return tf.sin(10 * np.pi * x)
    # ########################## TEST №3 ###########################

    # ########################## TEST №4 ###########################
    # filename = "Peak"

    # spatial_range = [-5, 5]
    # peak_center = 0
    # peak_width = 0.5
    # peak_height = 10

    # def f(x):
    #     return peak_height * tf.exp(-((x - peak_center)**2) / (2 * peak_width**2))

    # def U(x):
    #     return -6.26657*x*tf.math.erf(np.sqrt(2)*x) - 2.5*tf.exp(-2*x*x) + 31.3329
    # ########################## TEST №4 ###########################

    # ########################### TEST №5 ###########################
    # filename = "xsin(x*x)"

    # spatial_range = [0, np.pi]

    # def f(x):
    #     return x * tf.sin(x**2)

    # def U(x):
    #     c1 = tf.math.special.fresnel_cos(tf.sqrt(2 / np.pi) * x)
    #     c2 = tf.math.special.fresnel_cos(tf.sqrt(2 * np.pi)) * x

    #     u = (np.pi * c1 - c2) / (2 * tf.sqrt(2 * np.pi))

    #     return u
    # ########################### TEST №5 ###########################

    # ########################### TEST №6 ###########################
    # filename = "-y'' + 4*y = -cos(x)"

    # spatial_range = [0, np.pi]

    # def p(x):
    #     return tf.ones_like(x, dtype=tf.float32)
    
    # def q(x):
    #     return 4 * tf.ones_like(x, dtype=tf.float32)

    # def f(x):
    #     return -tf.cos(x)

    # def U(x):
    #     return (tf.exp(2*np.pi - 2*x) - tf.exp(2*x) - np.exp(2 * np.pi)*tf.cos(x) + tf.cos(x)) / (5*np.exp(2*np.pi) - 5)
    # ########################### TEST №6 ###########################

    ########################### TEST №7 ###########################
    filename = "-exp(x)*y'' - exp(x)*y' = sin(x)"

    spatial_range = [-np.pi/2, np.pi/2]

    def p(x):
        return tf.cast(tf.exp(x), dtype=tf.float32)
    
    def q(x):
        return tf.zeros_like(x, dtype=tf.float32)

    def f(x):
        return tf.sin(x)

    def U(x):
        return tf.exp(-x) / (2*np.exp(np.pi) - 1)*(-2*tf.exp(x + np.pi/2) + (np.exp(np.pi) - 1)*tf.sin(x) - ((np.exp(np.pi) - 1)*tf.cos(x)) + np.exp(np.pi) + 1)
    ########################### TEST №7 ###########################
    
    solver(f, U, num_points, spatial_range, num_hidden, batch_size, num_iters, learning_rate, num_layers, optimizer, filename, p, q)
