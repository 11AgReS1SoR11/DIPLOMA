import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time
from varnn import VariationalNeuralNetwork, l2_norm_square
from derivative import calculate_auto_derivative, get_cubic_interpolation


def solver(f, U, spatial_range, num_hidden, batch_size, num_iters, learning_rate, num_layers, optimizer, filename = "output"):
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
    num_points = 150
    x_test = np.linspace(spatial_range[0], spatial_range[1], num_points)

    # Make predictions
    u_predicted = varnn.predict(x_test)
    du_dx_predicted = varnn.predict_derivative(x_test)

    # Compute exact solution
    u_exact = U(x_test)
    du_dx_exact = get_cubic_interpolation(x_test, u_exact, derivative=1) # the same np.pi * np.cos(np.pi * x_test) or calculate_auto_derivative(x_test, U).numpy()

    ### ERRORS ###

    aposterrori_error_norm_estimate = varnn.compute_aposterrori_error_estimate(x_test, U)
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



if __name__ == "__main__":
    # Define parameters
    spatial_range = [0, 1]
    num_hidden = 20
    batch_size = 200
    num_iters = 2000
    learning_rate = 1e-3
    num_layers = 3
    optimizer = 'adam'

    def f(x):
        return (np.pi**2) * tf.sin(np.pi * x)

    def U(x):
        return tf.sin(np.pi * x)

    solver(f, U, spatial_range, num_hidden, batch_size, num_iters, learning_rate, num_layers, optimizer, "solve_glad")


    spatial_range2 = [-1, 1]

    def f2(x):
        return tf.where(x < 0, -1.0, 1.0)

    def U2(x):
        return tf.where(x < 0, 0.5 * x * (x + 1), -0.5 * x * (x - 1))

    solver(f2, U2, spatial_range2, num_hidden, batch_size, num_iters, learning_rate, num_layers, optimizer, "solve_not_glad")

