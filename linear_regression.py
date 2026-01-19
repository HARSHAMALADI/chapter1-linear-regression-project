import numpy as np


def gradient_descent(X, y, lr=0.01, epochs=1000):
    """Perform batch gradient descent for linear regression.

    Args:
        X (ndarray): Feature matrix (m x n).
        y (ndarray): Target vector (m,).
        lr (float): Learning rate.
        epochs (int): Number of iterations.

    Returns:
        theta (ndarray): Learned parameter vector (n,).
    """
    m, n = X.shape
    theta = np.zeros(n)
    for _ in range(epochs):
        predictions = X.dot(theta)
        errors = predictions - y
        gradient = (1 / m) * X.T.dot(errors)
        theta -= lr * gradient
    return theta


def normal_equation(X, y):
    """Compute parameters using the normal equation.

    Args:
        X (ndarray): Feature matrix (m x n).
        y (ndarray): Target vector (m,).

    Returns:
        theta (ndarray): Parameter vector computed analytically.
    """
    # Add small lambda for numerical stability (regularization)
    lambda_identity = 1e-8 * np.eye(X.shape[1])
    return np.linalg.inv(X.T.dot(X) + lambda_identity).dot(X.T).dot(y)


if __name__ == "__main__":
    # Example usage with a toy dataset
    # Create a simple dataset
    X = np.array([
        [1.0, 1.0],
        [1.0, 2.0],
        [1.0, 3.0],
        [1.0, 4.0],
    ])
    y = np.array([2.0, 3.0, 4.0, 5.0])

    # Fit using gradient descent
    theta_gd = gradient_descent(X, y, lr=0.1, epochs=1000)
    print("Parameters from gradient descent:", theta_gd)

    # Fit using normal equation
    theta_ne = normal_equation(X, y)
    print("Parameters from normal equation:", theta_ne)
