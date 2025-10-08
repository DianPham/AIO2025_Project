import numpy as np
import matplotlib.pyplot as plt

def prepare_data(file_path):
    """Load CSV and split into X (features) and y (target)."""
    data = np.genfromtxt(file_path, delimiter=',', skip_header=1)
    X = data[:, :-1]  # all columns except last
    y = data[:, -1]   # last column
    return X, y

def initialize_params(n_features):
    """Initialize weights and bias."""
    W = np.random.normal(0, 0.01, size=n_features)
    b = 0.0
    return W, b

def predict(X, W, b):
    """Predict outputs for all samples."""
    return X @ W + b  # matrix multiplication

def compute_loss(y_hat, y):
    """Mean squared error loss."""
    return np.mean((y_hat - y) ** 2)

def compute_gradients(X, y, y_hat):
    """Gradients wrt weights and bias."""
    N = len(y)
    error = y_hat - y
    dW = (2/N) * (X.T @ error)
    db = (2/N) * np.sum(error)
    return dW, db

def update_params(W, b, dW, db, lr):
    """Gradient descent update."""
    W -= lr * dW
    b -= lr * db
    return W, b

def linear_regression(X, y, epochs=1000, lr=1e-5):
    """Train linear regression using gradient descent."""
    n_features = X.shape[1]
    W, b = initialize_params(n_features)
    losses = []

    for epoch in range(epochs):
        y_hat = predict(X, W, b)
        loss = compute_loss(y_hat, y)
        dW, db = compute_gradients(X, y, y_hat)
        W, b = update_params(W, b, dW, db, lr)
        losses.append(loss)

    return W, b, losses

# -------------------------
# Run training
X, y = prepare_data('assets/advertising.csv')
W, b, losses = linear_regression(X, y, epochs=50, lr=1e-5)

print("Trained weights:", W)
print("Bias:", b)

plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.show()
