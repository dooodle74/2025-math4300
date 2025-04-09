import numpy as np
import matplotlib.pyplot as plt

# Dataset
X = np.array([
    [1, 2],
    [2, 3],
    [-1, -1],
    [-2, -3]
])
y = np.array([1, 1, -1, -1])

# Initialize weights and bias
w = np.zeros(2)
b = 0
eta = 1  # Learning rate

def sign(x):
    return 1 if x >= 0 else -1

# Plotting helper
def plot_decision_boundary(X, y, w, b, epoch):
    plt.figure()
    for i in range(len(X)):
        if y[i] == 1:
            plt.scatter(X[i][0], X[i][1], color='blue', marker='o', label='Positive' if i == 0 else "")
        else:
            plt.scatter(X[i][0], X[i][1], color='red', marker='x', label='Negative' if i == 2 else "")

    # Decision boundary: w1*x + w2*y + b = 0 → y = -(w1*x + b)/w2
    x_vals = np.linspace(-4, 4, 100)
    if w[1] != 0:
        y_vals = -(w[0] * x_vals + b) / w[1]
        plt.plot(x_vals, y_vals, 'k--', label='Decision boundary')
    else:
        plt.axvline(-b/w[0], color='k', linestyle='--', label='Decision boundary')

    plt.title(f"Epoch {epoch + 1}")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.grid(True)
    plt.legend()
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)
    plt.show()

# Training function
def train_perceptron(X, y, w, b, eta=1, max_iter=10):
    for epoch in range(max_iter):
        print(f"Epoch {epoch + 1}")
        error_count = 0
        for i in range(len(X)):
            activation = np.dot(w, X[i]) + b
            prediction = sign(activation)
            if prediction != y[i]:
                w += eta * y[i] * X[i]
                b += eta * y[i]
                error_count += 1
                print(f"  Misclassified x{i+1}: updated w = {w}, b = {b}")
            else:
                print(f"  Correctly classified x{i+1}")
        
        plot_decision_boundary(X, y, w, b, epoch)

        if error_count == 0:
            print("Converged.\n")
            break
    return w, b

# Run the training
final_w, final_b = train_perceptron(X, y, w, b)
print(f"Final weights: {final_w}, Final bias: {final_b}")
