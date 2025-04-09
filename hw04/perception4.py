import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 3D Dataset: linearly separable
X = np.array([
    [2, 3, 1],
    [1, 1, 1],
    [2, 0, 3],
    [-1, -1, -1],
    [-2, -3, -1],
    [-1, -2, -4]
])
y = np.array([1, 1, 1, -1, -1, -1])

# Initialize weights and bias
w = np.zeros(3)
b = 0
eta = 1  # Learning rate

def sign(x):
    return 1 if x >= 0 else -1

def plot_3d(X, y, w, b, epoch):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot positive and negative samples
    for i in range(len(X)):
        if y[i] == 1:
            ax.scatter(X[i][0], X[i][1], X[i][2], color='blue', marker='o', label='Positive' if i == 0 else "")
        else:
            ax.scatter(X[i][0], X[i][1], X[i][2], color='red', marker='^', label='Negative' if i == 3 else "")

    # Create a plane if w[2] ≠ 0
    if w[2] != 0:
        xx, yy = np.meshgrid(np.linspace(-4, 4, 10), np.linspace(-4, 4, 10))
        zz = -(w[0] * xx + w[1] * yy + b) / w[2]
        ax.plot_surface(xx, yy, zz, alpha=0.4, color='green', label='Decision plane')

    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('x3')
    ax.set_title(f"Epoch {epoch + 1}")
    ax.legend()
    plt.show()

def train_perceptron(X, y, w, b, eta=1, max_iter=10):
    for epoch in range(max_iter):
        print(f"Epoch {epoch + 1}")
        errors = 0
        for i in range(len(X)):
            prediction = sign(np.dot(w, X[i]) + b)
            if prediction != y[i]:
                w += eta * y[i] * X[i]
                b += eta * y[i]
                errors += 1
                print(f"  Misclassified x{i+1}: w = {w}, b = {b}")
            else:
                print(f"  Correctly classified x{i+1}")
        plot_3d(X, y, w, b, epoch)
        if errors == 0:
            print("Converged.")
            break
    return w, b

# Run training and plot
final_w, final_b = train_perceptron(X, y, w, b)
print(f"Final weights: {final_w}, Final bias: {final_b}")