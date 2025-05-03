import numpy as np
import matplotlib.pyplot as plt

# Define the actual function
def f(x):
    return np.sin(x)

# Define the Lagrange interpolating polynomial P(x)
def P(x):
    pi = np.pi
    # L1(x)
    L1 = x * (x - pi/2) / (-pi**2 / 16)
    # L2(x)
    L2 = x * (x - pi/4) / (pi**2 / 8)
    return (np.sqrt(2)/2) * L1 + L2

# Generate x values in [0, pi/2]
x_vals = np.linspace(0, np.pi/2, 500)
sin_vals = f(x_vals)
poly_vals = P(x_vals)
error_vals = poly_vals - sin_vals

# Plot sin(x) and the interpolating polynomial
plt.figure(figsize=(10, 5))
plt.plot(x_vals, sin_vals, label='f(x) = sin(x)', color='blue')
plt.plot(x_vals, poly_vals, label='P(x) - Lagrange Polynomial', color='orange', linestyle='--')
plt.title('Interpolating Polynomial vs sin(x)')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()

# Plot the difference
plt.figure(figsize=(10, 4))
plt.plot(x_vals, error_vals, label='P(x) - sin(x)', color='red')
plt.title('Error Between Polynomial and sin(x)')
plt.xlabel('x')
plt.ylabel('Error')
plt.grid(True)
plt.legend()
plt.show()
