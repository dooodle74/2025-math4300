import numpy as np

# Given data
x = np.array([0, 5, 10, 15, 20, 25, 30, 35, 40, 45], dtype=float)
y = np.array([320, 325, 331, 338, 345, 354, 360, 369, 379, 389], dtype=float)

# Normalize x and y to improve gradient descent performance
x = (x - np.mean(x)) / np.std(x)
y_mean, y_std = np.mean(y), np.std(y)
y = (y - y_mean) / y_std

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def gradient_descent(x, y, model, lr=1e-5, epsilon=3.907e-12, max_epochs=1000):
    params = np.random.uniform(-1, 1, size=model.param_count)  # Better initial values
    prev_loss = float('inf')
    
    for epoch in range(max_epochs):
        y_pred, gradients = model.compute_gradients(x, y, params)
        params -= lr * gradients  # Gradient Descent update
        loss = mean_squared_error(y, y_pred)
        
        if abs(prev_loss - loss) < epsilon:
            break
        prev_loss = loss
    
    return params, loss

# Linear Model: y = mx + b
class LinearModel:
    param_count = 2
    
    @staticmethod
    def compute_gradients(x, y, params):
        m, b = params
        y_pred = m * x + b
        grad_m = -2 * np.mean(x * (y - y_pred))
        grad_b = -2 * np.mean(y - y_pred)
        return y_pred, np.array([grad_m, grad_b])

# Quadratic Model: y = ax^2 + bx + c
class QuadraticModel:
    param_count = 3
    
    @staticmethod
    def compute_gradients(x, y, params):
        a, b, c = params
        y_pred = a * x**2 + b * x + c
        grad_a = -2 * np.mean(x**2 * (y - y_pred))
        grad_b = -2 * np.mean(x * (y - y_pred))
        grad_c = -2 * np.mean(y - y_pred)
        return y_pred, np.array([grad_a, grad_b, grad_c])

# Exponential Model: y = a * e^(bx) + c
class ExponentialModel:
    param_count = 3
    
    @staticmethod
    def compute_gradients(x, y, params):
        a, b, c = params
        exp_bx = np.exp(np.clip(b * x, -10, 10))  # Prevent overflow
        y_pred = a * exp_bx + c
        grad_a = -2 * np.mean(exp_bx * (y - y_pred))
        grad_b = -2 * np.mean(x * a * exp_bx * (y - y_pred))
        grad_c = -2 * np.mean(y - y_pred)
        return y_pred, np.array([grad_a, grad_b, grad_c])

# Logarithmic Model: y = a * ln(bx + c)
class LogarithmicModel:
    param_count = 3
    
    @staticmethod
    def compute_gradients(x, y, params):
        a, b, c = params
        bx_c = np.clip(b * x + c, 1e-6, None)  # Ensure log input is positive
        y_pred = a * np.log(bx_c)
        grad_a = -2 * np.mean(np.log(bx_c) * (y - y_pred))
        grad_b = -2 * np.mean(a * x / bx_c * (y - y_pred))
        grad_c = -2 * np.mean(a / bx_c * (y - y_pred))
        return y_pred, np.array([grad_a, grad_b, grad_c])

# Running Gradient Descent on Each Model
models = {
    'Linear': LinearModel(),
    'Quadratic': QuadraticModel(),
    'Exponential': ExponentialModel(),
    'Logarithmic': LogarithmicModel()
}

results = {}
for name, model in models.items():
    params, final_loss = gradient_descent(x, y, model, lr=1e-5)  # Lower LR for all models
    results[name] = (params, final_loss)
    print(f"{name} Model: Parameters = {params}, Final Loss = {final_loss}")

