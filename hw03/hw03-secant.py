import numpy as np
import sympy as sp

def secant_method(f, p0, p1, tol=1e-10, max_iter=10):
    iterations = []
    for i in range(max_iter):
        f_p0, f_p1 = f(p0), f(p1)
        if abs(f_p1 - f_p0) < tol:
            break

        p_next = p1 - f_p1 * (p1 - p0) / (f_p1 - f_p0)
        iterations.append((i + 1, p_next, f(p_next)))
        p0, p1 = p1, p_next

        if abs(f(p_next)) < tol:
            break

    return iterations

x = sp.symbols('x')
f1_expr = x * (1 - sp.cos(x))
f2_expr = 27*x**4 + 162*x**3 - 180*x**2 + 62*x - 7
f3_expr = (x / (1 + x**2)) - (500 / 841) * (1 - (21*x / 125))

f1 = sp.lambdify(x, f1_expr, 'numpy')
f2 = sp.lambdify(x, f2_expr, 'numpy')
f3 = sp.lambdify(x, f3_expr, 'numpy')

iterations_f1 = secant_method(f1, -1, 2)
iterations_f2 = secant_method(f2, 0.2, 0.4)  # Around 1/3
iterations_f3 = secant_method(f3, 2.3, 2.7)  # Around 2.5

print("\nSecant Method for f1:")
for i, p_n, f_pn in iterations_f1:
    print(f"Iteration {i}: p_n = {p_n:.10f}, f(p_n) = {f_pn:.10f}")

print("\nSecant Method for f2:")
for i, p_n, f_pn in iterations_f2:
    print(f"Iteration {i}: p_n = {p_n:.10f}, f(p_n) = {f_pn:.10f}")

print("\nSecant Method for f3:")
for i, p_n, f_pn in iterations_f3:
    print(f"Iteration {i}: p_n = {p_n:.10f}, f(p_n) = {f_pn:.10f}")