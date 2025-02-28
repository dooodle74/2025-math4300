import numpy as np
import sympy as sp

def secant_method(f, p0, p1, tol=5e-7, max_iter=100):
    iterations = []
    for i in range(max_iter):
        f_p0, f_p1 = f(p0), f(p1)
        if abs(f_p1 - f_p0) < tol: 
            break

        p_next = p1 - f_p1 * (p1 - p0) / (f_p1 - f_p0)
        iterations.append((i + 1, p_next, f(p_next)))
        
        if abs(p_next - p1) < tol: 
            break

        p0, p1 = p1, p_next

    return iterations

x = sp.symbols('x')
f_expr = x**3 + 2*x**2 - 3*x - 1
f = sp.lambdify(x, f_expr, 'numpy')

cases = {
    "Case (a)": (-3, -2),
    "Case (b)": (-2, -3),
    "Case (c)": (-4, -2),
    "Case (d)": (-2, -4)
}

for case, (p0, p1) in cases.items():
    print(f"\nSecant Method {case} (p0 = {p0}, p1 = {p1}):")
    iterations = secant_method(f, p0, p1, tol=5e-7, max_iter=100)
    
    for i, p_n, f_pn in iterations:
        print(f"Iteration {i}: p_n = {p_n:.10f}, f(p_n) = {f_pn:.10f}")