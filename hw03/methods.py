import numpy as np

def redlich_kwong(V, R, T, P, a, b):
    term1 = (R * T) / (V - b)
    term2 = a / (V * (V + b) * np.sqrt(T))
    return P - term1 + term2

def secant_method(f, p0, p1, tol=1e-6, max_iter=100):
    for i in range(max_iter):
        f_p0, f_p1 = f(p0), f(p1)
        
        if abs(f_p1 - f_p0) < tol: 
            break
        
        p_next = p1 - f_p1 * (p1 - p0) / (f_p1 - f_p0)
        
        if abs(p_next - p1) < tol:
            return p_next
        
        p0, p1 = p1, p_next
    
    return p1