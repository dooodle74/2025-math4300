from methods import redlich_kwong, secant_method

R = 0.08206
T, P, Tc, Pc = 450, 56, 405.5, 111.3

a = 0.42747 * (R**2 * Tc**(5/2)) / Pc
b = 0.08664 * (R * Tc) / Pc

f = lambda V: redlich_kwong(V, R, T, P, a, b)
V = secant_method(f, 0.1, 1)

print(f"Result: {V:.4f} liters")