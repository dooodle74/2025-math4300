from methods import redlich_kwong, secant_method

R = 0.08206
T, P, Tc, Pc = 323.15, 1, 304.2, 72.9

a = 0.42747 * (R**2 * Tc**(5/2)) / Pc
b = 0.08664 * (R * Tc) / Pc

f = lambda V: redlich_kwong(V, R, T, P, a, b)
V = secant_method(f, 20, 30)

print(f"Result: {V:.4f} liters")