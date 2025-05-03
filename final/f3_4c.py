import numpy as np

def approx_derivative_ln(x0, h):
    return (np.log(x0 + 2*h) - np.log(x0 - h)) / (3*h)

x0 = 2
exact = 1 / x0
hs = [0.1, 0.05, 0.025]
errors = []

for h in hs:
    approx = approx_derivative_ln(x0, h)
    error = abs(approx - exact)
    errors.append(error)
    print(f"h = {h:.3f}, Approx = {approx:.8f}, Error = {error:.2e}")

for i in range(2):
    rate = np.log(errors[i]/errors[i+1]) / np.log(hs[i]/hs[i+1])
    print(f"Order between h={hs[i]} and h={hs[i+1]} ≈ {rate:.2f}")