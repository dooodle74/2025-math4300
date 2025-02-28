import numpy as np
import matplotlib.pyplot as plt

x_values = np.linspace(-5e-8, 5e-8, 1001)

f_direct = 1 - np.cos(x_values)

plt.figure(figsize=(6, 4))
plt.plot(x_values, f_direct, color="blue")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.grid(True)

plt.show()