import numpy as np
import matplotlib.pyplot as plt

bound = np.pi/2

x_values = np.linspace(-5e-8 + bound, 5e-8 + bound, 1001)

f_reformulated = np.cos(x_values) ** 2 / (1+np.sin(x_values))

plt.figure(figsize=(6, 4))
plt.plot(x_values, f_reformulated, color="red")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.legend()
plt.grid(True)

plt.show()