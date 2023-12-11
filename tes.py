import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
x = np.linspace(0, 10, 100)
y = np.sin(x) + 0.8 * np.random.randn(100)  # Adding some noise
# y = np.sin(x) + 0.2 * np.random.randn(100)


def calculate_rms(data):
    mean_value = np.mean(data)
    squared_diff = [(x - mean_value) ** 2 for x in data]
    mean_squared_diff = np.mean(squared_diff)
    rms_value = np.sqrt(mean_squared_diff)
    return rms_value


plt.plot(x, y, label='Original Data')
plt.title(f'RMS (Smoothness): {calculate_rms(y):.3f}')
plt.legend()
plt.show()
