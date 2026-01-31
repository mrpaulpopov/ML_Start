import numpy as np
import matplotlib.pyplot as plt
plt.style.use('dark_background')


# Простой способ понять, как работают numpy array:
# Кодирование точек в 2D с признаком 0 и 1:
features = np.array([[1,0],[0,2],[1,1],[1,2]])
labels = np.array([0,0,1,1])

plt.scatter(features[:, 0], features[:, 1], c=labels)
plt.show()