import os
import numpy as np
import matplotlib.pyplot as plt

for item in os.listdir('results - no matrix profile'):
    values = np.load(os.path.join('results - no matrix profile', item))
    shape_values = values.shape
    mean_values = np.mean(values, 0)
    plt.title(item)
    plt.plot(range(mean_values.shape[0]), mean_values, 'r')
    plt.show()
