import os
import numpy as np
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt


type = 'mp'  # loss or prediction or mp

dir = os.path.join('results', 'plots', 'prediction')

folders = os.listdir(dir)
items = os.listdir(os.path.join(dir, folders[0]))

color_dict = {'results - matrix profile': 'g', 'results - no matrix profile': 'r',
              'results - relative matrix profile': 'b'}

prediction_color_dict = {'matrix profile': 'g', 'no matrix': 'r', 'relative matrix profile': 'b', 'actual data': 'k' }


def plot_loss():
    for item in items:
        plt.title(f'{item} 5 fold loss')

        if 'test' not in item:
            continue

        plots = []
        for folder in folders:
            filename = os.path.join(dir, folder, item)
            loss = np.load(filename)

            mean_loss = np.mean(loss, 0)
            current_plot = plt.plot(range(mean_loss.shape[0]), mean_loss, color_dict[folder])
            plots.append(current_plot)

        red_patch = mpatches.Patch(color='r', label='no matrix profile')
        blue_patch = mpatches.Patch(color='b', label='relative matrix profile')
        green_patch = mpatches.Patch(color='g', label='matrix profile')
        plt.legend(handles=[red_patch, blue_patch, green_patch])
        plt.show()


def plot_prediction():
    for item in items:
        plt.title(f'{item} prediction')

        for folder in folders:
            filename = os.path.join(dir, folder, item)
            prediction = np.load(filename)

            plt.plot(range(prediction.shape[0]), prediction[:, 0], prediction_color_dict[folder])

        red_patch = mpatches.Patch(color='r', label='no matrix profile')
        blue_patch = mpatches.Patch(color='b', label='relative matrix profile')
        green_patch = mpatches.Patch(color='g', label='matrix profile')
        plt.legend(handles=[red_patch, blue_patch, green_patch])
        plt.show()


def plot_mp():
    for item in items:
        plt.title(f'{item} prediction')

        for folder in folders:
            if folder == 'no matrix' or folder == 'matrix profile':
                continue

            filename = os.path.join(dir, folder, item)
            prediction = np.load(filename)

            plt.plot(range(prediction.shape[0]), prediction[:, 1], prediction_color_dict[folder])

        red_patch = mpatches.Patch(color='k', label='original relative matrix profile')
        blue_patch = mpatches.Patch(color='b', label='predicted relative matrix profile')
        plt.legend(handles=[red_patch, blue_patch])
        plt.show()



if type == 'loss':
    plot_loss()
elif type == 'prediction':
    plot_prediction()
else:
    plot_mp()
