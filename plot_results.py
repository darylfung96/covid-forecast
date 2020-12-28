import os
import numpy as np
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt


is_plot = True
type = 'prediction'  # loss or prediction or mp

# dir = os.path.join('results', 'prediction', '6 months')


# folders = os.listdir(dir)
# items = os.listdir(os.path.join(dir, folders[0]))
#
# color_dict = {'results - matrix profile': 'g', 'results - no matrix profile': 'r',
#               'results - relative matrix profile': 'b'}
#
prediction_color_dict = {'matrix profile prediction': 'g', 'raw prediction': 'r', 'relative matrix profile prediction': 'b',
                         'actual data': 'k', 'raw and matrix profile prediction': 'c',
                         'raw and relative matrix profile prediction': 'm'}


def obtain_loss():
    dir = os.path.join('results')
    loss_types = os.listdir(dir)

    for loss_type in loss_types:
        loss_dir = os.path.join(dir, loss_type)
        types = os.listdir(loss_dir)

        for type in types:
            type_dir = os.path.join(loss_dir, type)
            for item in os.listdir(type_dir):
                if 'test' not in item:
                    continue

                loss = np.load(os.path.join(type_dir, item))
                print(f'{loss_type}---{type}---{item}')
                mean_loss = []
                for idx, current_loss in enumerate(np.mean(loss, 1).tolist()):
                    rounded_up_loss = round(current_loss, 5)
                    mean_loss.append(rounded_up_loss)
                    print(f'Rep-Holdout {idx}: {rounded_up_loss}')
                print(f'mean: {round(sum(mean_loss)/len(mean_loss), 5)}')
                print()
                print()


def plot_loss():

    dir = os.path.join('results', 'predictions')
    items = os.listdir(dir)

    for item in items:
        # plt.title(f'{item} 5 fold loss')

        if 'test' not in item:
            continue

        plots = []
        for folder in folders:
            filename = os.path.join(dir, folder, item)
            loss = np.load(filename)

            mean_loss = np.mean(loss, 0)
            current_plot = plt.plot(range(mean_loss.shape[0]), mean_loss, color_dict[folder])
            plots.append(current_plot)

        red_patch = mpatches.Patch(color='r', label='prediction based on raw observed data')
        blue_patch = mpatches.Patch(color='b', label='prediction based on relative matrix profile')
        green_patch = mpatches.Patch(color='g', label='prediction based on matrix profile')
        plt.legend(handles=[red_patch, blue_patch, green_patch])
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.show()


def plot_prediction():
    dir = os.path.join('results', 'predictions')
    folders = os.listdir(dir)
    items = os.listdir(os.path.join(dir, folders[0]))
    for item in items:
        # plt.title(f'{item} prediction')
        print(item)
        for folder in folders:
            filename = os.path.join(dir, folder, item)
            prediction = np.load(filename)

            plt.plot(range(prediction.shape[0]), prediction[:, 0], prediction_color_dict[folder])

        red_patch = mpatches.Patch(color='r', label='prediction based on raw observed data')
        blue_patch = mpatches.Patch(color='b', label='prediction based on relative matrix profile')
        green_patch = mpatches.Patch(color='g', label='prediction based on matrix profile')
        black_patch = mpatches.Patch(color='k', label='raw observed data')
        cyan_patch = mpatches.Patch(color='c', label='prediction based on raw and matrix')
        magenta_patch = mpatches.Patch(color='m', label='prediction based on raw and relative matrix')
        plt.legend(handles=[red_patch, blue_patch, green_patch, black_patch, cyan_patch, magenta_patch])
        plt.xlabel('days')
        plt.ylabel('values')
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
        blue_patch = mpatches.Patch(color='b', label='prediction\'s relative matrix profile')
        plt.legend(handles=[red_patch, blue_patch])
        plt.xlabel('days')
        plt.ylabel('matrix profile')
        plt.show()


if is_plot:
    if type == 'loss':
        plot_loss()
    elif type == 'prediction':
        plot_prediction()
    else:
        plot_mp()
else:
    obtain_loss()
