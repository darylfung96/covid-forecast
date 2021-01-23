import os
import numpy as np
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd


is_plot = False
type = 'prediction'  # loss or prediction or mp
loss_type = 'rmse'
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

focused_prediction_plots = ['actual data', f'{loss_type} raw  ', f'{loss_type} raw  attention',
                            f'{loss_type} raw matrix attention',
                            f'{loss_type} raw relative attention', 'arima']
# focused_prediction_plots = [f'{loss_type} matrix attention']


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
                for idx, current_loss in enumerate(np.min(loss, 1).tolist()):
                    rounded_up_loss = round(current_loss, 3)
                    mean_loss.append(rounded_up_loss)
                    # print(f'Rep-Holdout {idx}: {rounded_up_loss}')
                    print(f'{rounded_up_loss}')
                print(f'{round(sum(mean_loss)/len(mean_loss), 3)}')
                print()
                print()


def obtain_prediction():
    dir = os.path.join('results', 'predictions')
    folders = os.listdir(dir)

    for folder in folders:

        # if folder not in focused_prediction_plots:
        #     continue

        df_dict = {'Date': pd.date_range(start="2020-03-01", end="2021-05-31")}
        for item in ['confirmed cases.npy', 'death cases.npy',
                 'hospital admission-adj (percentage of new admissions that are covid).npy']:
            filename = os.path.join(dir, folder, item)

            if not os.path.exists(filename):
                continue
            prediction = np.load(filename)[:, 0]

            if 'confirmed' in item.lower():
                nname = 'Confirmed.cases'
            elif 'death' in item.lower():
                nname = 'Death.cases'
            else:
                nname = 'Admission.cases'

            length = df_dict['Date'].shape[0]
            curr_length = prediction.shape[0]
            if length - curr_length > 0:
                prediction = np.concatenate([prediction, prediction[-(length-curr_length):]])

            df_dict[nname] = prediction

        df = pd.DataFrame(df_dict)

        save_dir = os.path.join('results', 'csv')
        os.makedirs(save_dir, exist_ok=True)
        save_filename = os.path.join(save_dir, f'{folder}.csv')
        df.to_csv(save_filename, index=False)









def plot_loss():

    dir = os.path.join('results', f'results - {loss_type}')
    folders = os.listdir(dir)

    items = os.listdir(os.path.join(dir, folders[0]))

    for item in items:
        if 'test' not in item:
            continue
        # plt.title(f'{item} 5 fold loss')

        all_color_patches = []
        for folder in folders:
            filename = os.path.join(dir, folder, item)
            loss = np.load(filename)

            mean_loss = np.mean(loss, 0)
            current_plot = plt.plot(range(mean_loss.shape[0]), mean_loss)
            color_id = current_plot[0].get_color()
            all_color_patches.append(mpatches.Patch(color=color_id, label=f'prediction based on {folder}'))

        plt.legend(handles=all_color_patches)
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.show()


def plot_prediction():
    dir = os.path.join('results', 'predictions')
    folders = os.listdir(dir)
    items = os.listdir(os.path.join(dir, folders[0]))

    os.makedirs(f'plots/predictions/{loss_type}', exist_ok=True)
    plt.rcParams["figure.figsize"] = [16, 9]

    y_labels = ['number of daily confirmed cases', 'number of daily death cases',
                'percentage of new hospital covid admission']
    index = 0
    for item in ['confirmed cases.npy', 'death cases.npy',
                 'hospital admission-adj (percentage of new admissions that are covid).npy']:
        # plt.title(f'{item} prediction')
        print(item)
        all_color_patches = []

        for folder in folders:

            # if 'actual data' not in folder and folder != 'arima':
            #     if loss_type not in folder:
            #         continue
            if folder not in focused_prediction_plots:
                continue

            if folder == 'actual data':
                label_name = 'actual data'
                filename = os.path.join(dir, folder, item)
                prediction = np.load(filename)
            else:
                label_name = f'{folder}'.replace('rmse', '')
                filename = os.path.join(dir, folder, item)
                prediction = np.load(filename)

            current_plot = plt.plot(range(prediction.shape[0]), prediction[:, 0])
            color_id = current_plot[0].get_color()
            all_color_patches.append(mpatches.Patch(color=color_id, label=label_name))



        params = {'legend.fontsize': 8,
                  'legend.handlelength': 1}
        plt.rcParams.update(params)
        plt.rcParams["figure.figsize"] = [16, 9]
        plt.legend(handles=all_color_patches)
        plt.xlabel('days')
        plt.ylabel(f'{y_labels[index]}')
        plt.xticks([0, 257, 312, 457], ['March 1 2020', 'Dec 13 2020', 'Jan 7 2020', 'May 31 2020'])

        plt.savefig(f'plots/predictions/{loss_type}/{item.split(".")[0]}')
        plt.clf()

        # plt.show()
        index += 1


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
    if type == 'loss':
        obtain_loss()
    elif type== 'prediction':
        obtain_prediction()
