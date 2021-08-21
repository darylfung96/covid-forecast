import os
import numpy as np
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler


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
    dir = os.path.join('results', 'loss')
    loss_types = os.listdir(dir)

    for loss_type in loss_types:
        loss_dir = os.path.join(dir, loss_type)
        types = os.listdir(loss_dir)

        outer_all_rmse_loss = []
        outer_all_mae_loss = []
        outer_all_mape_loss = []

        for type in types:
            type_dir = os.path.join(loss_dir, type)

            for item in os.listdir(type_dir):
                if 'test' not in item:
                    continue

                all_rmse_loss = []
                all_mae_loss = []
                all_mape_loss = []

                with open(os.path.join(type_dir, item), 'rb') as f:
                    losses = pickle.load(f)
                print(f'{loss_type}---{type}---{item}')

                all_rmse_loss.append(f'{type}_{item}_rmse')
                all_mae_loss.append(f'{type}_{item}_mae')
                all_mape_loss.append(f'{type}_{item}_mape')

                for idx, loss in enumerate(losses):
                    rmse_loss = round(np.min(loss['rmse']), 3)
                    mae_loss = round(np.min(loss['mae']), 3)
                    mape_loss = round(np.min(loss['mape']), 3)

                    all_rmse_loss.append(rmse_loss)
                    all_mae_loss.append(mae_loss)
                    all_mape_loss.append(mape_loss)

                    print(f'rep holdout {idx} rmse: {rmse_loss}')
                    print(f'rep holdout {idx} mae: {mae_loss}')
                    print(f'rep holdout {idx} mape: {mape_loss}')

                mean_rmse_loss = np.mean(all_rmse_loss[1:])
                mean_mae_loss = np.mean(all_mae_loss[1:])
                mean_mape_loss = np.mean(all_mape_loss[1:])

                # write to csv for writing to table temporarily
                all_rmse_loss.append(mean_rmse_loss)
                all_mae_loss.append(mean_mae_loss)
                all_mape_loss.append(mean_mape_loss)
                print(f'average rmse: {mean_rmse_loss}')
                print(f'average mae: {mean_mae_loss}')
                print(f'average mape: {mean_mape_loss}')

                outer_all_rmse_loss.append(all_rmse_loss)
                outer_all_mae_loss.append(all_mae_loss)
                outer_all_mape_loss.append(all_mape_loss)

        all_total_loss = np.hstack([outer_all_rmse_loss, outer_all_mae_loss, outer_all_mape_loss])
        pd.DataFrame(all_total_loss).to_csv('temp.csv')


                # for key in loss.keys():
                #     type_loss = loss[key]
                #     for idx, current_loss in enumerate(np.min(type_loss, 1).tolist()):
                #         rounded_up_loss = round(current_loss, 3)
                #         mean_loss.append(rounded_up_loss)
                #         # print(f'Rep-Holdout {idx}: {rounded_up_loss}')
                #         print(f'loss type: {type_loss}')
                #         print(f'{rounded_up_loss}')
                #     print(f'{round(sum(mean_loss)/len(mean_loss), 3)}')
                #     print()
                #     print()


def obtain_prediction():
    dir = os.path.join('results', 'predictions')
    folders = os.listdir(dir)

    for folder in folders:

        # if folder not in focused_prediction_plots:
        #     continue

        df_dict = {'Date': pd.date_range(start="2020-03-01", end="2021-10-31")}
        for item in ['confirmed cases', 'death cases',
                 'hospital admission-adj (percentage of new admissions that are covid)']:
        # for item in ['Normalized.Admission_holdout', 'Normalized.Confirmed_holdout', 'Normalized.Death_holdout']:
            nname = f'{item}'
            filename = os.path.join(dir, folder, f'{nname}.npy')

            # if not os.path.exists(filename):
            #     continue
            prediction = np.load(filename)[:, 0]

            # if 'confirmed' in item.lower():
            #     nname = 'Confirmed.cases'
            # elif 'death' in item.lower():
            #     nname = 'Death.cases'
            # else:
            #     nname = 'Admission.cases'

            length = df_dict['Date'].shape[0]
            curr_length = prediction.shape[0]
            if length - curr_length > 0:
                prediction = np.concatenate([prediction, prediction[-(length-curr_length):]])
            elif length - curr_length < 0:
                prediction = prediction[:length]

            df_dict[nname] = prediction

        df = pd.DataFrame(df_dict)

        save_dir = os.path.join('results', 'csv')
        os.makedirs(save_dir, exist_ok=True)
        save_filename = os.path.join(save_dir, f'{folder}.csv')
        df.to_csv(save_filename, index=False)


def obtain_prediction_folds():
    dir = os.path.join('results', 'predictions')
    folders = os.listdir(dir)

    actual_df = pd.read_csv('covid-forecast.csv')
    all_values = np.hstack([actual_df.values[:, 3:4].repeat(6, axis=1), actual_df.values[:, 5:6].repeat(6, axis=1),
                            actual_df.values[:, 2:3].repeat(6, axis=1)])
    scaler = StandardScaler()
    scaler.fit(all_values)
    for folder in folders:

        # if folder not in focused_prediction_plots:
        #     continue

        date_range = pd.date_range(start="2020-03-01", end="2021-10-31")
        df_dict = {'Date': date_range}
        for item in ['confirmed cases', 'death cases',
                 'hospital admission-adj (percentage of new admissions that are covid)']:
        # for item in ['Normalized.Admission_holdout', 'Normalized.Confirmed_holdout', 'Normalized.Death_holdout']:
            for i in range(1, 6):
                nname = f'{item}_holdout_{i}'
                filename = os.path.join(dir, folder, f'{nname}.npy')

                # if not os.path.exists(filename):
                #     continue
                prediction = np.load(filename)[:, 0]

                # if 'confirmed' in item.lower():
                #     nname = 'Confirmed.cases'
                # elif 'death' in item.lower():
                #     nname = 'Death.cases'
                # else:
                #     nname = 'Admission.cases'

                length = date_range.shape[0]
                curr_length = prediction.shape[0]
                if length - curr_length > 0:
                    prediction = np.concatenate([prediction, prediction[-(length-curr_length):]])
                elif length - curr_length < 0:
                    prediction = prediction[:length]

                df_dict[nname] = prediction

            # get the average
            average_value = None
            num = 0
            for i in range(1, 6):
                current_key = f'{item}_holdout_{i}'

                if average_value is None:
                    average_value = df_dict[current_key]
                else:
                    average_value += df_dict[current_key]
                num += 1
            average_value /= num
            df_dict[f'{item}_average'] = average_value

        df = pd.DataFrame(df_dict)
        actual_prediction_df = scaler.inverse_transform(df.values[:, 1:])
        actual_prediction_df = np.hstack([np.expand_dims(date_range.values, 1).astype(np.str), actual_prediction_df])
        actual_df = pd.DataFrame(actual_prediction_df, columns=df.columns)

        save_dir = os.path.join('results', 'csv')
        os.makedirs(save_dir, exist_ok=True)
        save_filename_normalized = os.path.join(save_dir, f'{folder}_normalized.csv')
        save_filename_actual = os.path.join(save_dir, f'{folder}_actual.csv')
        df.to_csv(save_filename_normalized, index=False)
        actual_df.to_csv(save_filename_actual, index=False)


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
        obtain_prediction_folds()
