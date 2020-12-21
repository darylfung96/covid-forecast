from matrixprofile.matrixProfile import stomp
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def get_normalize_data(data_source):
    df = pd.read_csv('covid-forecast.csv')
    numpy_data = df[data_source].values

    numpy_data = np.expand_dims(numpy_data, 1)
    scaler = StandardScaler()
    normalized_numpy_data = scaler.fit_transform(numpy_data)

    return normalized_numpy_data, [data_source], scaler


def get_matrix_normalize_data(data_source, window_size=7, is_relative=False):
    normalized_numpy_data, columns, scaler = get_normalize_data(data_source)
    matrix_profile, profile_index = stomp(normalized_numpy_data.squeeze(1), window_size)
    matrix_profile = np.expand_dims(matrix_profile, 1)

    # remove the first 6 real value so we can fit the same dimension as the matrix profile
    normalized_numpy_data = normalized_numpy_data[window_size-1:]
    relative_profile_index = profile_index - range(len(profile_index))

    relative_scaler = StandardScaler()

    if is_relative:
        relative_profile_index = np.expand_dims(relative_profile_index, 1)
        relative_profile_index = relative_scaler.fit_transform(relative_profile_index)
        matrix_normalized_numpy_data = np.concatenate([normalized_numpy_data,
                                                       relative_profile_index], 1)
    else:
        matrix_normalized_numpy_data = np.concatenate([normalized_numpy_data, matrix_profile], 1)

    columns += ['matrix_profile']
    return matrix_normalized_numpy_data, columns, scaler, relative_scaler, relative_profile_index


def get_mp_from_data(data, relative_scaler, window_size=7, is_relative=False):
    matrix_profile, profile_index = stomp(data.squeeze(1), window_size)

    data = data[window_size-1:]

    if is_relative:
        relative_profile_index = profile_index - range(len(profile_index))
        relative_profile_index = np.expand_dims(relative_profile_index, 1)
        relative_profile_index = relative_scaler.transform(relative_profile_index)
        matrix_normalized_numpy_data = np.concatenate([data,
                                                       relative_profile_index], 1)
    else:
        matrix_normalized_numpy_data = np.concatenate([data, np.expand_dims(matrix_profile, 1)], 1)

    return matrix_normalized_numpy_data


def get_new_mp_from_data(old_data, new_data, relative_scaler, window_size=7, is_relative=False):
    raw_values = old_data[:, 0:1]
    raw_values = np.concatenate([raw_values, new_data], axis=0)
    matrix_profile, profile_index = stomp(raw_values.squeeze(1), window_size)

    if is_relative:
        relative_profile_index = profile_index - range(len(profile_index))
        relative_profile_index = np.expand_dims(relative_profile_index, 1)
        relative_profile_index = relative_scaler.transform(relative_profile_index)
        new_relative_profile_index = np.concatenate(
            [old_data[:old_data[:, 1:].shape[0] - relative_profile_index.shape[0] + 1, 1:],
             relative_profile_index], 0)
        raw_values = np.concatenate([raw_values, new_relative_profile_index], axis=1)
    else:
        new_matrix_profile = np.concatenate([old_data[:old_data[:, 1:].shape[0] - matrix_profile.shape[0] + 1, 1:],
                                             np.expand_dims(matrix_profile, 1)], 0)
        raw_values = np.concatenate([raw_values, new_matrix_profile], axis=1)

    return raw_values


def get_all_normalize_data():
    df = pd.read_csv('covid-forecast.csv')
    columns = np.array(df.columns[1:])
    values = df.values[:, 1:]

    scaler = StandardScaler()
    normalized_values = scaler.fit_transform(values)

    return normalized_values, columns, scaler


