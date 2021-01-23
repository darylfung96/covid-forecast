from matrixprofile.matrixProfile import stomp
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def get_normalize_data(data_source):
    df = pd.read_csv('covid-forecast.csv')
    numpy_data = df[data_source].values

    numpy_data = np.expand_dims(numpy_data, 1)
    numpy_data = numpy_data.astype(np.float32)
    scaler = StandardScaler()
    normalized_numpy_data = scaler.fit_transform(numpy_data)

    return normalized_numpy_data, [data_source], scaler


def get_matrix_normalize_data(data_source, window_size=7, is_relative=False, is_global_pos=False, is_combined=False):
    normalized_numpy_data, columns, scaler = get_normalize_data(data_source)
    matrix_profile, profile_index = stomp(normalized_numpy_data.squeeze(1), window_size)

    # remove the first 6 real value so we can fit the same dimension as the matrix profile
    sliced_normalized_numpy_data = normalized_numpy_data[window_size-1:]
    relative_profile_index = profile_index - range(len(profile_index))

    matrix_scaler = StandardScaler()
    global_pos_scaler = StandardScaler()
    relative_scaler = StandardScaler()

    # relative
    relative_profile_index = np.expand_dims(relative_profile_index, 1)
    relative_profile_index = relative_scaler.fit_transform(relative_profile_index)

    # global pos
    profile_index = np.expand_dims(profile_index, 1)
    profile_index = global_pos_scaler.fit_transform(profile_index)

    # matrix profile
    matrix_profile = np.expand_dims(matrix_profile, 1)
    matrix_profile = matrix_scaler.fit_transform(matrix_profile)

    if is_combined:
        matrix_normalized_numpy_data = np.concatenate([sliced_normalized_numpy_data, matrix_profile,
                                                       relative_profile_index, profile_index], 1)
        columns += ['matrix_profile', 'relative_matrix_profile', 'global_matrix_profile']
        return matrix_normalized_numpy_data, columns, scaler, [matrix_scaler, global_pos_scaler,
                                                               relative_scaler], relative_profile_index

    if is_relative:
        matrix_normalized_numpy_data = np.concatenate([sliced_normalized_numpy_data,
                                                       relative_profile_index], 1)
        columns += ['relative_matrix_profile']
    elif is_global_pos:

        matrix_normalized_numpy_data = np.concatenate([sliced_normalized_numpy_data, profile_index], 1)
        columns += ['global_pos_matrix_profile']
    else:
        matrix_normalized_numpy_data = np.concatenate([sliced_normalized_numpy_data, matrix_profile], 1)
        columns += ['matrix_profile']

    return normalized_numpy_data, matrix_normalized_numpy_data, columns, scaler, [matrix_scaler, global_pos_scaler, relative_scaler], relative_profile_index


def get_mp_from_data(data, scalers, window_size=7, is_relative=False, is_combined=False):
    # scalers: [matrix_scaler, global_pos_scaler, relative_scaler]
    matrix_profile, profile_index = stomp(data.squeeze(1), window_size)

    sliced_data = data[window_size-1:]

    # relative
    relative_profile_index = profile_index - range(len(profile_index))
    relative_profile_index = np.expand_dims(relative_profile_index, 1)
    relative_profile_index = scalers[2].transform(relative_profile_index)

    # global pos
    profile_index = np.expand_dims(profile_index, 1)
    profile_index = scalers[1].transform(profile_index)

    # matrix profile
    matrix_profile = np.expand_dims(matrix_profile, 1)
    matrix_profile = scalers[0].transform(matrix_profile)

    if is_combined:
        matrix_normalized_numpy_data = np.concatenate([sliced_data, matrix_profile, relative_profile_index, profile_index],
                                                      1)
        return matrix_normalized_numpy_data

    if is_relative:
        matrix_normalized_numpy_data = np.concatenate([sliced_data,
                                                       relative_profile_index], 1)
    else:
        matrix_normalized_numpy_data = np.concatenate([sliced_data, matrix_profile], 1)

    return matrix_normalized_numpy_data, data


def get_new_mp_from_data(old_data, new_data, scalers, window_size=7, is_relative=False, is_combined=False):
    # scalers: [matrix_scaler, global_pos_scaler, relative_scaler]

    raw_values = old_data[:, 0:1]
    raw_values = np.concatenate([raw_values, new_data], axis=0)
    matrix_profile, profile_index = stomp(raw_values.squeeze(1), window_size)
    profile_index[profile_index == np.inf] = 0

    # relative
    relative_profile_index = profile_index - range(len(profile_index))
    relative_profile_index = np.expand_dims(relative_profile_index, 1)
    relative_profile_index = scalers[2].transform(relative_profile_index)

    # global pos
    profile_index = np.expand_dims(profile_index, 1)
    profile_index = scalers[1].transform(profile_index)

    # matrix profile
    matrix_profile = np.expand_dims(matrix_profile, 1)
    matrix_profile = scalers[0].transform(matrix_profile)

    if is_combined:
        new_relative_profile_index = np.concatenate(
            [old_data[:old_data[:, 1:].shape[0] - relative_profile_index.shape[0] + 1, 2:3],
             relative_profile_index], 0)
        new_profile_index = np.concatenate(
            [old_data[:old_data[:, 1:].shape[0] - profile_index.shape[0] + 1, 3:4],
             relative_profile_index], 0)
        new_matrix_profile = np.concatenate([old_data[:old_data[:, 1:].shape[0] - matrix_profile.shape[0] + 1, 1:2],
                                             matrix_profile], 0)
        raw_values = np.concatenate([raw_values, new_matrix_profile, new_relative_profile_index, new_profile_index], 1)
        return raw_values

    if is_relative:
        new_relative_profile_index = np.concatenate(
            [old_data[:old_data[:, 1:].shape[0] - relative_profile_index.shape[0] + 1, 1:2],
             relative_profile_index], 0)
        raw_values = np.concatenate([raw_values, new_relative_profile_index], axis=1)
    else:
        new_matrix_profile = np.concatenate([old_data[:old_data[:, 1:].shape[0] - matrix_profile.shape[0] + 1, 1:2],
                                             matrix_profile], 0)
        raw_values = np.concatenate([raw_values, new_matrix_profile], axis=1)

    return raw_values


def get_all_normalize_data():
    df = pd.read_csv('covid-forecast.csv')
    columns = np.array(df.columns[1:])
    values = df.values[:, 1:]

    scaler = StandardScaler()
    normalized_values = scaler.fit_transform(values)

    return normalized_values, columns, scaler


