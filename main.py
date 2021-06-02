import numpy as np
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import random
import os
import pickle
import optuna
import json
from optuna.integration import PyTorchLightningPruningCallback


from k_fold import RepHoldout
# from data_preprocessing import get_matrix_normalize_data, get_normalize_data, get_mp_from_data, get_new_mp_from_data
from data_preprocessing import get_matrix_already_normalize_data, get_already_normalize_data, get_mp_from_data, get_new_mp_from_data
from lstm import LightningModel, LightningModelAttention
from dataset import ForecastDataset, TestingForecastDataset


batch_size = 8
seq_length = 12
seed = 1000
window_size = 7
# is_matrix = True
# is_relative = False
# is_combined = False
# only_mp_features = False
# model_type = 'attention'
device = 'cpu'

# [raw, raw relative attention, raw matrix attention, raw attention]
is_matrix_list = [False, True, True, False]
is_relative_list = [False, True, False, False]
is_combined_list = [False, False, False, False]
is_only_mp_features = [False, False, False, False]
is_model_type = ['lstm', 'attention', 'attention', 'attention']

for i in range(len(is_matrix_list)):
    is_matrix = is_matrix_list[i]
    is_relative = is_relative_list[i]
    is_combined = is_combined_list[i]
    only_mp_features = is_only_mp_features[i]
    model_type = is_model_type[i]

    ### setting model name, loss, model choice ###
    if not is_combined:
        if is_matrix:
            if is_relative:
                matrix_str = 'relative'
            else:
                matrix_str = 'matrix'
        else:
            matrix_str = ''
    else:
        matrix_str = 'matrix relative'

    attention_str = 'attention' if model_type == 'attention' else ''
    raw_str = '' if only_mp_features else 'raw'

    # all_data_sources = [
    #     'hospital admission-adj (percentage of new admissions that are covid)', 'confirmed cases', 'death cases'
    # ]
    all_data_sources = [
        'Normalized.Admission', 'Normalized.Confirmed', 'Normalized.Death'
    ]
    model_dict = {'lstm': LightningModel, 'attention': LightningModelAttention}
    k_fold = False  # if not k fold, then train the whole data and do prediction
    prediction_only = True
    loss = 'rmse'

    model_name = f'{loss} {raw_str} {matrix_str} {attention_str}'
    ### end of setting model name ###

    def make_predictions(data_source, data_source_normalized_data, normalized_data, scaler, all_other_scalers, lightningModel, current_rep_holdout):
        lstm_model = lightningModel.lstm_model.to(device)
        lstm_model.eval()
        hidden_states, cell_states = lstm_model.init_hiddenlstm_state()

        # train the whole data
        # if there is mp we can the first index (real value) and not the matrix profile first
        predictions = normalized_data[:seq_length, 0:1]
        for i in range(0, normalized_data.shape[0] - seq_length):

            backward_flow = normalized_data[i:i + seq_length].shape[0] - seq_length
            input_data = normalized_data[i + backward_flow:i + seq_length]
            if only_mp_features:
                current_input_data = input_data[:, 1:]
            else:
                current_input_data = input_data
            tensor_data = torch.Tensor(np.expand_dims(current_input_data, 0)).to(device)

            if model_type == 'attention':
                if is_matrix:
                    if only_mp_features:
                        tensor_data = torch.Tensor(np.expand_dims(input_data[:, 1:], 0)).to(device)
                        tensor_memory = torch.Tensor(np.expand_dims(input_data[:, 1:], 0)).to(device)
                    else:
                        tensor_data = torch.Tensor(np.expand_dims(input_data, 0)).to(device)
                        tensor_memory = torch.Tensor(np.expand_dims(input_data, 0)).to(device)
                        # tensor_data = torch.Tensor(np.expand_dims(input_data[:, 1:], 0)).to(device)
                        # tensor_memory = torch.Tensor(np.expand_dims(input_data[:, 0:1], 0)).to(device)
                else:
                    tensor_data = torch.Tensor(np.expand_dims(input_data, 0)).to(device)
                    tensor_memory = torch.Tensor(np.expand_dims(input_data, 0)).to(device)
                outputs, hidden_states, cell_states = lstm_model(tensor_data, tensor_memory,
                                                                 hidden_states, cell_states)
            else:
                outputs, hidden_states, cell_states = lstm_model(tensor_data,
                                                                 hidden_states, cell_states)

            next_prediction = outputs.cpu().detach().numpy()[0][-1:]

            if predictions is None:
                predictions = next_prediction
            else:
                predictions = np.concatenate([predictions, next_prediction], axis=0)

        if is_matrix:
            predictions, ori_data = get_mp_from_data(predictions, all_other_scalers, window_size=window_size, is_relative=is_relative, is_combined=is_combined)

        for i in range(152):
            predicted_data = predictions[-seq_length:]
            if only_mp_features:
                current_predicted_data = predicted_data[:, 1:]
            else:
                current_predicted_data = predicted_data
            tensor_data = torch.Tensor(np.expand_dims(current_predicted_data, 0)).to(device)

            if model_type == 'attention':
                if is_matrix:
                    if only_mp_features:
                        tensor_data = torch.Tensor(np.expand_dims(predicted_data[:, 1:], 0)).to(device)
                        tensor_memory = torch.Tensor(np.expand_dims(predicted_data[:, 1:], 0)).to(device)
                    else:
                        tensor_data = torch.Tensor(np.expand_dims(predicted_data, 0)).to(device)
                        tensor_memory = torch.Tensor(np.expand_dims(predicted_data, 0)).to(device)
                        # tensor_data = torch.Tensor(np.expand_dims(input_data[:, 1:], 0)).to(device)
                        # tensor_memory = torch.Tensor(np.expand_dims(input_data[:, 0:1], 0)).to(device)
                else:
                    tensor_data = torch.Tensor(np.expand_dims(predicted_data, 0)).to(device)
                    tensor_memory = torch.Tensor(np.expand_dims(predicted_data, 0)).to(device)
                outputs, hidden_states, cell_states = lstm_model(tensor_data, tensor_memory, hidden_states, cell_states)
            else:
                outputs, hidden_states, cell_states = lstm_model(tensor_data, hidden_states, cell_states)

            next_prediction = outputs.cpu().detach().numpy()[0][-1:]

            if is_matrix:
                predictions = get_new_mp_from_data(predictions, next_prediction, all_other_scalers, window_size=window_size, is_relative=is_relative,
                                                   is_combined=is_combined)
            else:
                predictions = np.concatenate([predictions, next_prediction], 0)

        # for i in range(normalized_data.shape[1]):
        #     plt.title(data_source)
        #     plt.plot(range(normalized_data.shape[0]), normalized_data[:, i], 'b')
        #     plt.plot(range(len(predictions)), predictions[:, i], 'g')
        #     plt.show()

        # rejoin the removed first few value as a result from matrix profile
        predictions = predictions[:, 0:1]
        predictions = np.concatenate([data_source_normalized_data[:2*window_size-2, 0:1], predictions])

        dir = os.path.join('results', 'predictions')
        current_dir = os.path.join(dir, model_name)

        os.makedirs(current_dir, exist_ok=True)
        # np.save(os.path.join(current_dir, f'{data_source}_actual_data'), scaler.inverse_transform(normalized_data))

        if scaler is not None:
            predictions = scaler.inverse_transform(predictions)

        if current_rep_holdout is None:
            np.save(os.path.join(current_dir, f'{data_source}'), predictions)
        else:
            np.save(os.path.join(current_dir, f'{data_source}_holdout_{current_rep_holdout}'), predictions)


    # objective for hyperparameter optimization
    def k_fold_training(trial, data_source, data_source_normalized_data, normalized_data, scaler, all_other_scalers):

        ### trial objective ###
        if type(trial) is not dict:
            dropout = trial.suggest_uniform('dropout', 0.0, 1.0)
            hidden_dim = trial.suggest_categorical('hidden_dim',
                                                   [8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128])
            ff_dim = trial.suggest_categorical('ff_dim', [8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128])
            learning_rate = trial.suggest_loguniform('learning_rate', 1e-4, 1e-2)
            weight_decay = trial.suggest_loguniform('weight_decay', 1e-5, 1e-2)
        else:
            dropout = trial['dropout']
            hidden_dim = trial['hidden_dim']
            ff_dim = trial['ff_dim']
            learning_rate = trial['learning_rate']
            weight_decay = trial['weight_decay']

        # create k fold split = 5
        k_fold_normalized_data = RepHoldout(n_splits=5).split(normalized_data)
        all_train_loss = []
        all_test_loss = []
        ### end of trial objective ###

        current_rep_holdout = 0
        for train_index, test_index in k_fold_normalized_data:
            current_rep_holdout += 1
            current_model_name = f'{model_name}/{data_source}_holdout_{current_rep_holdout}'
            np.random.seed(seed)
            torch.manual_seed(seed)
            random.seed(seed)
            torch.cuda.manual_seed_all(seed)

            training_normalized_data = normalized_data[train_index]
            testing_normalized_data = normalized_data[test_index]


            # create datasets
            train_dataset = ForecastDataset(training_normalized_data, seq_length, only_mp_features=only_mp_features)
            test_dataset = TestingForecastDataset(testing_normalized_data, seq_length, only_mp_features=only_mp_features)
            train_data_loader = DataLoader(train_dataset, batch_size=batch_size, drop_last=False)
            test_data_loader = DataLoader(test_dataset, batch_size=1, drop_last=False)

            get_new_mp_from_data_func = lambda predictions, next_prediction: get_new_mp_from_data(predictions, next_prediction, all_other_scalers, window_size=window_size,
                                                                                                  is_relative=is_relative,
                                               is_combined=is_combined)
            lightningModel = model_dict[model_type](training_normalized_data, get_new_mp_from_data_func, scaler, batch_size, seq_length, input_dim=train_dataset[0][0].shape[1],
                                            teacher_forcing=True, loss=loss, is_matrix=is_matrix,
                                                    only_mp_features=only_mp_features, model_name=current_model_name,
                                                    hidden_dim=hidden_dim, ff_dim=ff_dim, weight_decay=weight_decay,
                                                    dropout=dropout, learning_rate=learning_rate)

            if device == 'cuda':
                gpus = 1
            else:
                gpus = 0
            trainer = pl.Trainer(max_epochs=200, gpus=gpus)
            trainer.fit(lightningModel, train_data_loader, test_data_loader)
            all_train_loss.append(lightningModel.all_train_loss)
            all_test_loss.append(lightningModel.all_test_loss)

            make_predictions(data_source, data_source_normalized_data, normalized_data, scaler, all_other_scalers, lightningModel, current_rep_holdout)

        # objective = np.mean(np.min(np.array(all_test_loss), 1))
        # return objective

        dir = os.path.join('results', f'results - {loss}')
        current_dir = os.path.join(dir, f'{raw_str} {matrix_str} {attention_str}')
        os.makedirs(current_dir, exist_ok=True)

        with open(os.path.join(current_dir,f'train_{data_source}'), 'wb') as f:
           pickle.dump(all_train_loss, f, pickle.HIGHEST_PROTOCOL)

        with open(os.path.join(current_dir, f'test_{data_source}'), 'wb') as f:
            pickle.dump(all_test_loss, f, pickle.HIGHEST_PROTOCOL)


    def whole_data_train(trial, data_source, data_source_normalized_data, normalized_data, scaler, all_other_scalers):
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)
        torch.cuda.manual_seed_all(seed)

        dropout = trial['dropout']
        hidden_dim = trial['hidden_dim']
        ff_dim = trial['ff_dim']
        learning_rate = trial['learning_rate']
        weight_decay = trial['weight_decay']

        # create datasets
        train_dataset = ForecastDataset(normalized_data, seq_length, only_mp_features)
        train_data_loader = DataLoader(train_dataset, batch_size=batch_size, drop_last=False)

        current_model_name = f'{model_name}/{data_source}_all'

        get_new_mp_from_data_func = lambda predictions, next_prediction: get_new_mp_from_data(predictions, next_prediction,
                                                                                              all_other_scalers,
                                                                                              window_size=window_size,
                                                                                              is_relative=is_relative,
                                                                                              is_combined=is_combined)

        lightningModel = model_dict[model_type](normalized_data, get_new_mp_from_data_func, scaler, batch_size, seq_length, input_dim=train_dataset[0][0].shape[1],
                                            teacher_forcing=True, loss=loss, is_matrix=is_matrix,
                                                    only_mp_features=only_mp_features, model_name=current_model_name,
                                                    hidden_dim=hidden_dim, ff_dim=ff_dim, weight_decay=weight_decay,
                                                    dropout=dropout, learning_rate=learning_rate)
        trainer = pl.Trainer(max_epochs=200)
        trainer.fit(lightningModel, train_data_loader)
        make_predictions(data_source, data_source_normalized_data, normalized_data, scaler, all_other_scalers, lightningModel, None)


    def main():
        for data_source in all_data_sources:
            if is_matrix:
                data_source_normalized_data, normalized_data, columns, scaler, all_other_scalers, _ = get_matrix_already_normalize_data(data_source,
                                                                                                                                window_size=window_size,
                                                                                               is_relative=is_relative,
                                                                                                 is_combined=is_combined)
            else:
                normalized_data, columns, scaler = get_already_normalize_data(data_source)
                all_other_scalers = None
                data_source_normalized_data = normalized_data

            trial = {'dropout': 0.2, 'hidden_dim': 8, 'ff_dim': 8, 'learning_rate': 0.0004, 'weight_decay': 1e-5}
            if k_fold:
                # trial = {'dropout': 0.045, 'hidden_dim': 64, 'ff_dim': 80, 'learning_rate': 0.001, 'weight_decay': 0.001}

                k_fold_training(trial, data_source, data_source_normalized_data, normalized_data, scaler, all_other_scalers)

            elif prediction_only:
                get_new_mp_from_data_func = lambda predictions, next_prediction: get_new_mp_from_data(predictions,
                                                                                                      next_prediction,
                                                                                                      all_other_scalers,
                                                                                                      window_size=window_size,
                                                                                                      is_relative=is_relative,
                                                                                                      is_combined=is_combined)

                dataset = ForecastDataset(normalized_data, seq_length, only_mp_features=only_mp_features)
                for current_rep_holdout in range(1, 6):
                    current_model_name = f'{model_name}/{data_source}_holdout_{current_rep_holdout}'

                    lightningModel = model_dict[model_type](normalized_data, get_new_mp_from_data_func, scaler,
                                                            batch_size, seq_length, input_dim=dataset[0][0].shape[1],
                                                            teacher_forcing=True, loss=loss, is_matrix=is_matrix,
                                                            only_mp_features=only_mp_features,
                                                            model_name=current_model_name,
                                                            hidden_dim=trial['hidden_dim'], ff_dim=trial['ff_dim'], weight_decay=trial['weight_decay'],
                                                            dropout=trial['dropout'], learning_rate=trial['learning_rate'])
                    lightningModel.load(current_model_name)
                    make_predictions(data_source, data_source_normalized_data, normalized_data, scaler, all_other_scalers, lightningModel, current_rep_holdout)
            else:
                whole_data_train(trial, data_source, data_source_normalized_data, normalized_data, scaler, all_other_scalers)

    main()
