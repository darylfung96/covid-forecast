import numpy as np
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import random
import os
import pickle
from ray.tune.integration.pytorch_lightning import TuneReportCallback
import ray.tune as tune
from ray.tune import CLIReporter
from functools import partial


from k_fold import RepHoldout
from data_preprocessing import get_matrix_normalize_data, get_normalize_data, get_mp_from_data, get_new_mp_from_data
# from data_preprocessing import get_matrix_already_normalize_data, get_already_normalize_data, get_mp_from_data, get_new_mp_from_data
from lstm import LightningModel, LightningModelAttention, LightningModelCNNLSTM
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
is_matrix_list = [True]
is_relative_list = [False]
is_combined_list = [False]
is_only_mp_features = [False]
is_model_type = ['attention']  # ['lstm', 'attention', 'attention', 'attention']

hidden_dim_list = [32]
dropout_list = [0.85]
ff_dim_list = [8]

# is_matrix_list = [True]
# is_relative_list = [False]
# is_combined_list = [False]
# is_only_mp_features = [False]
# is_model_type = ['attention']

# k_fold = False  # if not k fold, then train the whole data and do prediction #TODO change back to True
# find_optimal_param = True
# prediction_only = False
run_type = 'k_fold'

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

    raw_str = '' if only_mp_features else 'raw'

    all_data_sources = [
        'hospital admission-adj (percentage of new admissions that are covid)', 'confirmed cases', 'death cases'
    ]
    # all_data_sources = [
    #     'Normalized.Admission', 'Normalized.Confirmed', 'Normalized.Death'
    # ]
    model_dict = {'lstm': LightningModel, 'attention': LightningModelAttention, 'cnnlstm': LightningModelCNNLSTM}
    loss = 'rmse'

    model_name = f'{loss} {raw_str} {matrix_str} {model_type}'
    ### end of setting model name ###

    def make_predictions(data_source, data_source_normalized_data, normalized_data, scaler, all_other_scalers, lightningModel, current_rep_holdout):
        lightningModel.load_best_state_dict()
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

        for i in range(88):
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
            test_data_loader = DataLoader(test_dataset, batch_size=1, drop_last=False, shuffle=False)

            get_new_mp_from_data_func = lambda predictions, next_prediction: get_new_mp_from_data(predictions, next_prediction, all_other_scalers, window_size=window_size,
                                                                                                  is_relative=is_relative,
                                               is_combined=is_combined)
            lightningModel = model_dict[model_type](training_normalized_data, get_new_mp_from_data_func, scaler, batch_size, seq_length, input_dim=train_dataset[0][0].shape[1],
                                            teacher_forcing=True, loss=loss, is_matrix=is_matrix,
                                                    only_mp_features=only_mp_features, model_name=current_model_name,
                                                     weight_decay=weight_decay,
                                                     learning_rate=learning_rate, config=trial)

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

        dir = os.path.join('results', 'loss', f'results - {loss}')
        current_dir = os.path.join(dir, f'{raw_str} {matrix_str} {model_type}')
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


    def run_prediction_only(trial, data_source, data_source_normalized_data, normalized_data, scaler, all_other_scalers):
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
                                                    hidden_dim=trial['hidden_dim'], ff_dim=trial['ff_dim'],
                                                    weight_decay=trial['weight_decay'],
                                                    dropout=trial['dropout'], learning_rate=trial['learning_rate'])
            lightningModel.load(current_model_name)
            make_predictions(data_source, data_source_normalized_data, normalized_data, scaler, all_other_scalers,
                             lightningModel, current_rep_holdout)



    def find_optimal_parameters(trial, data_source, data_source_normalized_data, normalized_data, scaler, all_other_scalers):
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
        k_fold_normalized_data = RepHoldout(n_splits=1).split(normalized_data)
        training_normalized_data = normalized_data[k_fold_normalized_data[0][0]]
        testing_normalized_data = normalized_data[k_fold_normalized_data[0][1]]

        current_model_name = f'{model_name}/{data_source}_all'

        # create datasets
        train_dataset = ForecastDataset(training_normalized_data, seq_length, only_mp_features=only_mp_features)
        test_dataset = TestingForecastDataset(testing_normalized_data, seq_length, only_mp_features=only_mp_features)
        train_data_loader = DataLoader(train_dataset, batch_size=batch_size, drop_last=False)
        test_data_loader = DataLoader(test_dataset, batch_size=1, drop_last=False)

        get_new_mp_from_data_func = lambda predictions, next_prediction: get_new_mp_from_data(predictions,
                                                                                              next_prediction,
                                                                                              all_other_scalers,
                                                                                              window_size=window_size,
                                                                                              is_relative=is_relative,
                                                                                              is_combined=is_combined)

        # for hyperparameter optimization
        config = {
            'hidden_dim': tune.choice([32, 64, 128]),
            'dropout': tune.uniform(0.0, 1.0)
        }
        if model_type is 'attention':
            config['ff_dim'] = tune.choice([8, 16, 32, 64])

        def train_tune(config, epochs=200, gpus=0):
            lightningModel = model_dict[model_type](training_normalized_data, get_new_mp_from_data_func, scaler,
                                                    batch_size,
                                                    seq_length, input_dim=train_dataset[0][0].shape[1],
                                                    teacher_forcing=True, loss=loss, is_matrix=is_matrix,
                                                    only_mp_features=only_mp_features, model_name=current_model_name,
                                                    weight_decay=weight_decay,
                                                    learning_rate=learning_rate, config=config)
            callback = TuneReportCallback(metrics={'loss': 'scaled_rmse_test_loss'},
                                          on='validation_end')
            trainer = pl.Trainer(max_epochs=200, callbacks=[callback, EarlyStopping(monitor='scaled_rmse_test_loss',
                                                                                     patience=5)])
            trainer.fit(lightningModel, train_data_loader, test_data_loader)

        reporter = CLIReporter(parameter_columns=list(config.keys()), metric_columns=['loss', 'training_iteration'])
        tune.run(partial(train_tune, epochs=200), config=config, num_samples=10, progress_reporter=reporter)

    run_type_dict = {'k_fold': k_fold_training,
                'prediction_only': run_prediction_only, 'find_optimal_param': find_optimal_parameters}


    def main():
        for data_source in all_data_sources:
            if is_matrix:
                data_source_normalized_data, normalized_data, columns, scaler, all_other_scalers, _ = get_matrix_normalize_data(data_source,
                                                                                                                                window_size=window_size,
                                                                                               is_relative=is_relative,
                                                                                                 is_combined=is_combined)
            else:
                normalized_data, columns, scaler = get_normalize_data(data_source)
                all_other_scalers = None
                data_source_normalized_data = normalized_data

            trial = {'dropout': dropout_list[i], 'hidden_dim': hidden_dim_list[i], 'ff_dim': ff_dim_list[i], 'learning_rate': 0.0004, 'weight_decay': 1e-5}

            run_type_dict[run_type](trial, data_source, data_source_normalized_data, normalized_data, scaler, all_other_scalers)

    main()

# more dataset
# more performance metrics
# report several parameters results