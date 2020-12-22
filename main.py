import numpy as np
import torch
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import random
import matplotlib.pyplot as plt

from data_preprocessing import get_matrix_normalize_data, get_normalize_data, get_mp_from_data, get_new_mp_from_data
from lstm import LightningModel
from dataset import ForecastDataset


batch_size = 8
seq_length = 12
seed = 1000
is_matrix = True
is_relative = True
is_matrix_str = 'matrix' if is_matrix else 'not matrix'
all_data_sources = [
    'hospital admission (percentage of new admissions that are covid)',
    'hospital admission-adj (percentage of new admissions that are covid)','confirmed cases','total confirmed cases',
    'death cases','total death cases']
k_fold = False  # if not k fold, then train the whole data and do prediction


def k_fold_training(normalized_data):
    # create k fold split = 5
    k_fold_normalized_data = KFold(n_splits=5).split(normalized_data)
    all_train_loss = []
    all_test_loss = []

    for train_index, test_index in k_fold_normalized_data:
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)
        torch.cuda.manual_seed_all(seed)

        training_normalized_data = normalized_data[train_index]
        testing_normalized_data = normalized_data[test_index]

        # create datasets
        train_dataset = ForecastDataset(training_normalized_data, seq_length)
        test_dataset = ForecastDataset(testing_normalized_data, seq_length)
        train_data_loader = DataLoader(train_dataset, batch_size=batch_size, drop_last=False)
        test_data_loader = DataLoader(test_dataset, batch_size=batch_size, drop_last=False)

        lightningModel = LightningModel(batch_size, seq_length, input_dim=normalized_data.shape[1],
                                        teacher_forcing=True)
        trainer = pl.Trainer(max_epochs=200)
        trainer.fit(lightningModel, train_data_loader, test_data_loader)
        all_train_loss.append(lightningModel.all_train_loss)
        all_test_loss.append(lightningModel.all_test_loss)

    all_train_loss = np.array(all_train_loss)
    np.save(f'train_{data_source}', all_train_loss)

    all_test_loss = np.array(all_test_loss)
    np.save(f'test_{data_source}', all_test_loss)


def whole_data_train(normalized_data, relative_scaler, data_source):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)

    # create datasets
    train_dataset = ForecastDataset(normalized_data, seq_length)
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, drop_last=False)

    lightningModel = LightningModel(batch_size, seq_length, input_dim=normalized_data.shape[1],
                                    teacher_forcing=True)
    trainer = pl.Trainer(max_epochs=200)
    trainer.fit(lightningModel, train_data_loader)

    lstm_model = lightningModel.lstm_model
    lstm_model.eval()
    hidden_states, cell_states = lstm_model.init_hiddenlstm_state()

    # train the whole data
    # if there is mp we can the first index (real value) and not the matrix profile first
    predictions = normalized_data[:seq_length, 0:1]
    for i in range(0, normalized_data.shape[0] - seq_length):

        backward_flow = normalized_data[i:i + seq_length].shape[0] - seq_length
        tensor_data = torch.Tensor(np.expand_dims(normalized_data[i + backward_flow:i + seq_length], 0))
        outputs, hidden_states, cell_states = lstm_model(tensor_data,
                                                         hidden_states, cell_states)

        next_prediction = outputs.cpu().detach().numpy()[0][-1:]

        if predictions is None:
            predictions = next_prediction
        else:
            predictions = np.concatenate([predictions, next_prediction], axis=0)

    if is_matrix:
        predictions = get_mp_from_data(predictions, relative_scaler, is_relative=is_relative)

    for i in range(180):
        predicted_data = predictions[-seq_length:]
        tensor_data = torch.Tensor(np.expand_dims(predicted_data, 0))
        outputs, hidden_states, cell_states = lstm_model(tensor_data, hidden_states, cell_states)

        next_prediction = outputs.cpu().detach().numpy()[0][-1:]

        if is_matrix:
            predictions = get_new_mp_from_data(predictions, next_prediction, relative_scaler, is_relative=is_relative)
        else:
            predictions = np.concatenate([predictions, next_prediction], 0)

    # for i in range(normalized_data.shape[1]):
    #     plt.title(data_source)
    #     plt.plot(range(normalized_data.shape[0]), normalized_data[:, i], 'b')
    #     plt.plot(range(len(predictions)), predictions[:, i], 'g')
    #     plt.show()

    np.save(f'{data_source}_actual_data', normalized_data)
    np.save(f'{data_source}', predictions)


for data_source in all_data_sources:
    if is_matrix:
        normalized_data, columns, scaler, relative_scaler, _ = get_matrix_normalize_data(data_source,
                                                                                       is_relative=is_relative)
    else:
        normalized_data, columns, scaler = get_normalize_data('confirmed cases')
        relative_scaler = None


    if k_fold:
        k_fold_training(normalized_data)
    else:
        whole_data_train(normalized_data, relative_scaler, data_source)


    # use radio graphs of hands and feets to classify
    # using machine learning to try to predict sharp scores (scoring the x-ray themselves)
    # x-ray are complex but there are specific areas more important than others, pertain to the joints (arthirtis)
    # The ability for AI to locate joints and subsequently score them, identify important findings on bone disease
    # can we create an algorithm that can read and identify joints before they can score them
