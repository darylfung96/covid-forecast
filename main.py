import numpy as np
import torch
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import random
import matplotlib.pyplot as plt

from data_preprocessing import get_matrix_normalize_data, get_normalize_data
from lstm import LightningModel
from dataset import ForecastDataset


batch_size = 8
seq_length = 12
seed = 1000
all_data_sources = [
    'hospital admission (percentage of new admissions that are covid)',
    'hospital admission-adj (percentage of new admissions that are covid)','confirmed cases','total confirmed cases',
    'death cases','total death cases']

for data_source in all_data_sources:
    is_matrix = True
    is_relative = False
    is_matrix_str = 'matrix' if is_matrix else 'not matrix'

    if is_matrix:
        normalized_data, columns, scaler, _ = get_matrix_normalize_data(data_source, is_relative=is_relative)
    else:
        normalized_data, columns, scaler = get_normalize_data('confirmed cases')

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
        trainer = pl.Trainer(max_epochs=300)
        trainer.fit(lightningModel, train_data_loader, test_data_loader)
        all_train_loss.append(lightningModel.all_train_loss)
        all_test_loss.append(lightningModel.all_test_loss)

        lstm_model = lightningModel.lstm_model
        lstm_model.eval()
        hidden_states, cell_states = lstm_model.init_hiddenlstm_state()

        # if there is mp we can the first index (real value) and not the matrix profile first
        predictions = normalized_data[:seq_length][0:1]

        for i in range(0, normalized_data.shape[0]):

            backward_flow = normalized_data[i:i+seq_length].shape[0] - seq_length
            tensor_data = torch.Tensor(np.expand_dims(normalized_data[i+backward_flow:i+seq_length], 0))
            outputs, hidden_states, cell_states = lstm_model(tensor_data,
                                                             hidden_states, cell_states)

            next_prediction = outputs.cpu().detach().numpy()[0][-1:]

            if predictions is None:
                predictions = next_prediction
            else:
                predictions = np.concatenate([predictions, next_prediction], axis=0)

        predictions



        # for i in range(60):
        #     predicted_data = predictions[-seq_length:]
        #     tensor_data = torch.Tensor(np.expand_dims(predicted_data, 0))
        #     outputs, hidden_states, cell_states = lstm_model(tensor_data, hidden_states, cell_states)
        #
        #     next_prediction = outputs.cpu().detach().numpy()[0][-1:]
        #     predictions = np.concatenate([predictions, next_prediction], axis=0)

        # for i in range(normalized_data.shape[1]):
        #     plt.title(columns[i])
        #     plt.plot(range(normalized_data.shape[0]), normalized_data[:, i], 'b')
        #     plt.plot(range(len(predictions)), predictions[:, i], 'g')
        #     plt.show()

    all_train_loss = np.array(all_train_loss)
    np.save(f'train_{data_source}_{is_matrix_str}', all_train_loss)
    # mean_train_loss = mean_train_loss.squeeze(1).tolist()
    # plt.title(f'mean train loss ({is_matrix_str})')
    # plt.plot(range(len(mean_train_loss)), mean_train_loss, 'r')

    all_test_loss = np.array(all_test_loss)
    np.save(f'test_{data_source}_{is_matrix_str}', all_test_loss)
    # mean_test_loss = mean_test_loss.squeeze(1).tolist()
    # plt.title(f'mean test loss ({is_matrix_str})')
    # plt.plot(range(len(mean_test_loss)), mean_test_loss, 'r')


    # use radio graphs of hands and feets to classify
    # using machine learning to try to predict sharp scores (scoring the x-ray themselves)
    # x-ray are complex but there are specific areas more important than others, pertain to the joints (arthirtis)
    # The ability for AI to locate joints and subsequently score them, identify important findings on bone disease
    # can we create an algorithm that can read and identify joints before they can score them
