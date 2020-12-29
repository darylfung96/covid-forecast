import numpy as np
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import random
import matplotlib.pyplot as plt


from k_fold import RepHoldout
from data_preprocessing import get_matrix_normalize_data, get_normalize_data, get_mp_from_data, get_new_mp_from_data
from lstm import LightningModel, LightningModelAttention
from dataset import ForecastDataset


batch_size = 8
seq_length = 12
seed = 1000
is_matrix = False
is_relative = False
only_mp_features = False
is_matrix_str = 'matrix' if is_matrix else 'not matrix'
model_type = 'attention'
all_data_sources = [
    'hospital admission-adj (percentage of new admissions that are covid)', 'confirmed cases', 'death cases'
]
model_dict = {'lstm': LightningModel, 'attention': LightningModelAttention}
k_fold = True  # if not k fold, then train the whole data and do prediction


def k_fold_training(normalized_data):
    # create k fold split = 5
    k_fold_normalized_data = RepHoldout(n_splits=5).split(normalized_data)
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
        train_dataset = ForecastDataset(training_normalized_data, seq_length, only_mp_features=only_mp_features)
        test_dataset = ForecastDataset(testing_normalized_data, seq_length, only_mp_features=only_mp_features)
        train_data_loader = DataLoader(train_dataset, batch_size=batch_size, drop_last=False)
        test_data_loader = DataLoader(test_dataset, batch_size=batch_size, drop_last=False)

        lightningModel = model_dict[model_type](batch_size, seq_length, input_dim=train_dataset[0][0].shape[1],
                                        teacher_forcing=True, loss='mape', is_matrix=is_matrix)
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
    train_dataset = ForecastDataset(normalized_data, seq_length, only_mp_features)
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, drop_last=False)

    lightningModel = model_dict[model_type](batch_size, seq_length, input_dim=train_dataset[0][0].shape[1],
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
        input_data = normalized_data[i + backward_flow:i + seq_length]
        if only_mp_features:
            input_data = input_data[:, 1:]
        tensor_data = torch.Tensor(np.expand_dims(input_data, 0))
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
        if only_mp_features:
            predicted_data = predicted_data[:, 1:]
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


#  TODO: attention mechanism
### Your way - integrate attention mechanism to the LSTM model since matrix profiling algorithm tells us which point in the value that it is closest against
### Treat your matrix profile or the position as attention and model the raw observed data.
### May use other external data as attention, some research mention temperature is a factor, this can be used as attention?

