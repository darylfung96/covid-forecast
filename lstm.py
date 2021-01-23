import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pytorch_lightning as pl


class LSTM(nn.Module):
    def __init__(self, batch_size=12, seq_length=5, input_dim=6, hidden_dim=8, output_dim=1):
        super(LSTM, self).__init__()

        self.hidden_dim = hidden_dim
        self.seq_length = seq_length
        self.input_dim = input_dim
        self.batch_size = batch_size
        self.output_dim = output_dim

        self.first_layer = nn.Linear(input_dim, hidden_dim)
        self.dropout_first_layer = nn.Dropout(0.2)
        self.lstm = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, dropout=0.2)
        self.output_layer = nn.Linear(hidden_dim, self.output_dim)
        self.num_layers = 1
        self.device = None

    def forward(self, inputs, hidden_state, cell_state):
        inputs = inputs.view(inputs.shape[0], inputs.shape[1], self.input_dim)
        outputs = self.dropout_first_layer(F.relu(self.first_layer(inputs)))
        outputs, (hidden_state, cell_state) = self.lstm(outputs, (hidden_state, cell_state))
        outputs = self.output_layer(outputs)
        return outputs, hidden_state, cell_state

    def init_hiddenlstm_state(self):
        if self.device is None:
            self.device = next(self.lstm.parameters()).device

        return torch.zeros(self.num_layers, self.seq_length, self.hidden_dim).to(self.device), \
               torch.zeros(self.num_layers, self.seq_length, self.hidden_dim).to(self.device)


class LightningModel(pl.LightningModule):
    def __init__(self, training_normalized_data, get_new_mp_from_data_func, scaler, batch_size, seq_length, input_dim, teacher_forcing=True, loss='rmse', is_matrix=False,
                 only_mp_features=None, model_name='', weight_decay=1e-2, hidden_dim=64, ff_dim=64, dropout=0.2, learning_rate=1e-3):
        super(LightningModel, self).__init__()
        self.training_normalized_data  = training_normalized_data
        self.lstm_model = LSTM(batch_size, seq_length=seq_length, input_dim=input_dim, hidden_dim=hidden_dim)
        self.criterion = nn.MSELoss()
        self.is_matrix = is_matrix
        self.scaler = scaler
        self.seq_length = seq_length
        self.only_mp_features = only_mp_features
        self.model_name = model_name
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.hidden_dim=hidden_dim
        self.dropout = dropout
        self.get_new_mp_from_data_func = get_new_mp_from_data_func
        self.linear_decay = 0.5
        self.epsilon = 0.000

        self.all_train_loss = []
        self.current_train_loss = []
        self.all_test_loss = []
        self.current_test_loss = []
        self.validation_input = None

        self.teacher_forcing = teacher_forcing
        self.loss = self.create_loss(loss)
        self.best_loss = 1e9

        self.save_filename = os.path.join('models', f'{self.model_name}.ckpt')
        os.makedirs(os.path.dirname(self.save_filename), exist_ok=True)

    def create_loss(self, loss):
        if loss == 'rmse':
            return lambda predict, target: torch.sqrt(self.criterion(predict, target) + 1e-6)
        elif loss == 'mape':
            return lambda predict, target: torch.mean((target - predict).abs() / target.abs())

    def forward(self, inputs):
        outputs, hidden_states, cell_states = self.lstm_model(inputs)
        return outputs,hidden_states, cell_states

    def on_train_epoch_start(self):
        self.current_train_loss = []
        self.hidden_states, self.cell_states = self.lstm_model.init_hiddenlstm_state()
        self.lstm_model.train()

    def training_step(self, batch, batch_index):
        x, y = batch

        x = x.to(self.device)
        y = y.to(self.device)

        if batch_index != 0:
            if self.teacher_forcing:
                if random.random() > 0.5:
                    x[:, :, 0:] = self.last_outputs[-batch[0].shape[0]:]  # when it's not exactly the batch size
            elif not self.teacher_forcing:
                x[:, :, 0:] = self.last_outputs[-batch[0].shape[0]:]  # when it's not exactly the batch size

        outputs, self.hidden_states, self.cell_states = self.lstm_model(x, self.hidden_states, self.cell_states)
        loss = self.loss(outputs, y)  # Root mean squared error

        self.last_outputs = outputs.detach()
        self.hidden_states = self.hidden_states.detach()
        self.cell_states = self.cell_states.detach()

        self.log('train_loss', loss.item(), prog_bar=True)
        self.current_train_loss.append(loss.item())

        return loss

    def on_train_epoch_end(self, *args):
        self.all_train_loss.append(sum(self.current_train_loss) / len(self.current_train_loss))

    def on_validation_epoch_start(self):
        self.current_test_loss = []
        self.hidden_states, self.cell_states = self.lstm_model.init_hiddenlstm_state()
        self.lstm_model.eval()
        self.validation_input = self.training_normalized_data[-2*self.seq_length:]

    def validation_step(self, batch, batch_index):
        x, y = batch

        x = torch.Tensor(self.validation_input[-self.seq_length:]).to(self.device).unsqueeze(0)
        if self.only_mp_features:
            x = x[:, :, 1:]
        y = y.to(self.device)

        outputs, self.hidden_states, self.cell_states = self.lstm_model(x, self.hidden_states, self.cell_states)
        loss = self.loss(outputs[:, -1:, :], y)

        if self.is_matrix:
            self.validation_input = self.get_new_mp_from_data_func(self.validation_input, outputs.cpu().detach().numpy()[0, -1:])

        scaled_outputs = torch.Tensor(self.scaler.inverse_transform(outputs.cpu()))
        scaled_y = torch.Tensor(self.scaler.inverse_transform(y.cpu()))
        scaled_loss = self.loss(scaled_outputs[:, -1:, :], scaled_y)

        self.hidden_states = self.hidden_states.detach()
        self.cell_states = self.cell_states.detach()

        self.log('test_loss', loss.item(), prog_bar=True)
        self.current_test_loss.append(scaled_loss.item())
        self.log('scaled_test_loss', scaled_loss.item(), prog_bar=True)

    def on_validation_epoch_end(self):
        mean_test_loss = sum(self.current_test_loss) / len(self.current_test_loss)
        self.all_test_loss.append(mean_test_loss)
        if self.all_test_loss[-1] < self.best_loss:
            torch.save(self.lstm_model.state_dict(), os.path.join('models', f'{self.model_name}.ckpt'))

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        return optimizer


class LSTMAttention(nn.Module):
    def __init__(self, input_dim, seq_length, hidden_dim=128, num_heads=8, ff_dim=128, output_dim=1, dropout=0.5):
        super(LSTMAttention, self).__init__()
        self.seq_length = seq_length
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout = dropout

        self.pre_layer = nn.Linear(input_dim, hidden_dim)
        self.memory_pre_layer = nn.Linear(input_dim, hidden_dim)
        self.dropout_first_layer = nn.Dropout(self.dropout)
        self.dropout_memory_layer = nn.Dropout(self.dropout)
        self.lstm = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, dropout=0)
        self.output_layer = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                          nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, self.output_dim))
        self.num_layers = 1
        self.device = None

        self.encoder_input_pre_layer = nn.Linear(input_dim, hidden_dim)
        self.encoder_pre_layer = nn.Sequential(nn.Linear(2*hidden_dim, hidden_dim),
                                               nn.ReLU(),
                                               nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                               nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.first_enoder_layer = nn.TransformerEncoderLayer(hidden_dim, num_heads, ff_dim)
        self.first_decoder_layer = nn.TransformerDecoderLayer(hidden_dim, num_heads, ff_dim)

    def init_hiddenlstm_state(self):
        if self.device is None:
            self.device = next(self.lstm.parameters()).device
        return torch.zeros(self.num_layers, self.seq_length, self.hidden_dim).to(self.device), \
               torch.zeros(self.num_layers, self.seq_length, self.hidden_dim).to(self.device)

    def forward(self, inputs, memory, hidden_state, cell_state):  # memory is a longer sequence from more to the past
        # can try inputs as:
        # relative matrix
        # raw
        # matrix profile
        #
        # memory as:
        # raw
        pre_outputs = self.dropout_first_layer(F.relu(self.pre_layer(inputs)))
        lstm_outputs, (hidden_state, cell_state) = self.lstm(pre_outputs, (hidden_state, cell_state))

        pre_encoder_inputs = F.relu(self.encoder_input_pre_layer(inputs))
        pre_encoder_inputs = torch.cat([lstm_outputs, pre_encoder_inputs], -1)
        # pre_encoder_inputs = torch.cat([pre_outputs, pre_encoder_inputs], -1)
        pre_encoder_inputs = self.encoder_pre_layer(pre_encoder_inputs)

        first_encoder_output = self.first_enoder_layer(pre_encoder_inputs)

        encoded_memory = self.dropout_memory_layer(F.relu(self.memory_pre_layer(memory)))
        # first from the encoder inputs, second from the matrix profiling to focus on the nearest neighbours
        decoder_output = self.first_decoder_layer(first_encoder_output, encoded_memory)

        decoder_output = self.output_layer(decoder_output)
        return decoder_output, hidden_state, cell_state


class LightningModelAttention(LightningModel):
    def __init__(self, training_normalized_data, get_new_mp_from_data_func, scaler, batch_size, seq_length, input_dim, teacher_forcing=True, loss='rmse', is_matrix=False,
                 only_mp_features=False, model_name='', hidden_dim=4, ff_dim=4, weight_decay=1e-2, dropout=0.5,
                 learning_rate=1e-3):
        # because the second dimension is used as memory for the attention

        # try both TODO: remove
        # if is_matrix and not only_mp_features:
        #     input_dim = input_dim-1

        super(LightningModelAttention, self).__init__(training_normalized_data, get_new_mp_from_data_func, scaler, batch_size, seq_length, input_dim,
                                                      teacher_forcing, loss, is_matrix, only_mp_features, model_name,
                                                      weight_decay=weight_decay, learning_rate=learning_rate)
        self.only_mp_features = only_mp_features
        self.lstm_model = LSTMAttention(input_dim, seq_length, hidden_dim=hidden_dim, ff_dim=ff_dim, dropout=dropout)
        self.model_name = model_name

    def on_train_epoch_start(self):
        super(LightningModelAttention, self).on_train_epoch_start()

    def training_step(self, batch, batch_index):
        x, y = batch

        x = x.to(self.device)
        y = y.to(self.device)
        self.linear_decay -= self.epsilon
        if batch_index != 0:
            if self.teacher_forcing:
                if random.random() > self.linear_decay:
                    x[:, :, 0:1] = self.last_outputs[-batch[0].shape[0]:]  # when it's not exactly the batch size
            elif not self.teacher_forcing:
                x[:, :, 0:1] = self.last_outputs[-batch[0].shape[0]:]  # when it's not exactly the batch size

        # if self.is_matrix and not self.only_mp_features:
        #     x_inputs = x[:, :, 1:]
        #     memory_inputs = x[:, :, 0:1]
        # else:
        #     x_inputs = x
        #     memory_inputs = x
        x_inputs = x
        memory_inputs = x

        outputs, self.hidden_states, self.cell_states = self.lstm_model(x_inputs, memory_inputs, self.hidden_states,
                                                                        self.cell_states)
        loss = self.loss(outputs, y)  # Root mean squared error

        self.last_outputs = outputs.detach()
        self.hidden_states = self.hidden_states.detach()
        self.cell_states = self.cell_states.detach()

        self.log('train_loss', loss.item(), prog_bar=True)
        self.current_train_loss.append(loss.item())

        return loss

    def validation_step(self, batch, batch_index):
        x, y = batch

        x = torch.Tensor(self.validation_input[-self.seq_length:]).to(self.device).unsqueeze(0)
        if self.only_mp_features:
            x = x[:, :, 1:]
        # x = x.to(self.device)
        y = y.to(self.device)

        # if self.is_matrix and not self.only_mp_features:
        #     x_inputs = x[:, :, 1:]
        #     memory_inputs = x[:, :, 0:1]
        # else:
        #     x_inputs = x
        #     memory_inputs = x
        x_inputs = x
        memory_inputs = x

        outputs, self.hidden_states, self.cell_states = self.lstm_model(x_inputs, memory_inputs, self.hidden_states,
                                                                        self.cell_states)

        loss = self.loss(outputs[:, -1:, :], y)

        if self.is_matrix:
            self.validation_input = self.get_new_mp_from_data_func(self.validation_input, outputs.cpu().detach().numpy()[0, -1:])

        scaled_outputs = torch.Tensor(self.scaler.inverse_transform(outputs.cpu()))
        scaled_y = torch.Tensor(self.scaler.inverse_transform(y.cpu()))
        scaled_loss = self.loss(scaled_outputs[:, -1:, :], scaled_y)

        self.hidden_states = self.hidden_states.detach()
        self.cell_states = self.cell_states.detach()

        self.log('test_loss', loss.item(), prog_bar=True)
        self.current_test_loss.append(scaled_loss.item())
        self.log('scaled_test_loss', scaled_loss.item(), prog_bar=True)

