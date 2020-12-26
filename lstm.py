import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import random


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

    def forward(self, inputs, hidden_state, cell_state):
        inputs = inputs.view(inputs.shape[0], inputs.shape[1], self.input_dim)
        outputs = self.dropout_first_layer(F.relu(self.first_layer(inputs)))
        outputs, (hidden_state, cell_state) = self.lstm(outputs, (hidden_state, cell_state))
        outputs = self.output_layer(outputs)
        return outputs, hidden_state, cell_state

    def init_hiddenlstm_state(self):
        return torch.zeros(self.num_layers, self.seq_length, self.hidden_dim), \
               torch.zeros(self.num_layers, self.seq_length, self.hidden_dim)


class LightningModel(pl.LightningModule):
    def __init__(self, batch_size, seq_length, input_dim, teacher_forcing=True, loss='rmse'):
        super(LightningModel, self).__init__()
        self.lstm_model = LSTM(batch_size, seq_length=seq_length, input_dim=input_dim)
        self.criterion = nn.MSELoss()

        self.all_train_loss = []
        self.current_train_loss = []
        self.all_test_loss = []
        self.current_test_loss = []

        self.teacher_forcing = teacher_forcing
        self.loss = self.create_loss(loss)

    def create_loss(self, loss):
        if loss == 'rmse':
            return lambda predict, target: torch.sqrt(self.criterion(predict, target) + 1e-6)
        elif loss == 'mape':
            return lambda predict, target: torch.mean((target - predict).abs() / target.abs())

    def forward(self, inputs):
        outputs, hidden_states, cell_states = self.lstm_model(inputs)
        return outputs,hidden_states, cell_states

    def on_train_epoch_start(self):
        self.hidden_states, self.cell_states = self.lstm_model.init_hiddenlstm_state()
        self.lstm_model.train()

    def training_step(self, batch, batch_index):
        x, y = batch

        if batch_index != 0:
            if self.teacher_forcing:
                if random.random() > 0.5:
                    x[:, :, 0:1] = self.last_outputs[-batch[0].shape[0]:]  # when it's not exactly the batch size
            elif not self.teacher_forcing:
                x[:, :, 0:1] = self.last_outputs[-batch[0].shape[0]:]  # when it's not exactly the batch size

        outputs, self.hidden_states, self.cell_states = self.lstm_model(x, self.hidden_states, self.cell_states)
        loss = self.loss(outputs, y)  # Root mean squared error

        self.last_outputs = outputs.detach()
        self.hidden_states = self.hidden_states.detach()
        self.cell_states = self.cell_states.detach()

        self.log('train_loss', loss.item(), prog_bar=True)
        self.current_train_loss.append(loss.item())

        return loss

    def on_train_epoch_end(self):
        self.all_train_loss.append(sum(self.current_train_loss) / len(self.current_train_loss))
        self.current_train_loss = []

    def on_validation_epoch_start(self):
        self.hidden_states, self.cell_states = self.lstm_model.init_hiddenlstm_state()
        self.lstm_model.eval()

    def validation_step(self, batch, batch_index):
        x, y = batch
        outputs, self.hidden_states, self.cell_states = self.lstm_model(x, self.hidden_states, self.cell_states)
        loss = self.loss(outputs, y)

        self.hidden_states = self.hidden_states.detach()
        self.cell_states = self.cell_states.detach()

        self.log('test_loss', loss.item(), prog_bar=True)
        self.current_test_loss.append(loss.item())

    def on_validation_epoch_end(self):
        self.all_test_loss.append(sum(self.current_test_loss) / len(self.current_test_loss))
        self.current_test_loss = []

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001, weight_decay=1e-5)
        return optimizer


class LSTMAttention(nn.Module):
    def __init__(self):
        super(LSTMAttention, self).__init__()
        self.memory_bank = None
        self.memory_bank_size = 200

        self.first_enoder_layer = nn.TransformerEncoderLayer(32, 8, 128)
        self.first_decoder_layer = nn.TransformerDecoderLayer(32, 8, 128)

    def init_memory_bank(self, memories):
        self.memory_bank = memories[-200:]

    def forward(self, inputs, memory, src_mask):  # memory is a longer sequence from more to the past
        # can try inputs as:
        # relative matrix
        # raw
        # matrix profile
        #
        # memory as:
        # raw

        first_encoder_output = self.first_enoder_layer(inputs)

        # first from the encoder inputs, second from the matrix profiling to focus on the nearest neighbours
        decoder_output = self.first_decoder_layer(first_encoder_output, memory)
        return decoder_output


class LightningModelAttention(LightningModel):
    def __init__(self):
        super(LightningModelAttention, self).__init__()

