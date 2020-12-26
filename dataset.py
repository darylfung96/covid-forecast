import torch
from torch.utils.data import Dataset


class ForecastDataset(Dataset):
    def __init__(self, normalized_data, seq_length, only_mp_features):
        super(ForecastDataset, self).__init__()
        self.seq_length = seq_length
        self.normalized_data = normalized_data
        self.only_mp_features = only_mp_features

    def __len__(self):
        return len(self.normalized_data) - self.seq_length

    def __getitem__(self, index):

        if self.only_mp_features:
            assert self.normalized_data.ndim > 1
            return torch.Tensor(self.normalized_data[index:index + self.seq_length, 1:]), \
                   torch.Tensor(self.normalized_data[index + 1:index + self.seq_length + 1][:, 0:1])
        else:
            return torch.Tensor(self.normalized_data[index:index+self.seq_length]), \
                    torch.Tensor(self.normalized_data[index + 1:index + self.seq_length + 1][:, 0:1])
        # remove outputting the matrix profile, that's why we only get the first index [0:1] because the first index is the raw value, the second index is the matrix profile
