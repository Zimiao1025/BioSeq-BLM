import torch
from torch import nn


class AutoEncoder(nn.Module):
    def __init__(self, input_size, hidden_dim, n_class):
        super(AutoEncoder, self).__init__()
        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.fc = nn.Sequential(nn.Linear(self.input_size, self.hidden_dim),
                                nn.Sigmoid())
        self.encoder = nn.Linear(hidden_dim, n_class)
        self.decoder = nn.Sequential(nn.Linear(n_class, hidden_dim),
                                     nn.Sigmoid(),
                                     nn.Linear(hidden_dim, input_size))

    def extract_feature(self, x):
        # print(x.size())
        # exit()
        out = self.fc(x)
        return out

    def forward(self, x):
        out = self.extract_feature(x)
        encode = self.encoder(out)
        decode = self.decoder(encode)
        return encode, decode
