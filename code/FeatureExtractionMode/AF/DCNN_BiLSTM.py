import torch
from torch import nn


class DCNNBiLSTM(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, n_layer, fea_dim, n_class, prob=0.6):
        super(DCNNBiLSTM, self).__init__()
        self.in_dim = in_dim
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=1,
                      out_channels=16,
                      kernel_size=1,
                      stride=1),
            # torch.nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.rnn = nn.LSTM(in_dim, hidden_dim, n_layer,
                           bidirectional=True,
                           batch_first=True,
                           dropout=prob)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 1024),
            nn.Linear(1024, fea_dim)
        )
        self.classifier = nn.Linear(fea_dim, n_class)

    def extract_feature(self, x):
        out = x.unsqueeze(1)
        out = self.cnn(out)
        out = out.view(out.size()[0], -1, self.in_dim)
        out, _ = self.rnn(out)  # torch.Size([50, 80, 256])
        out = out[:, -1, :]
        out = self.fc(out)
        return out

    def forward(self, x):
        out = self.extract_feature(x)
        out = self.classifier(out)
        return out
