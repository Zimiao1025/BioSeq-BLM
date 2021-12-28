import torch
from torch import nn
import torch.nn.functional as func
from ..utils.utils_motif import MotifFile2Matrix, motif_init


def cnn_init(x, kernels):
    out = x.unsqueeze(1)  # [batch_size, 1, 20, 20]
    w_init = torch.empty(len(kernels), 1, 7, 20)  # 20 for Protein, 4 for DNA/RNA
    b_init = torch.tensor([0.1] * len(kernels))
    out = func.conv2d(out,
                      weight=torch.nn.init.uniform_(w_init),
                      bias=b_init)
    out = func.relu(out)
    out = func.max_pool2d(out, (2, 1))  # torch.Size([50, 301, 7, 1])  301 为模体个数，同时也是输出通道数。
    out_mean = torch.mean(out, dim=2, keepdim=True).squeeze()  # torch.Size([50, 301)
    out_max = torch.max(out, dim=2, keepdim=True)[0].squeeze()
    cnn_out = torch.cat([out_max, out_mean], 1)  # torch.Size([50, 602])
    return cnn_out


# 定义网络结构
class MotifCNN(torch.nn.Module):
    def __init__(self, fea_dim, n_class, prob, args):
        super(MotifCNN, self).__init__()
        motif_file = args.motif_file
        if args.motif_database == 'Mega':
            motifs = MotifFile2Matrix(motif_file).mega_motif_to_matrix()
        else:
            motifs = MotifFile2Matrix(motif_file).elm_motif_to_matrix()
        self.kernel = motifs
        self.fc = nn.Sequential(
            nn.Linear(len(motifs) * 4, fea_dim),
            nn.ReLU()
        )
        self.classifier = nn.Sequential(nn.Dropout(prob),
                                        nn.Linear(fea_dim, n_class))

    def extract_feature(self, x):
        cnn_layer = cnn_init(x, self.kernel)
        motif_layer = motif_init(x, self.kernel)
        out = torch.cat([motif_layer, cnn_layer], 1)  # size: [50 x 1204] 1204是模体个数的四倍
        out = self.fc(out)
        return out

    def forward(self, x):
        out = self.extract_feature(x)
        out = self.classifier(out)
        return out
