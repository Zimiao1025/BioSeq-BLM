import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from collections import Counter

from ..utils.utils_algorithm import data_partition
from .CNN_BiLSTM import CNNBiLSTM
from .DCNN_BiLSTM import DCNNBiLSTM
from .Motif_CNN import MotifCNN
from .Motif_DCNN import MotifDCNN
from .Auto_Encoder import AutoEncoder
from ..utils.utils_write import vectors2files
from ..OHE.OHE4vec import one_hot_enc

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 让torch判断是否使用GPU，建议使用GPU环境，因为会快很多

Method_IN_AF = ['One-hot', 'Binary-5bit', 'One-hot-6bit', 'Position-specific-2', 'Position-specific-3',
                'Position-specific-4', 'AESNN3', 'DBE', 'NCP', 'DPC', 'TPC', 'PP', 'PSSM', 'PSFM',
                'PAM250', 'BLOSUM62', 'BLAST-matrix', 'SS', 'SASA', 'RSS', 'CS']


def train(model, device, train_loader, optimizer, criterion, epoch, auto=False):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        if auto is True:
            data = data.view(len(data), -1)
            _, output = model(data.float())
            loss = criterion(output, data.float())
        else:
            output = model(data.float())
            loss = criterion(output, target.long())
        loss.backward()
        optimizer.step()
        if (batch_idx + 1) % 30 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset), 100. * batch_idx / len(train_loader),
                loss.item()))


# 测试的操作也一样封装成一个函数
def test(model, device, test_loader, criterion, auto=False):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            if auto is True:
                data = data.view(len(data), -1)
                _, output = model(data.float())
                test_loss += criterion(output, data.float()).item()  # 将一批的损失相加
            else:
                output = model(data.float())
                test_loss += criterion(output, target.long()).item()  # 将一批的损失相加

            prediction = output.max(1, keepdim=True)[1]  # 找到概率最大的下标
            correct += prediction.eq(target.long().view_as(prediction)).sum().item()
    if auto is True:
        test_loss /= len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}\n'.format(test_loss))
    else:
        test_loss /= len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))


METHODS_Auto_features = ['MotifCNN', 'MotifDCNN', 'CNN-BiLSTM', 'DCNN-BiLSTM', 'Autoencoder']


def extract_feature(method, train_data, test_data, n_class, in_dim, args, **params_dict):
    auto = False

    train_loader = DataLoader(train_data, batch_size=params_dict['batch_size'], shuffle=True)
    test_loader = DataLoader(test_data, batch_size=params_dict['batch_size'], shuffle=False)
    if method in METHODS_Auto_features:
        if method == 'MotifCNN':
            rnn = MotifCNN(params_dict['fea_dim'], n_class, params_dict['prob'], args).to(DEVICE)
        elif method == 'MotifDCNN':
            rnn = MotifDCNN(args.hidden_dim, args.n_layer, params_dict['fea_dim'], n_class,
                            params_dict['prob'], args).to(DEVICE)
        elif method == 'CNN-BiLSTM':
            rnn = CNNBiLSTM(in_dim, args.hidden_dim, args.n_layer, params_dict['fea_dim'], n_class,
                            params_dict['prob']).to(DEVICE)
        elif method == 'DCNN-BiLSTM':
            rnn = DCNNBiLSTM(in_dim, args.hidden_dim, args.n_layer, params_dict['fea_dim'], n_class,
                             params_dict['prob']).to(DEVICE)
        else:
            rnn = AutoEncoder(in_dim, args.hidden_dim, n_class).to(DEVICE)
            auto = True

        if auto is True:
            criterion = nn.MSELoss()
        else:
            criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(rnn.parameters(), lr=params_dict['lr'])
        epochs = params_dict['epoch']
        for epoch in range(1, epochs + 1):
            train(rnn, DEVICE, train_loader, optimizer, criterion, epoch, auto)
            test(rnn, DEVICE, test_loader, criterion, auto)
        # 保存模型
        # save_path = save_path + '/' + str(n_round) + 'round_model.pth'
        # torch.save(rnn.state_dict(), save_path)
        feature_one_round = []
        with torch.no_grad():
            for data, target in test_loader:
                data = data.to(DEVICE)
                if auto is True:
                    data = data.view(len(data), -1)
                    vec_tensor = rnn.extract_feature(data.float())
                else:
                    vec_tensor = rnn.extract_feature(data.float())
                # vec_tensor = out[:, -1, :]  # [batch_size, hidden_dim]
                vec_numpy = vec_tensor.numpy()
                feature_one_round += list(vec_numpy)
        return np.array(feature_one_round)


def mat_list2mat(mat_list, fixed_len):
    mat_array = []
    width = mat_list[0].shape[1]
    for i in range(len(mat_list)):
        temp_arr = np.zeros((fixed_len, width))
        temp_len = mat_list[i].shape[0]
        if temp_len <= fixed_len:
            temp_arr[:temp_len, :] = mat_list[i]
        else:
            temp_arr = mat_list[i][:fixed_len, :]

        mat_array.append(temp_arr)
    mat_array = np.array(mat_array)
    return mat_array


def auto_feature(method, input_file, labels, sample_num_list, out_file_list, args, **params_dict):
    # args.format,  args.category, args.current_dir, args.pp_file, args.cpu, args.fixed_len
    assert args.in_af in Method_IN_AF, 'Please set correct value for -in_fa parameter!'
    args.res = True
    vec_mat_list = one_hot_enc(input_file, args.category, args.in_af, args.current_dir, args.pp_file, args.rss_file,
                               args.cpu)
    from_mat = mat_list2mat(vec_mat_list, args.fixed_len)

    auto_vectors = np.zeros([len(from_mat), params_dict['fea_dim']]) if method != 'Autoencoder' else \
        np.zeros([len(from_mat), args.hidden_dim])
    # print(from_mat.shape)
    # exit()
    folds = data_partition(sample_num_list)
    n_class = len(Counter(labels).keys())
    in_dim = len(from_mat[0][0]) if method != 'Autoencoder' else (len(from_mat[0]) * len(from_mat[0][0]))
    # print(len(from_mat[0]))
    # print(len(from_mat[0][0]))
    # print(in_dim)
    for i, (train_index, test_index) in enumerate(folds):
        print('Round [%s]' % (i + 1))
        train_xy = []
        test_xy = []
        for x in train_index:
            train_xy.append([from_mat[x], labels[x]])
        for y in test_index:
            test_xy.append([from_mat[y], labels[y]])
        feature_one_round = extract_feature(method, train_xy, test_xy, n_class, in_dim, args, **params_dict)
        auto_vectors[test_index, :] = feature_one_round

    vectors2files(auto_vectors, sample_num_list, args.format, out_file_list)
