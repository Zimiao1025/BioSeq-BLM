import sys

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader, TensorDataset

from ..utils.utils_math import CRF
from ..utils.utils_plot import plot_roc_curve, plot_pr_curve, plot_roc_ind, plot_pr_ind
from ..utils.utils_results import performance, final_results_output, print_metric_dict

START_TAG = "<START>"
STOP_TAG = "<STOP>"


def get_partition(feature, target, length, train_index, val_index):
    feature = np.array(feature)
    x_train = feature[train_index]
    x_val = feature[val_index]
    y_train = np.array(target)[train_index]
    y_val = np.array(target)[val_index]
    train_length = np.array(length)[train_index]
    test_length = np.array(length)[val_index]

    return x_train, x_val, y_train, y_val, train_length, test_length


def crf_cv_process(vectors, labels, seq_length_list, folds, out_dir, params_dict):
    results = []
    cv_labels = []
    cv_prob = []

    # predicted_labels = np.zeros(len(seq_length_list))
    # predicted_prob = np.zeros(len(seq_length_list))

    count = 0

    width = vectors.shape[-1]

    for train_index, val_index in folds:
        train_x, test_x, train_y, test_y, train_length, test_length = get_partition(vectors, labels, seq_length_list,
                                                                                    train_index, val_index)

        lr = params_dict['lr']
        num_epochs = params_dict['epochs']
        batch_size = params_dict['batch_size']
        # 筛选最后的模型参数
        min_loss = float('inf')  # 无穷大

        opt_test_prob, opt_test_y_hat = [], []
        for num_epoch in range(num_epochs):
            test_loss, test_prob, test_y_hat = crf_main(train_x, test_x, train_y, test_y, width, batch_size, num_epoch,
                                                        lr)
            if test_loss < min_loss:
                min_loss = test_loss
                opt_test_prob = test_prob
                opt_test_y_hat = test_y_hat

        test_label_list, test_prob_list, predict_label_list = preprocess4evaluate(test_y, opt_test_prob,
                                                                                  opt_test_y_hat, test_length)

        result = performance(test_label_list, predict_label_list, test_prob_list, bi_or_multi=False, res=True)
        results.append(result)

        cv_labels.append(test_label_list)
        cv_prob.append(test_prob_list)

        count += 1
        print(" Round[%d]: Accuracy = %.3f | minimum loss = %.4f" % (count, result[0], min_loss))
    print('\n')
    plot_roc_curve(cv_labels, cv_prob, out_dir)  # 绘制ROC曲线
    plot_pr_curve(cv_labels, cv_prob, out_dir)  # 绘制PR曲线

    final_results = np.array(results).mean(axis=0)
    print_metric_dict(final_results, ind=False)
    final_results_output(final_results, out_dir, ind=False, multi=False)  # 将指标写入文件


def crf_ind_process(vectors, labels, ind_vectors, ind_labels, ind_seq_length_list, out_dir, params_dict):
    lr = params_dict['lr']
    num_epochs = params_dict['epochs']
    batch_size = params_dict['batch_size']
    # 筛选最后的模型参数
    min_loss = float('inf')  # 无穷大
    width = vectors.shape[-1]

    opt_test_prob, opt_test_y_hat = [], []
    for num_epoch in range(num_epochs):
        test_loss, test_prob, test_y_hat = crf_main(vectors, ind_vectors, labels, ind_labels, width, batch_size,
                                                    num_epoch, lr)
        if test_loss < min_loss:
            min_loss = test_loss
            opt_test_prob = test_prob
            opt_test_y_hat = test_y_hat

    test_label_list, test_prob_list, predict_label_list = preprocess4evaluate(ind_labels, opt_test_prob,
                                                                              opt_test_y_hat, ind_seq_length_list)

    final_result = performance(test_label_list, predict_label_list, test_prob_list, bi_or_multi=False, res=True)
    print_metric_dict(final_result, ind=True)

    plot_roc_ind(test_label_list, test_prob_list, out_dir)  # 绘制ROC曲线
    plot_pr_ind(test_label_list, test_prob_list, out_dir)  # 绘制PR曲线

    final_results_output(final_result, out_dir, ind=True, multi=False)  # 将指标写入文件
    # prob_output_res(final_target_list, final_predict_list, final_prob_list, out_dir)


def preprocess4evaluate(test_y, test_prob, test_y_hat, test_length):
    """ 将正确的测试集标签读取出来，而非全部内容进行评测 """
    test_label_list = []
    test_prob_list = []
    predict_label_list = []
    for i in range(len(test_length)):
        seq_len = test_length[i]
        test_label_list += list(test_y[i][:seq_len])
        test_prob_list += list(test_prob[i][:seq_len])
        predict_label_list += list(test_y_hat[i][:seq_len])
    return test_label_list, test_prob_list, predict_label_list


def make_data(train_x, test_x, train_y, test_y, batch_size):
    # train_x = np.random.normal(0, 0.1, (num_seq, seq_len, width))
    # test_x = np.random.normal(0, 0.1, (num_seq//2, seq_len, width))
    #
    # train_y = np.random.randint(0, 2, size=(num_seq, seq_len))
    # test_y = np.random.randint(0, 2, size=(num_seq//2, seq_len))

    if sys.platform.startswith('win'):
        num_workers = 0
    else:
        num_workers = 4

    train_dataset = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
    train_data_iter = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=num_workers)

    test_dataset = TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y))
    test_data_iter = DataLoader(test_dataset, 1, shuffle=False, num_workers=num_workers)

    return train_data_iter, test_data_iter


def train_crf(model, data_iter, epoch, optimizer=None):
    train_loss_sum = 0.0
    n = 1
    for x, y in data_iter:
        # 步骤1. 记住，pytorch积累了梯度
        # We need to clear them out before each instance
        model.zero_grad()

        # 步骤3. 向前运行
        loss = model.neg_log_likelihood_parallel(x.float(), y.long())

        # 步骤4.通过optimizer.step()
        loss.backward()
        optimizer.step()

        train_loss_sum += loss.item()
        n += y.shape[0]

    print('epoch[%d]: train loss: %.4f ' % (epoch + 1, train_loss_sum / n))
    # for epoch in range(num_epochs):
    # train_loss_sum = 0.0
    # n = 1
    # for x, y in data_iter:
    #     # 步骤1. 记住，pytorch积累了梯度
    #     # We need to clear them out before each instance
    #     model.zero_grad()
    #
    #     # 步骤3. 向前运行
    #     loss = model.neg_log_likelihood_parallel(x.float(), y.long())
    #
    #     # 步骤4.通过optimizer.step()
    #     loss.backward()
    #     optimizer.step()
    #
    #     train_loss_sum += loss.item()
    #     n += y.shape[0]
    #
    # print('epoch[%d]: train loss: %.4f ' % (epoch + 1, train_loss_sum / n))


def test_crf(model, data_iter, test_x):
    tag_hat_list = []
    test_loss_sum = 0.0
    n = 1
    with torch.no_grad():
        for x, y in data_iter:
            score, tag_seq = model(x.float())
            tag_hat_list.append(tag_seq)
            loss = model.neg_log_likelihood_parallel(x.float(), y.long())
            test_loss_sum += loss.item()
            n += y.shape[0]
    test_loss = test_loss_sum / n

    if sys.platform.startswith('win'):
        num_workers = 0
    else:
        num_workers = 4

    test_y_hat = np.array(tag_hat_list)
    test_dataset = TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y_hat))
    test_data_iter = DataLoader(test_dataset, 1, shuffle=False, num_workers=num_workers)

    test_prob = []
    with torch.no_grad():
        for x, y_ in test_data_iter:
            prob_list = model.calculate_pro_new(x.float(), y_.long())
            test_prob.append(prob_list)

    # prob_list = []
    # with torch.no_grad():
    #     for x, y in test_data_iter:
    #         prob_list += model.calculate_pro(x.float(), y.long())

    return test_loss, np.array(test_prob), test_y_hat


def crf_main(train_x, test_x, train_y, test_y, width, batch_size, num_epoch, lr):
    train_data_iter, test_data_iter = make_data(train_x, test_x, train_y, test_y, batch_size)
    # for X, y in train_data_iter:
    #     print(X.size())
    #     print(y)
    #     break
    tag_to_ix = {"B": 0, "O": 1, START_TAG: 2, STOP_TAG: 3}
    model = CRF(width, tag_to_ix)

    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=1e-4)
    train_crf(model, train_data_iter, num_epoch, optimizer)

    test_loss, test_prob, test_y_hat = test_crf(model, test_data_iter, test_x)
    # print(test_prob)
    return test_loss, test_prob, test_y_hat

# if __name__ == '__main__':
#     START_TAG = "<START>"
#     STOP_TAG = "<STOP>"
#     TAG2INDEX = {"B": 0, "O": 1, START_TAG: 2, STOP_TAG: 3}
#
#     NUM_SEQ, SEQ_LEN, WIDTH, BATCH_SIZE, NUM_EPOCHS = 100, 10, 10, 5, 13
#     print('... CRF processing ...\n')
#
#     crf_main(TAG2INDEX, NUM_SEQ, SEQ_LEN, WIDTH, BATCH_SIZE, NUM_EPOCHS)
#
#     print('\nFinish!')
