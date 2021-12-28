from abc import ABC

import torch
import torch.nn as nn
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks
from numpy import random
from sklearn.model_selection import StratifiedKFold, KFold

START_TAG = "<START>"
STOP_TAG = "<STOP>"
# tag_to_ix = {"B": 0, "O": 1, START_TAG: 2, STOP_TAG: 3}
SEED = 42


# TODO: 将数据集划分为训练集和测试集
def construct_partition2two(labels, folds_num, stratified=True):
    # 将数据集划分为n折，并进行保存？ 如果需要进行相似性打分，则数据集划分提前到特征提取！
    vectors = random.normal(loc=0.0, scale=1, size=(len(labels), 64))
    if stratified is True:
        fold = StratifiedKFold(n_splits=folds_num, shuffle=True, random_state=random.RandomState(SEED))
        folds_temp = list(fold.split(vectors, labels))
    else:
        fold = KFold(n_splits=folds_num, shuffle=True, random_state=random.RandomState(SEED))
        folds_temp = list(fold.split(vectors))

    folds = []
    for i in range(folds_num):
        test_index = folds_temp[i][1]
        train_index = folds_temp[i][0]

        folds.append((train_index, test_index))
    return folds


def sampling(mode, x_train, y_train):
    # 只对训练数据进行采样
    if mode == 'over':
        # print('|*** Technique for sampling : oversampling ***|\n')
        x_train, y_train = SMOTE(random_state=42).fit_sample(x_train, y_train)
    elif mode == 'under':
        # print('|*** Technique for sampling : under sampling ***|\n')
        x_train, y_train = TomekLinks().fit_sample(x_train, y_train)
    else:
        # print('|*** Technique for sampling : combine oversampling  and under sampling ***|\n')
        x_train, y_train = SMOTETomek(random_state=42).fit_sample(x_train, y_train)

    # print(sorted(Counter(y_train).items()))
    return x_train, y_train


def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()


class CRF(nn.Module, ABC):

    def __init__(self, hidden_dim, tag_to_ix):
        super(CRF, self).__init__()
        self.tag_to_ix = tag_to_ix
        self.tag_set_size = len(tag_to_ix)

        # Maps the output of the LSTM into tag space.
        self.hidden2tag = nn.Linear(hidden_dim, self.tag_set_size)

        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.
        self.transitions = nn.Parameter(torch.normal(0.5, 0.167, [self.tag_set_size, self.tag_set_size]),
                                        requires_grad=True)
        # self.transitions = nn.Parameter(torch.randn(self.tag_set_size, self.tag_set_size))

        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag

        self.transitions.data[tag_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000
        # self.transitions.data[tag_to_ix[START_TAG], :] = 1e-3
        # self.transitions.data[:, tag_to_ix[STOP_TAG]] = 1e-3

    #     self.hidden = self.init_hidden()
    #
    # def init_hidden(self):
    #     return (torch.randn(2, 1, self.hidden_dim // 2),
    #             torch.randn(2, 1, self.hidden_dim // 2))

    def _forward_alg_new_parallel(self, feats):
        # Do the forward algorithm to compute the partition function
        init_alphas = torch.full([feats.shape[0], self.tag_set_size], -10000.)  # .to('cuda')
        # init_alphas = torch.full([feats.shape[0], self.tag_set_size], 1e-3)
        # START_TAG has all of the score.
        init_alphas[:, self.tag_to_ix[START_TAG]] = 0.

        # Wrap in a variable so that we will get automatic back-prop
        # Iterate through the sentence
        forward_var_list = [init_alphas]
        for feat_index in range(feats.shape[1]):  # -1
            gamar_r_l = torch.stack([forward_var_list[feat_index]] * feats.shape[2]).transpose(0, 1)
            t_r1_k = torch.unsqueeze(feats[:, feat_index, :], 1).transpose(1, 2)  # +1

            aa = gamar_r_l + t_r1_k + torch.unsqueeze(self.transitions, 0)

            forward_var_list.append(torch.logsumexp(aa, dim=2))
        terminal_var = forward_var_list[-1] + self.transitions[self.tag_to_ix[STOP_TAG]].repeat([feats.shape[0], 1])
        alpha = torch.logsumexp(terminal_var, dim=1)
        return alpha

    def _score_sentence_parallel(self, feats, tags):
        # Gives the score of provided tag sequences
        # feats = feats.transpose(0,1)

        score = torch.zeros(tags.shape[0])  # .to('cuda')
        tags = torch.cat([torch.full([tags.shape[0], 1], self.tag_to_ix[START_TAG], dtype=torch.long), tags], dim=1)
        for i in range(feats.shape[1]):
            feat = feats[:, i, :]
            # 明天把这一部分打印出来看看****
            score = score + self.transitions[tags[:, i + 1], tags[:, i]] + feat[range(feat.shape[0]), tags[:, i + 1]]
        score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[:, -1]]
        return score

    def neg_log_likelihood_parallel(self, feats, tags):
        feats = self.hidden2tag(feats)
        forward_score = self._forward_alg_new_parallel(feats)
        gold_score = self._score_sentence_parallel(feats, tags)
        return torch.sum(forward_score - gold_score)

    def _viterbi_decode_new(self, feats):
        back_pointers = []

        # Initialize the viterbi variables in log space
        init_vvars = torch.full((1, self.tag_set_size), -10000.)  # .to('cuda')
        # init_vvars = torch.full((1, self.tag_set_size), 1e-3)  # .to('cuda')
        init_vvars[0][self.tag_to_ix[START_TAG]] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var_list = [init_vvars]

        for feat_index in range(feats.shape[0]):
            gamma_r_l = torch.stack([forward_var_list[feat_index]] * feats.shape[1])
            gamma_r_l = torch.squeeze(gamma_r_l)
            next_tag_var = gamma_r_l + self.transitions
            # bptrs_t=torch.argmax(next_tag_var,dim=0)
            viterbi_vars_t, bptrs_t = torch.max(next_tag_var, dim=1)

            t_r1_k = torch.unsqueeze(feats[feat_index], 0)
            forward_var_new = torch.unsqueeze(viterbi_vars_t, 0) + t_r1_k

            forward_var_list.append(forward_var_new)
            back_pointers.append(bptrs_t.tolist())

        # Transition to STOP_TAG
        terminal_var = forward_var_list[-1] + self.transitions[self.tag_to_ix[STOP_TAG]]
        best_tag_id = torch.argmax(terminal_var).tolist()
        path_score = terminal_var[0][best_tag_id]
        # path_scores = terminal_var[0][:2].tolist()

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(back_pointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start == self.tag_to_ix[START_TAG]  # Sanity check
        best_path.reverse()
        return path_score, best_path

    def calculate_pro_new_(self, feat, tags_hat):
        # prob_batch = torch.zeros([feats.size()[0], feats.size()[1]])
        # print(self.transitions)
        self.transitions.data[self.tag_to_ix[START_TAG], :] = 1e-3
        self.transitions.data[:, self.tag_to_ix[STOP_TAG]] = 1e-3

        feat = feat.unsqueeze(0)
        feat = self.hidden2tag(feat).squeeze()  # [10, 4]

        feat.data[:, self.tag_to_ix[START_TAG]] = 1e-3
        feat.data[:, self.tag_to_ix[STOP_TAG]] = 1e-3

        trans_mat = self.transitions.data - torch.min(self.transitions.data, dim=0)[0].expand_as(self.transitions.data)
        state_mat = feat - torch.min(feat, dim=1, keepdim=True)[0].expand_as(feat)

        tags_hat = torch.cat([torch.tensor([self.tag_to_ix[START_TAG]], dtype=torch.long), tags_hat.view(-1)])
        # [3, 1, .., 0] 共11个标签
        prob_list = []
        for i in range(feat.shape[0]):
            # print(self.transitions[:, tags_hat[i]])  #
            # print(feat[i, :])

            score = trans_mat[tags_hat[i + 1], tags_hat[i]] + state_mat[i, tags_hat[i + 1]]

            score_total = trans_mat[:2, tags_hat[i]] + state_mat[i, :2]
            # print('score', score)
            # print('score_total', score_total)
            prob = score / torch.sum(score_total)
            # print('prob', prob)
            if tags_hat[i + 1] == 0:
                prob_list.append(1 - prob.item())
                # print('prob', 1 - prob.item())
            else:
                prob_list.append(prob.item())
                # print('prob', prob.item())

        return prob_list

    def calculate_pro_new(self, feat, tags_hat):
        # prob_batch = torch.zeros([feats.size()[0], feats.size()[1]])
        # print(self.transitions)
        # self.transitions.data[self.tag_to_ix[START_TAG], :] = 1e-3
        # self.transitions.data[:, self.tag_to_ix[STOP_TAG]] = 1e-3

        trans_mat = self.transitions.data[:2, :2]
        # print(trans_mat)
        trans_mat = trans_mat - torch.min(trans_mat, dim=0, keepdim=True)[0].expand_as(trans_mat)
        # print(trans_mat)

        feat = feat.unsqueeze(0)
        feat = self.hidden2tag(feat).squeeze()  # [10, 4]

        state_mat = feat[:, :2]
        state_mat = state_mat - torch.min(state_mat, dim=1, keepdim=True)[0].expand_as(state_mat)
        # print(state_mat[0])
        # state_mat = state_mat / torch.sum(state_mat, dim=0)
        # print(state_mat[0])
        # exit()

        # feat.data[:, self.tag_to_ix[START_TAG]] = 1e-3
        # feat.data[:, self.tag_to_ix[STOP_TAG]] = 1e-3

        tags_hat = torch.cat([torch.tensor([self.tag_to_ix[START_TAG]], dtype=torch.long), tags_hat.view(-1)])
        # tags_hat = tags_hat.view(-1)
        # print(tags_hat)
        # [3, 1, .., 0] 共11个标签
        prob_list = []
        for i in range(feat.shape[0]):
            # print(self.transitions[:, tags_hat[i]])
            # print(feat[i, :])
            if i == 0:
                score = state_mat[i, tags_hat[i + 1]]
                score_total = state_mat[i, ]
            else:
                score = trans_mat[tags_hat[i + 1], tags_hat[i]] + state_mat[i, tags_hat[i + 1]]
                score_total = trans_mat[:, tags_hat[i]] + state_mat[i, ]
            # print('score', score)
            # print('score_total', score_total)
            prob = score / torch.sum(score_total)
            # print('prob', prob)
            if tags_hat[i + 1] == 0:
                prob_list.append(1 - prob.item())
                # print('prob', 1 - prob.item())
            else:
                prob_list.append(prob.item())
                # print('prob', prob.item())

        return prob_list

    def forward(self, feats):  # dont confuse this with _forward_alg above.
        # Get the emission scores from the BiLSTM
        feats = self.hidden2tag(feats)
        feats = feats.squeeze(0)
        # Find the best path, given the features.
        score, tag_seq = self._viterbi_decode_new(feats)
        return score, tag_seq
