import math

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn.init as init
from torch import nn
from torch.autograd import Function


#  TODO: Transforemr
# Reference
# **Paper**
# - Vaswani et al., "Attention is All You Need", NIPS 2017
# - Ahmed et al., "Weighted Transformer Network for Machine Translation", Arxiv 2017
# **Code**
# https://github.com/jayparks/transformer


class Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(Linear, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        init.xavier_normal_(self.linear.weight)
        init.zeros_(self.linear.bias)

    def forward(self, inputs):
        return self.linear(inputs)


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k, n_heads, dropout=.1):
        super(ScaledDotProductAttention, self).__init__()
        self.scale_factor = np.sqrt(d_k)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.n_heads = n_heads

    def forward(self, q, k, v, attn_mask=None):
        # q: [b_size x n_heads x len_q x d_k]
        # k: [b_size x n_heads x len_k x d_k]
        # v: [b_size x n_heads x len_v x d_v] note: (len_k == len_v)

        # attn: [b_size x n_heads x len_q x len_k]
        scores = torch.matmul(q, k.transpose(-1, -2)) / self.scale_factor
        if attn_mask is not None:
            # attn_mask: [b_size x len_q x len_k]
            attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)  # [b_size x n_heads x len_q x len_k]
            assert attn_mask.size() == scores.size()
            scores.masked_fill_(attn_mask, -1e9)
        attn = self.dropout(self.softmax(scores))

        # outputs: [b_size x n_heads x len_q x d_v]
        context = torch.matmul(attn, v)

        return context, attn


class LayerNormalization(nn.Module):
    def __init__(self, d_hid, eps=1e-6):
        super(LayerNormalization, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_hid))
        self.beta = nn.Parameter(torch.zeros(d_hid))
        self.eps = eps

    def forward(self, z):
        mean = z.mean(dim=-1, keepdim=True, )
        std = z.std(dim=-1, keepdim=True, )
        ln_out = (z - mean) / (std + self.eps)
        ln_out = self.gamma * ln_out + self.beta

        return ln_out


class _MultiHeadAttention(nn.Module):
    def __init__(self, d_k, d_v, d_model, n_heads, dropout):
        super(_MultiHeadAttention, self).__init__()
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model
        self.n_heads = n_heads

        self.w_q = Linear(d_model, d_k * n_heads)
        self.w_k = Linear(d_model, d_k * n_heads)
        self.w_v = Linear(d_model, d_v * n_heads)

        self.attention = ScaledDotProductAttention(d_k, n_heads, dropout)

    def forward(self, q, k, v, attn_mask):
        # q: [b_size x len_q x d_model] - > 实例： torch.Size([50, 20, 64])
        # k: [b_size x len_k x d_model]
        # v: [b_size x len_k x d_model]
        b_size = q.size(0)

        # q_s: [b_size x n_heads x len_q x d_k]
        # k_s: [b_size x n_heads x len_k x d_k]
        # v_s: [b_size x n_heads x len_k x d_v]

        q_s = self.w_q(q).view(b_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k_s = self.w_k(k).view(b_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v_s = self.w_v(v).view(b_size, -1, self.n_heads, self.d_v).transpose(1, 2)

        # context: [b_size x n_heads x len_q x d_v], attn: [b_size x n_heads x len_q x len_k]
        context, attn = self.attention(q_s, k_s, v_s, attn_mask=attn_mask)
        # context: [b_size x len_q x n_heads * d_v]
        context = context.transpose(1, 2).contiguous().view(b_size, -1, self.n_heads * self.d_v)
        # 将多层缩放点积注意力得到的上下文向量串接起来
        # 如果在view之前用了transpose, permute等，需要用contiguous()来返回一个contiguous copy
        # return the context and attention weights
        return context, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_k, d_v, d_model, n_heads, dropout):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.multi_head_attn = _MultiHeadAttention(d_k, d_v, d_model, n_heads, dropout)
        self.projects = Linear(n_heads * d_v, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = LayerNormalization(d_model)

    def forward(self, q, k, v, attn_mask):
        # q: [b_size x len_q x d_model]
        # k: [b_size x len_k x d_model]
        # v: [b_size x len_v x d_model] note (len_k == len_v)
        residual = q
        # context: a tensor of shape [b_size x len_q x n_heads * d_v]
        context, attn = self.multi_head_attn(q, k, v, attn_mask=attn_mask)

        # project back to the residual size, outputs: [b_size x len_q x d_model]
        output = self.dropout(self.projects(context))  # 多头注意力的输入和输出尺寸完全相同
        return self.layer_norm(residual + output), attn


class MultiBranchAttention(nn.Module):
    def __init__(self, d_k, d_v, d_model, d_ff, n_branches, dropout):
        super(MultiBranchAttention, self).__init__()
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model
        self.d_ff = d_ff
        self.n_branches = n_branches

        self.multihead_attn = _MultiHeadAttention(d_k, d_v, d_model, n_branches, dropout)
        # additional parameters for BranchedAttention
        self.w_o = nn.ModuleList([Linear(d_v, d_model) for _ in range(n_branches)])
        self.w_kp = torch.rand(n_branches)
        self.w_kp = nn.Parameter(self.w_kp / self.w_kp.sum())
        self.w_a = torch.rand(n_branches)
        self.w_a = nn.Parameter(self.w_a / self.w_a.sum())

        self.pos_ffn = nn.ModuleList([
            PoswiseFeedForwardNet(d_model, d_ff // n_branches, dropout) for _ in range(n_branches)])
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = LayerNormalization(d_model)

        init.xavier_normal(self.w_o)

    def forward(self, q, k, v, attn_mask):
        # q: [b_size x len_q x d_model]
        # k: [b_size x len_k x d_model]
        # v: [b_size x len_v x d_model] note (len_k == len_v)
        residual = q

        # context: a tensor of shape [b_size x len_q x n_branches * d_v]
        context, attn = self.multih_attn(q, k, v, attn_mask=attn_mask)

        # context: a list of tensors of shape [b_size x len_q x d_v] len: n_branches
        context = context.split(self.d_v, dim=-1)

        # outputs: a list of tensors of shape [b_size x len_q x d_model] len: n_branches
        outputs = [self.w_o[i](context[i]) for i in range(self.n_branches)]
        outputs = [kappa * output for kappa, output in zip(self.w_kp, outputs)]
        outputs = [pos_ffn(output) for pos_ffn, output in zip(self.pos_ffn, outputs)]
        outputs = [alpha * output for alpha, output in zip(self.w_a, outputs)]

        # output: [b_size x len_q x d_model]
        output = self.dropout(torch.stack(outputs).sum(dim=0))
        return self.layer_norm(residual + output), attn


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PoswiseFeedForwardNet, self).__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = LayerNormalization(d_model)

    def forward(self, inputs):
        # inputs: [b_size x len_q x d_model]
        residual = inputs
        output = self.relu(self.conv1(inputs.transpose(1, 2)))

        # outputs: [b_size x len_q x d_model]
        output = self.conv2(output).transpose(1, 2)
        output = self.dropout(output)

        return self.layer_norm(residual + output)


class EncoderLayer(nn.Module):
    def __init__(self, d_k, d_v, d_model, d_ff, n_heads, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention(d_k, d_v, d_model, n_heads, dropout)
        self.pos_ffn = PoswiseFeedForwardNet(d_model, d_ff, dropout)

    def forward(self, enc_inputs, self_attn_mask):
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs,
                                               enc_inputs, attn_mask=self_attn_mask)
        enc_outputs = self.pos_ffn(enc_outputs)

        return enc_outputs, attn


class WeightedEncoderLayer(nn.Module):
    def __init__(self, d_k, d_v, d_model, d_ff, n_branches, dropout=0.1):
        super(WeightedEncoderLayer, self).__init__()
        self.enc_self_attn = MultiBranchAttention(d_k, d_v, d_model, d_ff, n_branches, dropout)

    def forward(self, enc_inputs, self_attn_mask):
        return self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, attn_mask=self_attn_mask)


def get_attn_pad_mask(seq_q, seq_k):
    assert seq_q.dim() == 2 and seq_k.dim() == 2
    # print(seq_q.size())
    b_size, len_q = seq_q.size()
    b_size, len_k = seq_k.size()
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # b_size x 1 x len_k
    return pad_attn_mask.expand(b_size, len_q, len_k)  # b_size x len_q x len_k


class Transformer(nn.Module):
    def __init__(self, enc_inputs_len, n_layers, d_k, d_v, d_model, d_ff, n_heads,
                 dropout=0.1, weighted=False):
        super(Transformer, self).__init__()
        self.d_model = d_model
        self.projects = Linear(enc_inputs_len, d_model)
        self.layer_type = EncoderLayer if not weighted else WeightedEncoderLayer
        self.layers = nn.ModuleList(
            [self.layer_type(d_k, d_v, d_model, d_ff, n_heads, dropout) for _ in range(n_layers)])
        # 原论文编码解码均有6层，上一层的输出为下一层的输入, 我们不要解码层

    def forward(self, enc_inputs, seq_mask, mask=False, return_attn=False):
        # enc_inputs.size() -> torch.Size([50, 20, 4]) (batch_size, seq_len, d_model)
        if mask is False:
            enc_self_attn_mask = None
        else:
            enc_self_attn_mask = get_attn_pad_mask(seq_mask, seq_mask)
            # 之所以修改是由于mask与字符序列的pad有关（因为补齐而导致部分序列后面一部分为0值），但当以矩阵作为输入时，这一点不存在了

        enc_outputs = self.projects(enc_inputs)
        # print('After project operation:')
        # print(enc_outputs.size())  # torch.Size([50, 20, 64])
        enc_self_attns = []
        for layer in self.layers:
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            if return_attn:
                enc_self_attns.append(enc_self_attn)
        # enc_outputs： [b_size x len_q x d_model], attn: [b_size x n_heads x len_q x len_k]
        return enc_outputs, enc_self_attns


# TODO: Reformer
# Reference
# **Paper**
# https://openreview.net/pdf?id=rkgNKkHtvB
# **Code**
# https://github.com/lucidrains/reformer-pytorch


def deterministic_dropout(x: torch.Tensor, seed=0, dropout=0):
    generator = torch.Generator(device=x.get_device())
    generator.manual_seed(seed)
    dropout_mask = torch.bernoulli(x, p=1 - dropout, generator=generator)
    return dropout_mask * x / (1 - dropout)


def look_back(input_tensor: torch.Tensor) -> torch.Tensor:
    """
    Looks back one bucket
    """
    shift = torch.cat([input_tensor[:, -1:], input_tensor[:, :-1]], dim=1)
    # [batch * head, n_buckets, bucket_length, d_k, rounds]
    concat = torch.cat([shift, input_tensor], dim=2)
    # [batch * head, n_buckets, bucket_length * 2, d_k, rounds]
    return concat


def reverse_sort(indice: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Unsorts sorted indice
    """
    new_size = [1] * indice.dim()
    new_size[dim] = indice.size(dim)
    arange = indice.new_empty(size=new_size)
    torch.arange(new_size[dim], out=arange)
    arange = arange.expand_as(indice)
    new_indice = torch.empty_like(indice)
    new_indice.scatter_(dim=dim, index=indice, src=arange)
    return new_indice


def expand(input_tensor: torch.Tensor, dim=0, num=1) -> torch.Tensor:
    """
    Shortcut for unsqueeze + expand
    """
    new_size = [-1] * (input_tensor.dim() + 1)
    new_size[dim] = num
    return input_tensor.unsqueeze(dim=dim).expand(new_size)


def expand_gather(input_tensor: torch.Tensor, dim: int, index: torch.Tensor, expand_dim=0, num=1) -> torch.Tensor:
    expanded_index = expand(index, dim=expand_dim, num=num)
    return input_tensor.gather(dim=dim, index=expanded_index)


def get_dup_keys(input_tensor: torch.Tensor, rounds=0) -> torch.Tensor:
    sorted_flat_key, flat_key_indice = torch.sort(input_tensor, dim=-1)
    # [batch * head, length, bucket_length * 2 * rounds]
    count_shift_keys = torch.ones_like(sorted_flat_key)
    # [batch * head, length, bucket_length * 2 * rounds]
    for i in range(1, rounds):
        equiv_flat_key = (sorted_flat_key[..., i:] == sorted_flat_key[..., :-i]).int()
        count_shift_keys[..., i:] += equiv_flat_key
        count_shift_keys[..., :-i] += equiv_flat_key
    count_key_indice = reverse_sort(flat_key_indice, dim=2)
    # [batch * head, length, bucket_length * 2 * rounds]
    return torch.gather(count_shift_keys, dim=-1, index=count_key_indice)


def top_p_sample(prob: torch.Tensor, perc=0.5) -> np.array:
    sorted_prob, sorted_indices = torch.sort(prob, dim=-1, descending=True)
    cumsum = torch.cumsum(sorted_prob, dim=-1)
    mask = cumsum < perc
    one_more_indice = mask.long().sum(dim=-1, keepdim=True)
    mask.scatter_(dim=-1, index=one_more_indice, value=True)
    sorted_prob.masked_fill_(~mask, value=0.0)
    masked_prob = sorted_prob.gather(dim=-1, index=reverse_sort(sorted_indices, dim=-1))
    return torch.multinomial(masked_prob, num_samples=1)


class LocalitySensitiveHash(nn.Module):
    """
    Implements Locality Sensitive Hash
    class is used to save random matrix used for hashing
    """

    def __init__(self, d_model, n_heads, rounds):
        super(LocalitySensitiveHash, self).__init__()
        self.d_k = d_model // n_heads
        self.rounds = rounds
        self.rand_matrix = None

    def forward(self, inp: torch.Tensor, n_buckets=0, random=True):
        # size of input tensor: [batch * head // chunk, length, d_k] 应该是这样才合理
        batch_size = inp.size(0)
        length = inp.size(1)
        inp = F.normalize(inp, p=2, dim=-1)  # 按照某个维度计算范数，p表示计算p范数（等于2就是2范数），dim计算范数的维度
        # [batch * head, length, d_k]
        if random:
            self.rand_matrix = torch.randn([batch_size, self.d_k, self.rounds, n_buckets // 2], device=inp.get_device())
            # [batch * head, d_k, rounds, n_buckets // 2]
            self.rand_matrix /= torch.norm(self.rand_matrix, dim=1, keepdim=True)
            # [batch * head, d_k, rounds, n_buckets // 2]
        matmul = torch.einsum('...ij,...jkl->...ikl', inp, self.rand_matrix)
        # [batch * head, length, rounds, n_buckets // 2]
        hashes = torch.argmax(torch.cat([matmul, -matmul], dim=-1), dim=-1).int()  # paper: h(x) = arg max([xR;-xR])
        # [batch * head, length, rounds]
        arange = hashes.new_empty((1, length, 1))
        # [1, length, 1]
        hashes = hashes * length + torch.arange(length, out=arange).expand_as(hashes)
        # 这里是用了P-stable hash 的假设和公式吗（参考：http://www.cppblog.com/humanchao/archive/2018/02/24/215521.html）
        # [batch * head, length, rounds]
        return hashes


class LSHAttention(nn.Module):
    """
    Implements LSHAttention
    class is used to save LocalitySensitiveHash
    """

    def __init__(self, d_model, n_heads, rounds, bucket_length, dropout_prob):
        super(LSHAttention, self).__init__()
        self.d_k = d_model // n_heads
        self.rounds = rounds
        self.bucket_length = bucket_length
        self.dropout = dropout_prob
        self.lsh = LocalitySensitiveHash(d_model, n_heads, rounds)

    def forward(self, query, value, seed, random=True):
        # size of query and value: [batch * head // chunk, length, d_k]
        # 以下注释应该均将batch * head // chunk视为batch * head！
        length = query.size(1)
        n_buckets = length // self.bucket_length

        sorted_hashes, hash_indice = torch.sort(self.lsh(query, n_buckets, random), dim=1)
        # [batch * head, length, rounds]
        original_indice = reverse_sort(hash_indice, dim=1)
        # [batch * head, length, rounds]

        reordered_query = expand_gather(
            expand(query, dim=3, num=self.rounds), dim=1,
            index=hash_indice, expand_dim=2, num=self.d_k
        )
        # [batch * head, length, d_k, rounds]
        reordered_query = reordered_query.reshape(
            -1, n_buckets, self.bucket_length, self.d_k, self.rounds
        )
        # [batch * head, n_buckets, bucket_length, d_k, rounds]
        lookback_key = F.normalize(look_back(reordered_query), p=2, dim=-2)
        # [batch * head, n_buckets, bucket_length * 2, d_k, rounds]
        matmul_qk = torch.einsum(
            '...ijk,...ljk->...ilk', reordered_query, lookback_key
        ) / math.sqrt(self.d_k)
        # [batch * head, n_buckets, bucket_length, bucket_length * 2, rounds]

        sorted_hashes = sorted_hashes.reshape(
            -1, n_buckets, self.bucket_length, self.rounds
        ) // length
        # [batch * head, n_buckets, bucket_length, rounds]
        matmul_qk.masked_fill_(
            mask=(sorted_hashes[..., None, :] != look_back(sorted_hashes)[..., None, :, :]),
            value=-1e9
        )

        query_indice = hash_indice.reshape(
            -1, n_buckets, self.bucket_length, self.rounds
        ).int()
        # [batch * head, n_buckets, bucket_length, rounds]
        key_indice = look_back(query_indice)
        # [batch * head, n_buckets, bucket_length * 2, rounds]
        matmul_qk.masked_fill_(
            mask=(query_indice[..., None, :] < key_indice[..., None, :, :]), value=-1e9
        )
        matmul_qk.masked_fill_(
            mask=(query_indice[..., None, :] == key_indice[..., None, :, :]), value=-1e5
        )

        key_indice = expand(key_indice, dim=2, num=self.bucket_length).flatten(1, 2)
        # [batch * head, length, bucket_length * 2, rounds]
        key_indice = expand_gather(
            key_indice,
            dim=1, index=original_indice,
            expand_dim=2, num=self.bucket_length * 2
        )
        # [batch * head, length, bucket_length * 2, rounds]
        count_key = get_dup_keys(
            key_indice.flatten(-2, -1), self.rounds
        ).reshape(-1, length, self.bucket_length * 2, self.rounds)
        # [batch * head, length, bucket_length * 2, rounds]
        count_key = expand_gather(
            count_key, dim=1, index=hash_indice, expand_dim=2, num=self.bucket_length * 2
        )
        # [batch * head, length, bucket_length * 2, rounds]
        matmul_qk = matmul_qk.flatten(1, 2)
        # [batch * head, length, bucket_length * 2, rounds]
        logsumexp_qk = torch.logsumexp(matmul_qk, dim=2)
        # [batch * head, length, rounds]
        softmax_qk = torch.exp(matmul_qk - count_key.float().log_() - logsumexp_qk[..., None, :])
        # [batch * head, length, bucket_length * 2, rounds]

        if self.training:
            softmax_qk = deterministic_dropout(softmax_qk, seed=seed, dropout=self.dropout)
            # [batch * head, length, bucket_length * 2, rounds]

        reordered_value = expand_gather(
            expand(value, dim=3, num=self.rounds), dim=1,
            index=hash_indice, expand_dim=2, num=self.d_k
        )
        # [batch * head, length, d_k, rounds]
        reordered_value = reordered_value.reshape(
            -1, n_buckets, self.bucket_length, self.d_k, self.rounds
        )
        # [batch * head, n_buckets, bucket_length, d_k, rounds]

        softmax_qk = softmax_qk.reshape(
            -1, n_buckets, self.bucket_length, self.bucket_length * 2, self.rounds
        )
        # [batch * head, n_buckets, bucket_length, bucket_length * 2, rounds]

        attention = torch.einsum('...ijl,...jkl->...ikl', softmax_qk, look_back(reordered_value))
        # [batch * head, n_buckets, bucket_length, d_k, rounds]
        attention = attention.flatten(1, 2)
        # [batch * head, length, d_k, rounds]
        attention = expand_gather(
            attention, dim=1, index=original_indice, expand_dim=2, num=self.d_k
        )
        # [batch * head, length, d_k, rounds]
        logsumexp_qk = torch.gather(logsumexp_qk, dim=1, index=original_indice)
        # [batch * head, length, rounds]
        logsumexp_qk = F.softmax(logsumexp_qk, dim=1)
        # [batch * head, length, rounds]
        attention = torch.einsum('...ij,...j->...i', attention, logsumexp_qk)
        # [batch * head, length, d_k]

        return attention


class MultiRoundLSHAttention(nn.Module):
    """
    Implements Multi Round LSH Attention
    class is defined to save LSHAttention
    """

    def __init__(self, d_model, n_heads, n_chunk, rounds, bucket_length, dropout_prob):
        super(MultiRoundLSHAttention, self).__init__()
        self.d_k = d_model // n_heads
        self.head = n_heads
        self.chunk = n_chunk
        self.linear_query = nn.Linear(d_model, d_model)
        self.linear_value = nn.Linear(d_model, d_model)
        self.linear_out = nn.Linear(d_model, d_model)
        self.lshattention = LSHAttention(d_model, n_heads, rounds, bucket_length, dropout_prob)

    def forward(self, input_tensor, seed, random=True):
        # input_tensor: [batch, head, d_model]
        length = input_tensor.size(1)

        query = self.linear_query(input_tensor).reshape(-1, length, self.head, self.d_k).transpose_(1, 2)
        # [batch, head, length, d_k]
        value = self.linear_value(input_tensor).reshape(-1, length, self.head, self.d_k).transpose_(1, 2)
        # [batch, head, length, d_k]

        chunked_query = torch.chunk(query.flatten(0, 1), chunks=self.chunk, dim=0)  # flatten: 合并或推平指定维度
        # [batch * head // chunk, length, d_k]
        chunked_value = torch.chunk(value.flatten(0, 1), chunks=self.chunk, dim=0)
        # [batch * head // chunk, length, d_k]  -> 这里指的是每一chunk的维度

        attention = torch.cat([
            self.lshattention(q, v, seed + i, random) for q, v, i
            in zip(chunked_query, chunked_value, range(self.chunk))
        ], dim=0).reshape(-1, self.head, length, self.d_k)
        # [batch, head, length, d_k]

        attention = attention.transpose(1, 2).flatten(-2, -1)
        # [batch, length, d_model]

        return self.linear_out(attention)


class Block(nn.Module):
    def __init__(self, d_model, dropout_prob, func):
        super(Block, self).__init__()
        self.func = func
        self.norm = nn.LayerNorm(d_model)
        self.dropout = dropout_prob

    def forward(self, x, seed, random=True):
        norm = self.norm(x)
        out = self.func(norm, (1 << 63) - seed, random)

        if self.training:
            return deterministic_dropout(out, seed=seed, dropout=self.dropout)

        return out


class ChunkFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout_prob):
        super(ChunkFeedForward, self).__init__()
        self.chunk = d_ff // d_model
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = dropout_prob

    def forward(self, input_tensor, seed, random=True):
        # [batch, length, d_model]
        chunks = torch.chunk(input_tensor, chunks=self.chunk, dim=1)
        # [batch, length // chunk, d_model]
        output = [F.gelu(self.linear1(chunk)) for chunk in chunks]
        # [batch, length // chunk, d_ff]
        if self.training:
            output = [
                deterministic_dropout(chunk, seed + i, dropout=self.dropout)
                for chunk, i in zip(output, range(self.chunk))]
            # [batch, length // chunk, d_ff]

        output = torch.cat([self.linear2(chunk) for chunk in output], dim=1)
        # [batch, length, d_model]
        return output


class Reversible(Function):

    def __init__(self):
        super(Reversible, self).__init__()

    @staticmethod
    def forward(ctx, *args):
        layer, input_1, input_2 = args
        ctx.layer = layer
        with torch.no_grad():
            output_1, output_2 = layer(input_1, input_2)
        Reversible.outputs = (output_1.detach(), output_2.detach())
        return output_1, output_2

    @staticmethod
    def backward(ctx, *grad_outputs):
        output_1_grad, output_2_grad = grad_outputs
        output_1, output_2 = Reversible.outputs
        output_1.requires_grad = True
        output_2.requires_grad = True

        with torch.enable_grad():
            g_output_1 = ctx.layer.g_block(output_1, ctx.layer.g_seed)
            g_output_1.backward(output_2_grad)

        with torch.no_grad():
            input_2 = output_2 - g_output_1
            del output_2, g_output_1
            input_1_grad = output_1_grad + output_1.grad
            del output_1_grad
            output_1.grad = None

        with torch.enable_grad():
            input_2.requires_grad = True
            f_input_2 = ctx.layer.f_block(input_2, ctx.layer.f_seed, False)
            f_input_2.backward(input_1_grad)

        with torch.no_grad():
            input_1 = output_1 - f_input_2
            del output_1, f_input_2
            input_2_grad = output_2_grad + input_2.grad
            del output_2_grad
            input_2.grad = None

            Reversible.outputs = (input_1.detach(), input_2.detach())

        return None, input_1_grad, input_2_grad, None


class ReversibleDecoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, n_heads, n_chunk, rounds, bucket_length, dropout_prob):
        super(ReversibleDecoderLayer, self).__init__()
        self.attn = MultiRoundLSHAttention(d_model, n_heads, n_chunk, rounds, bucket_length, dropout_prob)
        self.feed_forward = ChunkFeedForward(d_model, d_ff, dropout_prob)
        self.f_block = Block(d_model, dropout_prob, self.attn)
        self.g_block = Block(d_model, dropout_prob, self.feed_forward)

    def forward(self, x1, x2):
        y1 = x1 + self.f_block(x2, self.f_seed)
        y2 = x2 + self.g_block(y1, self.g_seed)
        return y1, y2


# 其实为decoder
class Reformer(nn.Module):
    def __init__(self, d_model, d_ff, n_heads, n_chunk, rounds, bucket_length, n_layer, dropout_prob):
        super(Reformer, self).__init__()
        self.layers = nn.ModuleList([ReversibleDecoderLayer(d_model, d_ff, n_heads, n_chunk, rounds, bucket_length,
                                                            dropout_prob) for _ in range(n_layer)])

    def forward(self, x1, x2):
        for layer in self.layers:
            layer.f_seed = int(np.random.randint(0, 1 << 63, dtype=np.int64))
            layer.g_seed = int(np.random.randint(0, 1 << 63, dtype=np.int64))
            x1, x2 = Reversible.apply(layer, x1, x2)
        return x2
