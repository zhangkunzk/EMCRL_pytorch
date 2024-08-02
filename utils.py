"""
@Author: zhkun
@Time:  21:33
@File: utils
@Description: part of codes are drawn from simcse,
@Something to attention
"""
import contextlib

import prettytable
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.metrics.pairwise import cosine_similarity

import math
import time
import numpy as np


class Pooler(nn.Module):
    """
    Parameter-free poolers to get the sentence embedding
    'cls': [CLS] representation with BERT/RoBERTa's MLP pooler.
    'cls_before_pooler': [CLS] representation without the original MLP pooler.
    'avg': average of the last layers' hidden states at each token.
    'avg_top2': average of the last two layers.
    'avg_first_last': average of the first and the last layers.
    """

    def __init__(self, pooler_type):
        super().__init__()
        self.pooler_type = pooler_type
        assert self.pooler_type in ["cls", "cls_before_pooler", "avg", "avg_topn",
                                    "avg_first_last", 'first_last', "last_topn"], "unrecognized pooling type %s" % self.pooler_type

    def forward(self, outputs, attention_mask, topn=2):
        last_hidden = outputs.last_hidden_state
        pooler_output = outputs.pooler_output
        hidden_states = outputs.hidden_states

        if self.pooler_type in ['cls_before_pooler', 'cls']:
            return last_hidden[:, 0]
        elif self.pooler_type == "avg":
            return ((last_hidden * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1))
        elif self.pooler_type == "avg_first_last":
            first_hidden = hidden_states[0]
            last_hidden = hidden_states[-1]
            pooled_result = ((first_hidden + last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(
                1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        elif self.pooler_type == 'first_last':
            first_hidden = hidden_states[0]
            last_hidden = hidden_states[-1]
            pooler_output = torch.stack([first_hidden, last_hidden], dim=2)
            return pooler_output
        elif self.pooler_type == "last_topn":
            # second_last_hidden = hidden_states[-2]
            # last_hidden = hidden_states[-1]
            # pooled_result = ((last_hidden + second_last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(
            #     1) / attention_mask.sum(-1).unsqueeze(-1)
            pooled_result = torch.stack(hidden_states[-topn:], dim=2)
            # pooled_result = (results * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        else:
            raise NotImplementedError


class MLPLayer(nn.Module):
    """
    Head for getting sentence representations over RoBERTa/BERT's CLS representation.
    """

    def __init__(self, input_size, output_size):
        super().__init__()
        self.dense = nn.Linear(input_size, output_size)
        self.activation = nn.Tanh()

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = self.activation(x)

        return x


class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp


class ProjectionHead(nn.Module):
    def __init__(self, emb_size, head_size):
        super(ProjectionHead, self).__init__()
        self.hidden = nn.Linear(emb_size, emb_size)
        self.bn = nn.BatchNorm1d(emb_size)
        self.out = nn.Linear(emb_size, head_size)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        h = self.hidden(h)
        h = self.bn(h)
        h = F.relu_(h)
        h = self.out(h)
        return h


class ResProjectionHead(nn.Module):
    def __init__(self, emb_size, head_size):
        super(ResProjectionHead, self).__init__()
        self.hidden = nn.Linear(emb_size, emb_size)
        self.bn = nn.BatchNorm1d(emb_size)
        self.out = nn.Linear(emb_size, head_size)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        h0 = h
        h = self.hidden(h)
        h = self.bn(h)
        h = F.relu_(h + h0)
        h = self.out(h)
        return h


class ProjectionHeadV2(nn.Module):
    def __init__(self, emb_size, head_size):
        super(ProjectionHeadV2, self).__init__()
        self.hidden = nn.Sequential(
            nn.Linear(in_features=emb_size, out_features=emb_size),
            nn.BatchNorm1d(emb_size),
            nn.ReLU(),
            nn.Linear(in_features=emb_size, out_features=emb_size),
            nn.BatchNorm1d(emb_size),
        )
        self.out = nn.Linear(emb_size, head_size)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        h0 = h
        h = self.hidden(h)
        return h


class ResProjectionHeadV2(nn.Module):
    def __init__(self, emb_size, head_size):
        super(ResProjectionHeadV2, self).__init__()
        self.hidden = nn.Sequential(
            nn.Linear(in_features=emb_size, out_features=emb_size),
            # nn.Tanh(),
            nn.ReLU(),
            nn.Linear(in_features=emb_size, out_features=emb_size)
        )
        self.bn = nn.BatchNorm1d(emb_size)
        self.out = nn.Linear(emb_size, head_size)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        h0 = h
        h = self.hidden(h)
        h = self.bn(h)
        h = F.relu_(h + h0)
        h = self.out(h)
        return h


def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, t1=0.3, eps=1e-10) -> torch.Tensor:
    batch_size = z1.shape[0]
    assert batch_size == z2.shape[0]
    assert batch_size > 1

    # compute the similarity matrix
    # values in the diagonal elements represent the similarity between the (POS, POS) pairs
    # while the other values are the similarity between the (POS, NEG) pairs
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    # matrix multiplication
    sim_mat = z1 @ z2.T
    scaled_prob_mat = F.softmax(sim_mat / t1, dim=1)

    # construct a cross-entropy loss to maximize the probability of the (POS, POS) pairs
    log_prob = torch.log(scaled_prob_mat + eps)
    return -torch.diagonal(log_prob).mean()


def nt_cl_loss(z1: torch.Tensor, z2: torch.Tensor, loss_func, t1=0.3, device='CPU') -> torch.Tensor:
    batch_size = z1.shape[0]
    assert batch_size == z2.shape[0]
    assert batch_size > 1

    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    # matrix multiplication
    sim_mat = z1 @ z2.T
    scaled_prob_mat = sim_mat / t1

    labels = torch.arange(batch_size).long().to(device)

    loss = loss_func(scaled_prob_mat, labels)

    return loss


def nt_em_loss(z1: torch.Tensor, z2: torch.Tensor, t1=0.3, t2=1.2, noise_prob=1e-5, eps=1e-10) -> torch.Tensor:
    batch_size = z1.shape[0]
    assert batch_size == z2.shape[0]
    assert batch_size > 1

    # compute the similarity matrix
    # values in the diagonal elements represent the similarity between the (POS, POS) pairs
    # while the other values are the similarity between the (POS, NEG) pairs
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    sim_mat = z1 @ z2.T

    prob_mat = F.softmax(sim_mat * t2, 1)
    # 噪声
    flip_mat = torch.lt(torch.rand((batch_size, batch_size)).to(prob_mat.device), noise_prob).float()
    mask = torch.eye(batch_size).to(prob_mat.device)
    mask = torch.clamp(mask + prob_mat + flip_mat, min=0.0, max=1.0)  # keep diagonal elements as 1
    mask = mask.detach()

    scaled_prob_mat = F.softmax(sim_mat / t1, dim=1)
    log_likelihood = mask * torch.log(scaled_prob_mat + eps) + (1.0 - mask) * torch.log(1.0 - scaled_prob_mat + eps)
    log_likelihood = log_likelihood.sum(dim=1)
    return -log_likelihood.mean()


def make_adv(model: nn.Module,
             x1: torch.Tensor,
             x2: torch.Tensor,
             xi: float = 10.0,
             eps: float = 2.0,
             ip: int = 1):
    with torch.no_grad():
        h = model(x1).detach()
    d = _l2_normalize(torch.rand(x2.shape).sub(0.5).to(x2.device))
    with _disable_tracking_bn_stats(model):
        for _ in range(ip):
            d.requires_grad_()
            h_hat = model(x2 + xi * d)
            adv_distance = -(F.normalize(h_hat, dim=1) * F.normalize(h, dim=1)).sum(dim=1).mean()
            adv_distance.backward()
            d = _l2_normalize(d + d.grad)
            model.zero_grad()
    return x2 + eps * d.detach()


@contextlib.contextmanager
def _disable_tracking_bn_stats(model):
    def switch_attr(m):
        if hasattr(m, 'track_running_stats'):
            m.track_running_stats ^= True

    model.apply(switch_attr)
    yield
    model.apply(switch_attr)


def _l2_normalize(d):
    d_reshaped = d.view(d.shape[0], -1, *(1 for _ in range(d.dim() - 2)))
    d /= torch.norm(d_reshaped, dim=1, keepdim=True) + 1e-8
    return d


class CosineDecay(object):

    def __init__(self,
                 max_value,
                 min_value,
                 num_loops):
        self._max_value = max_value
        self._min_value = min_value
        self._num_loops = num_loops

    def get_value(self, i):
        if i < 0:
            i = 0
        if i >= self._num_loops:
            i = self._num_loops - 1
        value = (math.cos(i * math.pi / self._num_loops) + 1.0) * 0.5
        value = value * (self._max_value - self._min_value) + self._min_value
        return value


class CosineWarmUp(object):

    def __init__(self,
                 max_value,
                 min_value,
                 num_loops):
        self._max_value = max_value
        self._min_value = min_value
        self._num_loops = num_loops

    def get_value(self, i):
        if i < 0:
            i = 0
        if i >= self._num_loops:
            i = self._num_loops - 1
        i = i - self._num_loops + 1
        value = (math.cos(i * math.pi / self._num_loops) + 1.0) * 0.5
        value = value * (self._max_value - self._min_value) + self._min_value
        return value


class CosineWarmUpDecay(object):

    def __init__(self,
                 max_value,
                 min_value,
                 num_loops,
                 warm_up=0.05):
        self._max_value = max_value
        self._min_value = min_value
        self._num_loops = num_loops
        self._warm_up_p = warm_up
        self._warm_up = CosineWarmUp(max_value, min_value, int(num_loops * warm_up))
        self._decay = CosineDecay(max_value, min_value, int(num_loops * (1.0 - warm_up)))

    def get_value(self, i):
        if i < self._num_loops * self._warm_up_p:
            return self._warm_up.get_value(i)
        else:
            return self._decay.get_value(i)


def get_current_time():
    return str(time.strftime('%Y%m%d%H%M%S', time.localtime(time.time())))[:-2]


def calc_eplased_time_since(start_time):
    curret_time = time.time()
    seconds = int(curret_time - start_time)
    hours = seconds // 3600
    seconds = seconds % 3600
    minutes = seconds // 60
    seconds = seconds % 60
    time_str = '{:0>2d}h{:0>2d}min{:0>2d}s'.format(hours, minutes, seconds)
    return time_str


def eval_map_mrr(qids, aids, preds, labels):
    # 衡量map指标和mrr指标
    dic = dict()
    pre_dic = dict()
    for qid, aid, pred, label in zip(qids, aids, preds, labels):
        pre_dic.setdefault(qid, [])
        pre_dic[qid].append([aid, pred, label])
    for qid in pre_dic:
        dic[qid] = sorted(pre_dic[qid], key=lambda k: k[1], reverse=True)
        aid2rank = {aid: [label, rank] for (rank, (aid, pred, label)) in enumerate(dic[qid])}
        dic[qid] = aid2rank

    MAP = 0.0
    MRR = 0.0
    useful_q_len = 0
    for q_id in dic:
        sort_rank = sorted(dic[q_id].items(), key=lambda k: k[1][1], reverse=False)
        correct = 0
        total = 0
        AP = 0.0
        mrr_mark = False
        for i in range(len(sort_rank)):
            if sort_rank[i][1][0] == 1:
                correct += 1
        if correct == 0:
            continue
        useful_q_len += 1
        correct = 0
        for i in range(len(sort_rank)):
            # compute MRR
            if sort_rank[i][1][0] == 1 and mrr_mark == False:
                MRR += 1.0 / float(i + 1)
                mrr_mark = True
            # compute MAP
            total += 1
            if sort_rank[i][1][0] == 1:
                correct += 1
                AP += float(correct) / float(total)

        AP /= float(correct)
        MAP += AP

    MAP /= useful_q_len
    MRR /= useful_q_len
    return MAP, MRR


def write_file(file_name, content):
    with open(file_name, 'a+', encoding='utf8') as w:
        if isinstance(content, list):
            for item in content:
                w.write(item + '\n')
        elif isinstance(content, str):
            w.write(content + '\n')
        # elif isinstance(content, prettytable.PrettyTable):
        #     w.write(str(content))
        else:
            raise ValueError('unrecognizing data type, please try again')



class BiClassCalculator(object):
    def __init__(self):
        self._tp = 0
        self._tn = 0
        self._fp = 0
        self._fn = 0

    def update(self, label_predict, label_true):
        hit = np.equal(label_predict, label_true)
        hit = np.float32(hit)
        miss = 1.0 - hit

        pos = np.float32(label_predict)
        neg = 1.0 - pos

        self._tp += np.sum(hit * pos, keepdims=False)
        self._tn += np.sum(hit * neg, keepdims=False)
        self._fp += np.sum(miss * pos, keepdims=False)
        self._fn += np.sum(miss * neg, keepdims=False)

    @property
    def precision(self):
        num_pos_pred = self._tp + self._fp
        return self._tp / num_pos_pred if num_pos_pred > 0 else math.nan

    @property
    def recall(self):
        num_pos_true = self._tp + self._fn
        return self._tp / num_pos_true if num_pos_true > 0 else math.nan

    @property
    def f1(self):
        pre = self.precision
        rec = self.recall
        return 2 * (pre * rec) / (pre + rec)

    @property
    def accuracy(self):
        num_hit = self._tp + self._tn
        num_all = self._tp + self._tn + self._fp + self._fn
        return num_hit / num_all if num_all > 0 else math.nan


"""
x: augmentation1 [batch_size, hidden_dim]
y: augmentation2 [batch_size, hidden_dim]
"""
def align_loss(x, y, alpha=2):
    return (x - y).norm(p=2, dim=1).pow(alpha).mean()


def uniform_loss(x, t=2):
    return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()


def align_uniform_loss(z1: torch.Tensor, z2: torch.Tensor, alpha=2, t=2, align_weight=0.75, uniform_weight=0.5, loss_type='all') -> torch.Tensor:
    if loss_type == 'all':
        loss1 = align_loss(z1, z2, alpha)
        loss2 = uniform_loss(z1, t) + uniform_loss(z2, t)
        loss = align_weight*loss1 + uniform_weight * loss2 / 2
    elif loss_type == 'align':
        loss = align_loss(z1, z2, alpha)
    else:
        loss = (uniform_loss(z1, t) + uniform_loss(z2, t)) / 2

    return loss

