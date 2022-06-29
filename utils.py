from abc import ABC
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import math


class PositionlEncoding(nn.Module, ABC):
    def __init__(self, d_hid, n_position=100):
        super(PositionlEncoding, self).__init__()
        self.register_buffer("pos_table", self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]
        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])
        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return x + self.pos_table[:, :x.size(1)].clone().detach()


def get_pad_mask(seq, pad_idx):
    return (seq != pad_idx).unsqueeze(-2)


def normalize(data):
    return data / np.linalg.norm(data, axis=1, keepdims=True)


def validate(valid_set, model, K, sim_measure):
    def recall(gold, prediction, results):
        sum = 0.
        for val in gold:
            try:
                index = prediction.index(val)
            except ValueError:
                index = -1
            if index <= results:
                sum += 1
        return sum / float(len(gold))

    def acc(gold, prediction):
        sum = 0.
        for val in gold:
            try:
                index = prediction.index(val)
            except ValueError:
                index = -1
            if index != -1:
                sum += 1
        return sum / float(len(gold))

    def map(gold, prediction):
        sum = 0.
        for idx, val in enumerate(gold):
            try:
                index = prediction.index(val)
            except ValueError:
                index = -1
            if index != -1:
                sum = sum + (idx + 1) / float(index + 1)
        return sum / float(len(gold))

    def mrr(gold, prediction):
        sum = 0.
        for val in gold:
            try:
                index = prediction.index(val)
            except ValueError:
                index = -1
            if index != -1:
                sum = sum + 1.0 / float(index + 1)
        return sum / float(len(gold))

    def ndcg(gold, prediction):
        dcg = 0.
        idcgs = idcg(len(gold))
        for i, predictItem in enumerate(prediction):
            if predictItem in real:
                item_relevance = 1
                rank = i + 1
                dcg += (math.pow(2, item_relevance) - 1.0) * (math.log(2) / math.log(rank + 1))
        return dcg / float(idcgs)

    def idcg(n):
        idcg = 0
        item_relevance = 1
        for i in range(n):
            idcg += (math.pow(2, item_relevance) - 1.0) * (math.log(2) / math.log(i + 2))
        return idcg

    model.eval()
    data_loader = torch.utils.data.DataLoader(dataset=valid_set, batch_size=10000, shuffle=True, drop_last=True, num_workers=12)
    device = next(model.parameters()).device
    
    re, accu, mrrs, maps, ndcgs = 0., 0., 0., 0., 0.
    code_reprs, desc_reprs = [], []
    for batch in data_loader:
        # batch ===> names, name_len, apis, api_len, toks, tok_len, descs, desc_len, bad_descs, bad_desc_len
        if len(batch) == 10:
            if device == torch.device("cuda:1"):
                code_batch = [tensor.cuda(1) for tensor in batch[:6]][0::2]
                desc_batch = [tensor.cuda(1) for tensor in batch[6:8]][0::2]
            else:
                code_batch = [tensor.cuda() for tensor in batch[:6]][0::2]
                desc_batch = [tensor.cuda() for tensor in batch[6:8]][0::2]
        with torch.no_grad():
            code_repr = model.code_encoding(*code_batch)
            desc_repr = model.description_encoding(*desc_batch)
            code_repr, desc_repr, _ = model.joint_encoding(code_repr, desc_repr, desc_repr)
            code_repr = code_repr.data.cpu().numpy().astype(np.float32)
            desc_repr = desc_repr.data.cpu().numpy().astype(np.float32)

        code_reprs.append(code_repr)
        desc_reprs.append(desc_repr)

    code_reprs, desc_reprs = np.vstack(code_reprs), np.vstack(desc_reprs)

    data_len = len(code_reprs)
    for i in tqdm(range(data_len), desc="-------- Eval --------"):

        desc_vec = np.expand_dims(desc_repr[i], axis=0)  # [1 x dim]
        n_results = K
        if sim_measure == 'cos':
            dot = np.dot(code_reprs, desc_vec.T)[:, 0]
            sum = (np.linalg.norm(code_reprs, axis=1) * np.linalg.norm(desc_vec))
            sims = dot / sum

        neg_sims = np.negative(sims)
        predict_origin = np.argsort(neg_sims)
        predict = predict_origin[:n_results]
        predict = [int(k) for k in predict]
        predict_origin = [int(k) for k in predict_origin]
        real = [i]

        re += recall(real, predict_origin, n_results)
        accu += acc(real, predict)
        mrrs += mrr(real, predict)
        maps += map(real, predict)
        ndcgs += ndcg(real, predict)

    re = re / float(data_len)
    accu = accu / float(data_len)
    mrrs = mrrs / float(data_len)
    maps = maps / float(data_len)
    ndcgs = ndcgs / float(data_len)

    return re, accu, mrrs, maps, ndcgs
