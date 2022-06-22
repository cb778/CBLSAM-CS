import os
import numpy as np
import math
from abc import ABC
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import PositionlEncoding, get_pad_mask
import torch.nn.init as weight_init
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch import optim

class basicAtnModule(nn.Module, ABC):
    def __init__(self, factor, dropout=0.1):
        super(basicAtnModule, self).__init__()
        self.factor = factor
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, attn_mask):
        attn_weight = torch.matmul(q, k.transpose(2, 3)) / self.factor
        if attn_mask is not None:
            attn_weight = attn_weight.masked_fill(attn_mask == 0, -1e9)

        attn_weight = self.dropout(F.softmax(attn_weight, dim=-1))
        context = torch.matmul(attn_weight, v)
        return context


class MultiHeadAttention(nn.Module, ABC):
    def __init__(self, n_heads, d_model, dk, dv, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        self.dk = dk
        self.dv = dv
        self.w_q = nn.Linear(d_model, n_heads * dk)
        self.w_k = nn.Linear(d_model, n_heads * dk)
        self.w_v = nn.Linear(d_model, n_heads * dv)
        self.fc = nn.Linear(n_heads * dv, d_model)
        self.attention = basicAtnModule(dk ** -0.5)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q, k, v, padding_mask=None):
        n_heads, dk, dv = self.n_heads, self.dk, self.dv
        batch_size, seq_q, seq_k, seq_v = q.size(0), q.size(1), k.size(1), v.size(1)
        residual = q
        q = self.w_q(q).view(batch_size, seq_q, n_heads, dk)
        k = self.w_k(k).view(batch_size, seq_k, n_heads, dk)
        v = self.w_v(v).view(batch_size, seq_v, n_heads, dv)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        if padding_mask is not None:
            padding_mask = padding_mask.unsqueeze(1)
        context = self.attention(q, k, v, padding_mask)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_q, -1)
        context = self.dropout(context)
        output = context + residual
        return output

class FeedForward(nn.Module, ABC):
    def __init__(self, d_model, d_ffn, dropout=0.1):
        super(FeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ffn)
        self.w_2 = nn.Linear(d_ffn, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, x):
        x = self.layer_norm(x)
        residual = x
        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        output = x + residual
        return output

class EncoderLayer(nn.Module, ABC):
    def __init__(self, d_model, d_ffn, n_heads, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_heads, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = FeedForward(d_model, d_ffn, dropout=dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, enc_input, padding_mask=None):

        enc_input = self.layer_norm(enc_input)
        enc_output = self.slf_attn(enc_input, enc_input, enc_input, padding_mask=padding_mask)
        enc_output = self.pos_ffn(enc_output)
        return enc_output

class Encoder(nn.Module, ABC):
    def __init__(self, vocab_size, d_word_dim, n_layers, n_heads, d_k, d_v,
                 d_model, d_ffn, pad_idx, hidden_size, emb_size,dropout=0.1, n_position=100):
        super(Encoder, self).__init__()
        self.word_emb = nn.Embedding(vocab_size, d_word_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(input_size=d_word_dim,hidden_size=hidden_size,batch_first=True,bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.pad_idx = pad_idx
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, d_ffn, n_heads, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.linear = nn.Linear(d_model, d_model)

    def forward(self, inputs):
        padding_mask = get_pad_mask(inputs, pad_idx=self.pad_idx)
        inputs = self.word_emb(inputs)
        inputs = F.dropout(inputs, 0.25, self.training)
        hids, (h_n,c_n) = self.lstm(inputs)
        enc_output = self.dropout(hids)
        before = enc_output
        for enc_layer in self.layers:
            enc_output = enc_layer(enc_output, padding_mask)
        enc_output = before + enc_output
        enc_output = self.layer_norm(enc_output)
        return enc_output


class JointEncoderLayer(nn.Module, ABC):
    def __init__(self, d_model, d_ffn, n_heads, d_k, d_v, dropout=0.1):
        super(JointEncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_heads, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = FeedForward(d_model, d_ffn, dropout=dropout)

    def forward(self, repr1, repr2, padding_mask=None):
        enc_output = self.slf_attn(repr1, repr2, repr2, padding_mask=padding_mask)
        enc_output = self.pos_ffn(enc_output)
        return enc_output

class JointEncoder(nn.Module, ABC):
    def __init__(self, n_heads, d_k, d_v, d_model, d_ffn, dropout=0.1):
        super(JointEncoder, self).__init__()
        self.layer = JointEncoderLayer(d_model, d_ffn, n_heads, d_k, d_v)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.linear = nn.Linear(d_model, d_model)

    def forward(self, repr1, repr2):
        repr_output = self.layer(repr1, repr2, padding_mask=None)
        repr_output = self.linear(repr_output)
        repr_output = self.layer_norm(repr_output)
        return repr_output


class JointEmbedder(nn.Module, ABC):
    def __init__(self, config):
        super(JointEmbedder, self).__init__()
        self.conf = config
        self.margin = config["margin"]
        self.name_enc = Encoder(config['vocab_size'], config['d_word_dim'], config['n_layers'],
                                config['n_heads'], config['d_k'], config['d_v'], config['d_model'],
                                config['d_ffn'], config['pad_idx'],config['emb_size'],config['hidden_size'])

        self.api_enc = Encoder(config['vocab_size'], config['d_word_dim'], config['n_layers'],
                               config['n_heads'], config['d_k'], config['d_v'], config['d_model'],
                               config['d_ffn'], config['pad_idx'],config['emb_size'],config['hidden_size'])

        self.token_enc = Encoder(config['vocab_size'], config['d_word_dim'], config['n_layers'],
                                 config['n_heads'], config['d_k'], config['d_v'], config['d_model'],
                                 config['d_ffn'], config['pad_idx'],config['emb_size'],config['hidden_size'])

        self.desc_enc = Encoder(config['vocab_size'], config['d_word_dim'], config['n_layers'],
                                config['n_heads'], config['d_k'], config['d_v'], config['d_model'],
                                config['d_ffn'], config['pad_idx'],config['emb_size'],config['hidden_size'])

        self.joint_enc1 = JointEncoder(config['n_heads'], config['d_k'], config['d_v'], config['d_model'],
                                     config['d_ffn'])
        self.joint_enc2 = JointEncoder(config['n_heads'], config['d_k'], config['d_v'], config['d_model'],
                                       config['d_ffn'])

        self.fc_code = nn.Linear(config["d_model"], config["d_model"])
        self.fc_desc = nn.Linear(config["d_model"], config["d_model"])

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def code_encoding(self, name, api, token):
        name_repr = self.name_enc(name)
        api_repr = self.api_enc(api)
        token_repr = self.token_enc(token)
        name_atn_token = self.joint_enc1(name_repr, token_repr)  # name and tokens
        token_atn_name = self.joint_enc1(token_repr, name_repr)
        name_atn_token = self.fc_code(name_atn_token)
        token_atn_name = self.fc_code(token_atn_name)
        code_repr = torch.cat((name_atn_token, api_repr, token_atn_name), dim=1)
        return code_repr

    def description_encoding(self, desc):
        desc_repr = self.desc_enc(desc)
        return desc_repr

    def joint_encoding(self, repr1, repr2, repr3):
        code_repr = self.joint_enc1(repr1, repr2)
        desc_pos_repr = self.joint_enc2(repr2, repr1)
        desc_neg_repr = self.joint_enc2(repr3, repr1)
        code_repr = torch.mean(self.fc_code(code_repr), dim=1)
        desc_pos_repr = torch.mean(self.fc_desc(desc_pos_repr), dim=1)
        desc_neg_repr = torch.mean(self.fc_desc(desc_neg_repr), dim=1)
        return code_repr, desc_pos_repr, desc_neg_repr

    def cal_similarity(self, code, desc):
        assert self.conf['sim_measure'] in \
               ['cos', 'poly', 'euc', 'sigmoid', 'gesd', 'aesd'], "invalid similarity measure"

        if self.conf["sim_measure"] == "cos":
            return F.cosine_similarity(code, desc)
        elif self.conf["sim_measure"] == "poly":
            return (0.5 * torch.matmul(code, desc.t()).diag() + 1) ** 2
        elif self.conf['sim_measure'] == 'sigmoid':
            return torch.tanh(torch.matmul(code, desc.t()).diag() + 1)
        elif self.conf['sim_measure'] in ['euc', 'gesd', 'aesd']:
            euc_dist = torch.dist(code, desc, 2)  # or torch.norm(code_vec-desc_vec,2)
            euc_sim = 1 / (1 + euc_dist)
            if self.conf['sim_measure'] == 'euc': return euc_sim
            sigmoid_sim = torch.sigmoid(torch.matmul(code, desc.t()).diag() + 1)
            if self.conf['sim_measure'] == 'gesd':
                return euc_sim * sigmoid_sim
            elif self.conf['sim_measure'] == 'aesd':
                return 0.5 * (euc_sim + sigmoid_sim)

    def forward(self, name, api, token, desc_pos, desc_neg):
        code_repr = self.code_encoding(name, api, token)
        desc_pos_repr = self.description_encoding(desc_pos)
        desc_neg_repr = self.description_encoding(desc_neg)
        code_repr, desc_pos_repr, desc_neg_repr = self.joint_encoding(code_repr, desc_pos_repr, desc_neg_repr)
        pos_sim = self.cal_similarity(code_repr, desc_pos_repr)
        neg_sim = self.cal_similarity(code_repr, desc_neg_repr)
        loss = (self.margin-pos_sim+neg_sim).clamp(min=1e-6).mean()
        return loss
