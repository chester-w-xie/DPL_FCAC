"""
-------------------------------File info-------------------------
% - File name: Meta_model_define.py
% - Description:
% - Input:
% - Output:  None
% - Calls: None
% - usage:
% - Version： V1.0
% - Last update: 2022-10-28
%  Copyright (C) PRMI, South China university of technology; 2022
%  ------For Educational and Academic Purposes Only ------
% - Author : Chester.Wei.Xie, PRMI, SCUT/ GXU
% - Contact: chester.w.xie@gmail.com
------------------------------------------------------------------
"""
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from utils import *
import math
from torch.nn import Parameter


class MYNET(nn.Module):

    def __init__(self, args, mode=None):
        super().__init__()

        self.mode = mode
        self.args = args

        self.encoder = models.resnet18(pretrained=False)
        self.encoder.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.num_features = self.encoder.fc.in_features

        self.encoder = nn.Sequential(*list(self.encoder.children())[:-1])

        self.fc = nn.Linear(self.num_features, (self.args.num_class - self.args.base_start_index), bias=False)

        hdim = self.num_features
        self.Selective_attn = SelectiveAttention(1, hdim, hdim, hdim, dropout=0.5)

    def forward(self, input):
        if self.mode == 'encoder':
            input = self.encode(input)
            return input
        else:
            support_idx, query_idx = input
            logits = self._forward(support_idx, query_idx)
            return logits

    def _forward(self, incremental_support, query, base_support):

        emb_dim = incremental_support.size(-1)

        incremental_proto = incremental_support.mean(dim=1)
        # [1, 30, 64]
        base_proto = base_support.mean(dim=1)  # [1, 15, 64]

        num_batch = incremental_proto.shape[0]
        num_incremental_proto = incremental_proto.shape[1]
        num_base_proto = base_proto.shape[1]
        num_query = query.shape[1] * query.shape[2]

        query = query.view(-1, emb_dim).unsqueeze(1)

        incremental_proto = incremental_proto.unsqueeze(1).expand(num_batch, num_query,
                                                                  num_incremental_proto, emb_dim).contiguous()

        incremental_proto = incremental_proto.view(num_batch * num_query, num_incremental_proto, emb_dim)

        base_proto = base_proto.unsqueeze(1).expand(num_batch, num_query, num_base_proto, emb_dim).contiguous()
        base_proto = base_proto.view(num_batch * num_query, num_base_proto, emb_dim)

        combined = torch.cat([incremental_proto, query], 1)  # Nk x (N + 1) x d, batch_size = NK

        combined = self.Selective_attn(combined, combined, combined, combined, base_proto, base_proto)

        incremental_proto, query = combined.split(num_incremental_proto, 1)

        logits = F.cosine_similarity(query, incremental_proto, dim=-1)

        logits = logits * self.args.temperature

        return logits

    def encode(self, x):
        x = self.encoder(x)

        x = x.squeeze(-1).squeeze(-1)
        return x

    def update_fc(self, dataloader, class_list, session):
        for batch in dataloader:
            data, label = [_.cuda() for _ in batch]
            data = self.encode(data).detach()

        if self.args.not_data_init:
            new_fc = nn.Parameter(
                torch.rand(len(class_list), self.num_features, device="cuda"),
                requires_grad=True)
            nn.init.kaiming_uniform_(new_fc, a=math.sqrt(5))
        else:
            new_fc = self.update_fc_avg(data, label, class_list)

        if 'ft' in self.args.new_mode:
            self.update_fc_ft(new_fc, data, label, session)

    def update_fc_avg(self, data, label, class_list):
        new_fc = []
        for class_index in class_list:
            class_index = class_index - self.args.base_start_index
            data_index = (label == class_index).nonzero().squeeze(-1)
            embedding = data[data_index]
            proto = embedding.mean(0)
            new_fc.append(proto)
            self.fc.weight.data[class_index] = proto
        new_fc = torch.stack(new_fc, dim=0)
        return new_fc


class ScaledDotProductAttention(nn.Module):
    """ Scaled Dot-Product Attention """

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v):

        attn = torch.bmm(q, k.transpose(1, 2))

        attn = attn / self.temperature
        log_attn = F.log_softmax(attn, 2)
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)
        return output, attn, log_attn


class SelectiveAttention(nn.Module):
    """ SelectiveAttention Attention module """

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):

        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_q = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_k = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_v = nn.Linear(d_model, n_head * d_v, bias=False)

        nn.init.normal_(self.w_q.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_k.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_v.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.self_attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.mutual_attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.selective_attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))

        self.fc_select = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc_select.weight)
        self.dropout_select = nn.Dropout(dropout)

        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, q_s, k_s, v_s, q_m, k_m, v_m):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b1, len_q1, _ = q_s.size()
        sz_b1, len_k1, _ = k_s.size()
        sz_b1, len_v1, _ = v_s.size()

        sz_b2, len_q2, _ = q_m.size()
        sz_b2, len_k2, _ = k_m.size()
        sz_b2, len_v2, _ = v_m.size()

        residual = q_s

        q_s = self.w_q(q_s).view(sz_b1, len_q1, n_head, d_k)
        k_s = self.w_k(k_s).view(sz_b1, len_k1, n_head, d_k)
        v_s = self.w_v(v_s).view(sz_b1, len_v1, n_head, d_v)

        q_m = self.w_q(q_m).view(sz_b2, len_q2, n_head, d_k)
        k_m = self.w_k(k_m).view(sz_b2, len_k2, n_head, d_k)
        v_m = self.w_v(v_m).view(sz_b2, len_v2, n_head, d_v)

        q_s = q_s.permute(2, 0, 1, 3).contiguous().view(-1, len_q1, d_k)  # (n*b) x lq x dk
        k_s = k_s.permute(2, 0, 1, 3).contiguous().view(-1, len_k1, d_k)  # (n*b) x lk x dk
        v_s = v_s.permute(2, 0, 1, 3).contiguous().view(-1, len_v1, d_v)  # (n*b) x lv x dv

        q_m = q_m.permute(2, 0, 1, 3).contiguous().view(-1, len_q2, d_k)  # (n*b) x lq x dk
        k_m = k_m.permute(2, 0, 1, 3).contiguous().view(-1, len_k2, d_k)  # (n*b) x lk x dk
        v_m = v_m.permute(2, 0, 1, 3).contiguous().view(-1, len_v2, d_v)  # (n*b) x lv x dv

        self_output, _, _ = self.self_attention(q_s, k_s, v_s)  # [num_head, num_samples, num_classes, fea_dim]
        self_output = self_output.view(n_head, sz_b1, len_q1, d_v)
        self_output = self_output.permute(1, 2, 0, 3).contiguous().view(sz_b1, len_q1, -1)  # b x lq x (n*dv)
        # [num_samples, num_classes, num_head, fea_dim] -> [num_samples, num_classes, num_head*fea_dim]

        mutual_output, _, _ = self.mutual_attention(q_m, k_m, v_m)  # [num_head, num_samples, num_classes, fea_dim]
        mutual_output = mutual_output.view(n_head, sz_b2, len_q2, d_v)
        mutual_output = mutual_output.permute(1, 2, 0, 3).contiguous().view(sz_b2, len_q2, -1)  # b x lq x (n*dv)
        # [num_samples, num_classes, num_head, fea_dim] -> [num_samples, num_classes, num_head*fea_dim]

        sz_b3, len_q3, _ = self_output.size()
        sz_b3, len_k3, _ = mutual_output.size()
        sz_b3, len_v3, _ = self_output.size()

        q_select = self.w_q(self_output).view(sz_b3, len_q3, n_head, d_k)
        k_select = self.w_k(mutual_output).view(sz_b3, len_k3, n_head, d_k)
        v_select = self.w_v(self_output).view(sz_b3, len_v3, n_head, d_v)

        q_select = q_select.permute(2, 0, 1, 3).contiguous().view(-1, len_q3, d_k)  # (n*b) x lq x dk
        k_select = k_select.permute(2, 0, 1, 3).contiguous().view(-1, len_k3, d_k)  # (n*b) x lk x dk
        v_select = v_select.permute(2, 0, 1, 3).contiguous().view(-1, len_v3, d_v)  # (n*b) x lv x dv

        selective_output, _, _ = self.selective_attention(q_select, k_select, v_select)
        selective_output = self.dropout_select(self.fc_select(selective_output))

        output = self.layer_norm(selective_output + residual)  # [num_samples, num_classes, fea_dim]

        return output


def replace_base_fc(trainset, model, args):
    # replace fc.weight with the embedding average of train data
    model = model.eval()

    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=128,
                                              num_workers=8, pin_memory=True, shuffle=False)
    # trainloader.dataset.transform = transform
    embedding_list = []
    label_list = []
    # data_list=[]
    with torch.no_grad():
        for i, batch in enumerate(trainloader):
            data, label = [_.cuda() for _ in batch]
            model.mode = 'encoder'
            embedding = model(data)

            embedding_list.append(embedding.cpu())
            label_list.append(label.cpu())
    embedding_list = torch.cat(embedding_list, dim=0)
    label_list = torch.cat(label_list, dim=0)

    proto_list = []

    for class_index in range(args.base_class):
        class_index = class_index - args.base_start_index
        data_index = (label_list == class_index).nonzero()
        embedding_this = embedding_list[data_index.squeeze(-1)]
        embedding_this = embedding_this.mean(0)
        proto_list.append(embedding_this)

    proto_list = torch.stack(proto_list, dim=0)

    model.fc.weight.data[:args.base_class] = proto_list

    return model
