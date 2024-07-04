
"""
-------------------------------File info-------------------------
% - File name: Base_model_define.py
% - Description:
% -
% - Input:
% - Output:  None
% - Calls: None
% - usage:
% - Version： V1.0
% - Last update: 2022-06-16
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



class FscilModel(nn.Module):

    def __init__(self, args, mode=None):
        super().__init__()

        self.mode = mode  # -
        self.args = args

        self.encoder = models.resnet18(pretrained=args.im_pretrain)

        self.encoder.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.num_features = self.encoder.fc.in_features

        self.encoder = nn.Sequential(*list(self.encoder.children())[:-1])

        self.fc = nn.Linear(self.num_features, (self.args.num_class - self.args.base_start_index), bias=False)

    def forward_metric(self, x):
        x = self.encode(x)
        if 'cos' in self.mode:

            x = F.linear(F.normalize(x, p=2, dim=-1), F.normalize(self.fc.weight, p=2, dim=-1))
            x = self.args.temperature * x

        elif 'dot' in self.mode:
            x = self.fc(x)

        return x

    def encode(self, x):

        x = self.encoder(x)

        x = x.squeeze(-1).squeeze(-1)

        return x

    def forward(self, input):
        if self.mode != 'encoder':
            input = self.forward_metric(input)
            return input
        elif self.mode == 'encoder':
            input = self.encode(input)
            return input
        else:
            raise ValueError('Unknown mode')

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

        if 'ft' in self.args.new_mode:  #
            self.update_fc_ft(new_fc, data, label, session)

    def update_fc_avg(self, data, label, class_list):
        new_fc = []
        for class_index in class_list:
            class_index = class_index -self.args.base_start_index
            data_index = (label == class_index).nonzero().squeeze(-1)
            embedding = data[data_index]
            proto = embedding.mean(0)
            new_fc.append(proto)

            self.fc.weight.data[class_index] = proto

        new_fc = torch.stack(new_fc, dim=0)
        return new_fc

    @property
    def proto(self):
        return self.proto_all[-1]

    @property
    def new_proto(self):
        return self.proto_all[-1]

    def add_classes(self, n_classes):
        self.proto_all.append(nn.Parameter(torch.zeros(n_classes, self.num_features)))

        self.cuda()


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
            # model.module.mode = 'encoder'
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

    model.fc.weight.data[:(args.base_class-args.base_start_index)] = proto_list

    return model
