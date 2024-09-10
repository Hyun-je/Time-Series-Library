import torch
import torch.nn as nn
import torch.nn.functional as F
from models.Autoformer import Model as Autoformer
from models.FEDformer import Model as FEDformer
from models.TimesNet import Model as TimesNet
from models.DLinear import Model as DLinear
import math
import numpy as np
import copy


class DownsampleConv(nn.Module):

    def __init__(self, in_channels, stride=2):
        super(DownsampleConv, self).__init__()
        self.conv = nn.Conv1d(in_channels, in_channels, kernel_size=stride, stride=stride)
        self.bn = nn.BatchNorm1d(in_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
    
class DownsampleAvg(nn.Module):

    def __init__(self, in_channels, stride=2):
        super(DownsampleAvg, self).__init__()
        self.avg = nn.AvgPool1d(kernel_size=stride, stride=stride)

    def forward(self, x):
        x = self.avg(x)
        return x
    
class DownsampleNearest(nn.Module):

    def __init__(self, in_channels, stride=2):
        super(DownsampleNearest, self).__init__()
        self.stride = stride

    def forward(self, x):
        return x[:,:,::self.stride]




class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()

        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len

        # self.downsample_1_to_2 = DownsampleNearest(in_channels=configs.enc_in, stride=2)
        # self.downsample_2_to_4 = DownsampleNearest(in_channels=configs.enc_in, stride=2)
        # self.downsample_4_to_8 = DownsampleNearest(in_channels=configs.enc_in, stride=2)
        
        configs.seq_len = configs.seq_len
        configs.label_len = configs.label_len
        configs.pred_len = configs.pred_len
        self.sub_model1 = Autoformer(configs)

        configs2 = copy.deepcopy(configs)
        configs2.seq_len = configs2.seq_len // 2
        configs2.label_len = configs2.label_len // 2
        configs2.pred_len = configs2.pred_len // 2
        self.sub_model2 = Autoformer(configs2)

        configs3 = copy.deepcopy(configs)
        configs3.seq_len = configs3.seq_len // 4
        configs3.label_len = configs3.label_len // 4
        configs3.pred_len = configs3.pred_len // 4
        self.sub_model3 = FEDformer(configs3)

        configs4 = copy.deepcopy(configs)
        configs4.seq_len = configs4.seq_len // 8
        configs4.label_len = configs4.label_len // 8
        configs4.pred_len = configs4.pred_len // 8
        self.sub_model4 = DLinear(configs4)


    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):

        B = x_enc.shape[0]  # batch size
        L = x_enc.shape[1]  # length of sequence
        C = x_enc.shape[2]  # number of channels

        # x1 = x_enc
        # x2 = self.downsample_1_to_2(x1.permute(0,2,1)).permute(0,2,1)
        # x4 = self.downsample_2_to_4(x2.permute(0,2,1)).permute(0,2,1)
        # x8 = self.downsample_4_to_8(x4.permute(0,2,1)).permute(0,2,1)

        x1 = x_enc
        x2 = x1[:, 1::2, :]
        x4 = x2[:, 1::2, :]
        x8 = x4[:, 1::2, :]
        
        x8_enc = self.sub_model4(x8, None, None, None, None)
        x4_enc = self.sub_model3(x4, None, None, None, None)
        x2_enc = self.sub_model2(x2, None, None, None, None)
        x1_enc = self.sub_model1(x1, None, None, None, None)

        return x1_enc, x2_enc, x4_enc, x8_enc