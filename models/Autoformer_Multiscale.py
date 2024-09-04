import torch
import torch.nn as nn
import torch.nn.functional as F
from models.Autoformer import Model as Autoformer
import math
import numpy as np


class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()

        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len

        self.downsample_1_to_2 = nn.Sequential(
            nn.Conv1d(configs.enc_in, configs.enc_in, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(configs.enc_in),
            nn.ReLU(),
        )
        self.downsample_2_to_4 = nn.Sequential(
            nn.Conv1d(configs.enc_in, configs.enc_in, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(configs.enc_in),
            nn.ReLU(),
        )
        self.downsample_4_to_8 = nn.Sequential(
            nn.Conv1d(configs.enc_in, configs.enc_in, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(configs.enc_in),
            nn.ReLU(),
        )
        
        configs.seq_len = configs.seq_len
        configs.label_len = configs.label_len
        configs.pred_len = configs.pred_len
        self.sub_model1 = Autoformer(configs)

        configs.seq_len = configs.seq_len // 2
        configs.label_len = configs.label_len // 2
        configs.pred_len = configs.pred_len // 2
        self.sub_model2 = Autoformer(configs)

        configs.seq_len = configs.seq_len // 4
        configs.label_len = configs.label_len // 4
        configs.pred_len = configs.pred_len // 4
        self.sub_model3 = Autoformer(configs)

        configs.seq_len = configs.seq_len // 8
        configs.label_len = configs.label_len // 8
        configs.pred_len = configs.pred_len // 8
        self.sub_model4 = Autoformer(configs)

        self.upsample_8_to_4 = nn.Sequential(
            nn.ConvTranspose1d(configs.enc_in, configs.c_out, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm1d(configs.c_out),
            nn.ReLU(),
        )
        self.upsample_4_to_2 = nn.Sequential(
            nn.ConvTranspose1d(configs.enc_in, configs.c_out, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm1d(configs.c_out),
            nn.ReLU(),
        )
        self.upsample_2_to_1 = nn.Sequential(
            nn.ConvTranspose1d(configs.enc_in, configs.c_out, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm1d(configs.c_out),
            nn.ReLU(),
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):

        x1 = x_enc
        x2 = self.downsample_1_to_2(x_enc.permute(0, 2, 1)).permute(0, 2, 1)
        x4 = self.downsample_2_to_4(x_enc.permute(0, 2, 1)).permute(0, 2, 1)
        x8 = self.downsample_4_to_8(x_enc.permute(0, 2, 1)).permute(0, 2, 1)
        
        x8_enc = self.sub_model4(x8, x_mark_enc, x_dec, x_mark_dec, mask)
        x4_enc = self.sub_model3(x4, x_mark_enc, x_dec, x_mark_dec, mask)
        x2_enc = self.sub_model2(x2, x_mark_enc, x_dec, x_mark_dec, mask)
        x1_enc = self.sub_model1(x1, x_mark_enc, x_dec, x_mark_dec, mask)

        x4_up = x4_enc + self.upsample_8_to_4(x8_enc.permute(0, 2, 1)).permute(0, 2, 1)
        x2_up = x2_enc + self.upsample_4_to_2(x4_enc.permute(0, 2, 1)).permute(0, 2, 1)
        x1_up = x1_enc + self.upsample_2_to_1(x2_enc.permute(0, 2, 1)).permute(0, 2, 1)

        return x1_up