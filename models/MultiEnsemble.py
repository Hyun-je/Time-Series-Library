import torch
import copy

from models.Autoformer import Model as Autoformer
from models.FEDformer import Model as FEDformer
from models.TimesNet import Model as TimesNet
from models.DLinear import Model as DLinear
from layers.Autoformer_EncDec import series_decomp

class BasicForcaster(torch.nn.Module):

    def __init__(self, configs):
        super(BasicForcaster, self).__init__()
        self.encoder = torch.nn.Linear(configs.seq_len, configs.pred_len)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        out = self.encoder(x_enc)
        return out


class Model(torch.nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()

        if configs.seq_len % 8 != 0:
            raise ValueError('seq_len must be divisible by 8')

        self.configs_1 = copy.deepcopy(configs)
        self.configs_1.task_name = 'short_term_forecast'
        self.configs_1.seq_len = configs.seq_len//1
        self.configs_1.pred_len = configs.pred_len
        self.configs_1.moving_avg = 100
        self.configs_1.enc_in = 51
        self.forcast_1 = BasicForcaster(self.configs_1)

        self.configs_2 = copy.deepcopy(configs)
        self.configs_2.task_name = 'short_term_forecast'
        self.configs_2.seq_len = configs.seq_len//2
        self.configs_2.pred_len = configs.pred_len
        self.configs_2.moving_avg = 50
        self.configs_2.enc_in = 51
        self.forcast_2 = BasicForcaster(self.configs_2)

        self.configs_4 = copy.deepcopy(configs)
        self.configs_4.task_name = 'short_term_forecast'
        self.configs_4.seq_len = configs.seq_len//4
        self.configs_4.pred_len = configs.pred_len
        self.configs_4.moving_avg = 25
        self.configs_4.enc_in = 51
        self.forcast_4 = BasicForcaster(self.configs_4)

        self.configs_8 = copy.deepcopy(configs)
        self.configs_8.task_name = 'short_term_forecast'
        self.configs_8.seq_len = configs.seq_len//8
        self.configs_8.pred_len = configs.pred_len
        self.configs_8.moving_avg = 12
        self.configs_8.enc_in = 51
        self.forcast_8 = BasicForcaster(self.configs_8)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):

        B = x_enc.shape[0]  # batch size
        L = x_enc.shape[1]  # length of sequence
        C = x_enc.shape[2]  # number of channels

        # Downsampling sequence
        x_enc_1 = x_enc[:, ::1, :]   # (B, 51, L//1)
        x_enc_2 = x_enc[:, ::2, :]   # (B, 51, L//2)
        x_enc_4 = x_enc[:, ::4, :]   # (B, 51, L//4)
        x_enc_8 = x_enc[:, ::8, :]   # (B, 51, L//8)

        # Forecast for each downsampled sequence
        out_1 = self.forcast_1(x_enc_1, None, None, None, None)   # (B, 51, pred_len)
        out_2 = self.forcast_2(x_enc_2, None, None, None, None)   # (B, 51, pred_len)
        out_4 = self.forcast_4(x_enc_4, None, None, None, None)   # (B, 51, pred_len)
        out_8 = self.forcast_8(x_enc_8, None, None, None, None)   # (B, 51, pred_len)

        # Combine the forecast
        out = (out_1 * 0.4) + (out_2 * 0.3) + (out_4 * 0.2) + (out_8 * 0.1)      # (B, 51, pred_len)
        
        return out