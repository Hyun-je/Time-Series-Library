import torch
import copy

from models.Autoformer import Model as Autoformer
from models.FEDformer import Model as FEDformer
from models.TimesNet import Model as TimesNet
from models.DLinear import Model as DLinear
from layers.Autoformer_EncDec import series_decomp


class MultiEnsemble(torch.nn.Module):

    def __init__(self, configs):
        super(MultiEnsemble, self).__init__()

        if configs.seq_len % 8 != 0:
            raise ValueError('seq_len must be divisible by 8')

        self.configs_1 = copy.deepcopy(configs)
        self.configs_1.task_name = 'short_term_forecast'
        self.configs_1.seq_len = configs.seq_len
        self.configs_1.pred_len = configs.pred_len
        self.configs_1.moving_avg = 100
        self.configs_1.enc_in = 51
        self.forcast_1 = DLinear(self.config_1)

        self.configs_2 = copy.deepcopy(configs)
        self.configs_2.task_name = 'short_term_forecast'
        self.configs_2.seq_len = configs.seq_len//2
        self.configs_2.pred_len = configs.pred_len
        self.configs_2.moving_avg = 50
        self.configs_2.enc_in = 51
        self.forcast_2 = DLinear(self.config_2)

        self.configs_4 = copy.deepcopy(configs)
        self.configs_4.task_name = 'short_term_forecast'
        self.configs_4.seq_len = configs.seq_len//4
        self.configs_4.pred_len = configs.pred_len
        self.configs_4.moving_avg = 25
        self.configs_4.enc_in = 51
        self.forcast_4 = DLinear(self.config_4)

        self.configs_8 = copy.deepcopy(configs)
        self.configs_8.task_name = 'short_term_forecast'
        self.configs_8.seq_len = configs.seq_len//8
        self.configs_8.pred_len = configs.pred_len
        self.configs_8.moving_avg = 12
        self.configs_8.enc_in = 51
        self.forcast_8 = DLinear(self.config_8)

    def forward(self, x):

        B = x.shape[0]  # batch size
        C = x.shape[1]  # number of channels
        L = x.shape[2]  # length of sequence

        # Downsampling sequence
        x_1 = x[:, :, ::1]   # (B, 51, L//1)
        x_2 = x[:, :, ::2]   # (B, 51, L//2)
        x_4 = x[:, :, ::4]   # (B, 51, L//4)
        x_8 = x[:, :, ::8]   # (B, 51, L//8)

        y_1 = self.forcast_1(x_1)   # (B, 51, pred_len)
        y_2 = self.forcast_2(x_2)   # (B, 51, pred_len)
        y_4 = self.forcast_4(x_4)   # (B, 51, pred_len)
        y_8 = self.forcast_8(x_8)   # (B, 51, pred_len)

        y = (y_1 * 0.5) + (y_2 * 0.3) + (y_4 * 0.2) + (y_8 * 0.1)      # (B, 51, pred_len)
        
        return y