import torch
import torch.nn as nn


# Encoder Class
class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size, dropout, seq_len):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.seq_len = seq_len

        self.lstm_enc = nn.GRU(input_size=input_size, hidden_size=hidden_size, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_size, latent_size)

    def forward(self, x):
        out, last_h_state = self.lstm_enc(x)
        x_enc = last_h_state.squeeze(dim=0)
        latent = self.fc(x_enc)
        return latent, out


# Decoder Class
class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size, dropout, seq_len, use_act):
        super(Decoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.dropout = dropout
        self.seq_len = seq_len
        self.use_act = use_act  # Parameter to control the last sigmoid activation - depends on the normalization used.
        self.act = nn.Sigmoid()

        self.lstm_dec = nn.GRU(input_size=latent_size, hidden_size=hidden_size, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_size, input_size)

    def forward(self, z):
        # z = z.unsqueeze(1).repeat(1, self.seq_len, 1)
        dec_out, hidden_state = self.lstm_dec(z)
        dec_out = self.fc(dec_out)
        if self.use_act:
            dec_out = self.act(dec_out)

        return dec_out, hidden_state


# LSTM Auto-Encoder Class
class Model(nn.Module):
    def __init__(self, configs):
    # def __init__(self, input_size, hidden_size, dropout_ratio, seq_len, use_act=True):
        super(Model, self).__init__()

        self.input_size = configs.enc_in
        self.hidden_size = configs.d_model
        self.dropout_ratio = configs.dropout
        self.latent_size = configs.d_ff
        self.seq_len = configs.seq_len
        use_act = True

        self.encoder = Encoder(input_size=self.input_size, hidden_size=self.hidden_size, latent_size=self.latent_size, dropout=self.dropout_ratio, seq_len=self.seq_len)
        self.decoder = Decoder(input_size=self.input_size, hidden_size=self.hidden_size, latent_size=self.latent_size, dropout=self.dropout_ratio, seq_len=self.seq_len, use_act=use_act)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):

        return_last_h=False
        return_enc_out=False

        latent, enc_out = self.encoder(x_enc)
        latent = latent.unsqueeze(1).repeat(1, x_enc.shape[1], 1)

        x_dec, last_h = self.decoder(latent)

        if return_last_h:
            return x_dec, last_h
        elif return_enc_out:
            return x_dec, enc_out
        return x_dec
    


if __name__ == '__main__':

    # Model Configuration
    class Configs:
        def __init__(self):
            self.enc_in = 51
            self.d_model = 128
            self.d_ff = 32
            self.dropout = 0.1
            self.seq_len = 10

    # Model Initialization
    model = Model(Configs())
    # print(model)
    # print("Model Initialized Successfully!")

    input = torch.randn(32, 10, 51)
    output = model(input, None, None, None)
    print(output.shape)