import torch
import torch.nn as  nn
import torch.nn.functional as F


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1):
        super(Bottleneck, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.batch_norm1 = nn.BatchNorm1d(out_channels)
        
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.batch_norm2 = nn.BatchNorm1d(out_channels)
        
        self.conv3 = nn.Conv1d(out_channels, out_channels*self.expansion, kernel_size=1, stride=1, padding=0)
        self.batch_norm3 = nn.BatchNorm1d(out_channels*self.expansion)
        
        self.i_downsample = i_downsample
        self.stride = stride
        self.relu = nn.ReLU()
        
    def forward(self, x):
        identity = x.clone()
        x = self.relu(self.batch_norm1(self.conv1(x)))
        
        x = self.relu(self.batch_norm2(self.conv2(x)))
        
        x = self.conv3(x)
        x = self.batch_norm3(x)
        
        #downsample if needed
        if self.i_downsample is not None:
            identity = self.i_downsample(identity)
        #add identity
        x+=identity
        x=self.relu(x)
        
        return x

class Block(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1):
        super(Block, self).__init__()
       

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        self.batch_norm1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        self.batch_norm2 = nn.BatchNorm1d(out_channels)

        self.i_downsample = i_downsample
        self.stride = stride
        self.relu = nn.ReLU()

    def forward(self, x):
      identity = x.clone()

      x = self.relu(self.batch_norm2(self.conv1(x)))
      x = self.batch_norm2(self.conv2(x))

      if self.i_downsample is not None:
          identity = self.i_downsample(identity)
      print(x.shape)
      print(identity.shape)
      x += identity
      x = self.relu(x)
      return x

class ResNet(nn.Module):
    def __init__(self, ResBlock, layer_list, num_channels=3):
        super(ResNet, self).__init__()

        planes = [64, 128, 256, 512]
        self.in_channels = 32
        
        self.conv1 = nn.Conv1d(num_channels, 32, kernel_size=7, stride=2, padding=3, bias=False)
        self.batch_norm1 = nn.BatchNorm1d(32)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool1d(kernel_size = 3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(ResBlock, layer_list[0], planes=planes[0]//2)
        self.layer2 = self._make_layer(ResBlock, layer_list[1], planes=planes[1]//2, stride=2)
        self.layer3 = self._make_layer(ResBlock, layer_list[2], planes=planes[2]//2, stride=2)
        # self.layer4 = self._make_layer(ResBlock, layer_list[3], planes=planes[3]//2, stride=2)
        
    def forward(self, x):
        x = self.relu(self.batch_norm1(self.conv1(x)))
        x = self.max_pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        # x = self.layer4(x)
        
        return x
        
    def _make_layer(self, ResBlock, blocks, planes, stride=1):
        ii_downsample = None
        layers = []
        
        if stride != 1 or self.in_channels != planes*ResBlock.expansion:
            ii_downsample = nn.Sequential(
                nn.Conv1d(self.in_channels, planes*ResBlock.expansion, kernel_size=1, stride=stride),
                nn.BatchNorm1d(planes*ResBlock.expansion)
            )
            
        layers.append(ResBlock(self.in_channels, planes, i_downsample=ii_downsample, stride=stride))
        self.in_channels = planes*ResBlock.expansion
        
        for i in range(blocks-1):
            layers.append(ResBlock(self.in_channels, planes))
            
        return nn.Sequential(*layers)

        
        
def ResNet50(channels=3):
    return ResNet(Bottleneck, [2,2,2,2], channels)



# Encoder Class
class Encoder(nn.Module):
    def __init__(self, input_channels):
        super(Encoder, self).__init__()
        self.resnet = ResNet(Bottleneck, [2,2,2,2], input_channels)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.resnet(x)
        return x.permute(0, 2, 1)


# Decoder Class
class Decoder(nn.Module):
    def __init__(self, input_size):
        super(Decoder, self).__init__()

        self.upsample1 = nn.Sequential(
            nn.ConvTranspose1d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )
        self.upsample2 = nn.Sequential(
            nn.ConvTranspose1d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        self.upsample3 = nn.Sequential(
            nn.ConvTranspose1d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        self.upsample4 = nn.Sequential(
            nn.ConvTranspose1d(in_channels=64, out_channels=input_size, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(input_size),
            nn.ReLU(),
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.upsample1(x)
        x = self.upsample2(x)
        x = self.upsample3(x)
        x = self.upsample4(x)
        x = x.permute(0, 2, 1)
        return x


# LSTM Auto-Encoder Class
class Model(nn.Module):
    def __init__(self, configs):
    # def __init__(self, input_size, hidden_size, dropout_ratio, seq_len, use_act=True):
        super(Model, self).__init__()

        self.input_size = configs.enc_in
        self.hidden_size = configs.d_model
        self.dropout_ratio = configs.dropout
        self.seq_len = configs.seq_len
        use_act = True

        self.encoder = Encoder(configs.enc_in)
        self.decoder = Decoder(configs.enc_in)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):

        x_enc = self.encoder(x_enc)
        x_dec = self.decoder(x_enc)

        return x_dec
    
if __name__ == '__main__':

    class Config:
        def __init__(self):
            self.enc_in = 51
            self.d_model = 64
            self.dropout = 0.1
            self.seq_len = 100

    configs = Config()
    model = Model(configs)

    # B = x_enc.shape[0]  # batch size
    # L = x_enc.shape[1]  # length of sequence
    # C = x_enc.shape[2]  # number of channels


    input = torch.randn(32, 128, 51)
    output = model(input, None, None, None)

    print(f'{input.shape=}')
    print(f'{output.shape=}')