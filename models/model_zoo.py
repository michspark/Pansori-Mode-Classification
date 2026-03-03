import torch.nn as nn
from .model_utils import calc_conv_param
from .modules import Conv2DBlock

class Conv2DGRU(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.params = calc_conv_param(config)

        self.kernel_size = config.kernel_size
        self.dilation = config.dilation
        self.num_classes = config.num_classes
        self.enc = self.build_enc()

        self.hidden_dim = config.hidden_dim
        self.num_gru = config.num_gru
        self.dropout = config.dropout

        if config.pool_size:
            out_freq = config.num_bins // (2 ** config.num_layers)
        else:
            out_freq = config.num_bins

        gru_input = self.params[-1]['output_channel'] * out_freq
        self.gru = nn.GRU(input_size=gru_input, hidden_size=self.hidden_dim,
                          num_layers=self.num_gru, batch_first=True,
                          dropout=self.dropout, bidirectional=True)
        self.fc = nn.Linear(self.hidden_dim * 2, self.num_classes)

    def build_enc(self):
        enc = nn.Sequential()
        for idx, param in enumerate(self.params):
            enc.add_module(f'conv_{idx}',
                Conv2DBlock(param['input_channel'], param['output_channel'],
                            kernel_size=self.kernel_size, padding='same',
                            dilation=tuple(self.dilation)))
            if self.config.cnn_dropout:
                enc.add_module(f'dropout_{idx}', nn.Dropout2d(self.config.cnn_dropout))
            if self.config.pool_size:
                enc.add_module(f'pool_{idx}', nn.MaxPool2d(tuple(param['max_pool'])))
        return enc

    def forward(self, x):
        if x.ndim == 3:
            x = x.unsqueeze(1)

        x = self.enc(x)
        b, _, _, t = x.shape
        x = x.permute(0, 3, 1, 2)
        x = x.reshape(b, t, -1)

        x, _ = self.gru(x)
        x = self.fc(x)
        return x
