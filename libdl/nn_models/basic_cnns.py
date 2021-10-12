import torch
import torch.nn as nn


class basic_cnn(torch.nn.Module):
    """
    Basic CNN as in Johannes Zeitler's report, for HCQT input with 75 (-1) context frames
    and variable amount of pitch bins (num octaves * 12 * 3 bins per semitone)
    The number of input channels, channels in the hidden layers, and output
    dimensions (e.g. for pitch output) can be parameterized.
    Layer normalization is only performed over frequency and channel dimensions,
    not over time (in order to work with variable length input).

    Args (Defaults: BasicCNN by Johannes Zeitler but with 6 input channels):
        n_chan_input:   Number of input channels (harmonics in HCQT)
        n_chan_layers:  Number of channels in the hidden layers (list)
        n_bins_in:      Number of input bins (12 * number of octaves)
        n_bins_out:     Number of output bins (12 for pitch class, 72 for pitch)
        a_lrelu:        alpha parameter (slope) of LeakyReLU activation function
        p_dropout:      Dropout probability
    """
    def __init__(self, n_chan_input=6, n_chan_layers=[20,20,10,1], n_bins_in=216, n_bins_out=12, a_lrelu=0.3, p_dropout=0.2):
        super(basic_cnn, self).__init__()

        n_in = n_chan_input
        n_ch = n_chan_layers
        last_kernel_size = n_bins_in//3 + 1 - n_bins_out

        # Layer normalization over frequency and channels (harmonics of HCQT)
        self.layernorm = nn.LayerNorm(normalized_shape=[n_in, n_bins_in])
        # Prefiltering
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=n_in, out_channels=n_ch[0], kernel_size=(15,15), padding=(7,7), stride=(1,1)),
            nn.LeakyReLU(negative_slope=a_lrelu),
            nn.MaxPool2d(kernel_size=(2,1), stride=(2,1), padding=(0,0)),
            nn.Dropout(p=p_dropout)
        )
        # Binning to MIDI pitches
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=n_ch[0], out_channels=n_ch[1], kernel_size=(3,3), padding=(0,0), stride=(3,3)),
            nn.LeakyReLU(negative_slope=a_lrelu),
            nn.MaxPool2d(kernel_size=(2,1), stride=(2,1), padding=(0,0)),
            nn.Dropout(p=p_dropout)
        )
        # Time reduction
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=n_ch[1], out_channels=n_ch[2], kernel_size=(6,1), padding=(0,0), stride=(1,1)),
            nn.LeakyReLU(negative_slope=a_lrelu),
            nn.Dropout(p=p_dropout)
        )
        # Chroma reduction
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=n_ch[2], out_channels=n_ch[3], kernel_size=(1,1), padding=(0,0), stride=(1,1)),
            nn.LeakyReLU(negative_slope=a_lrelu),
            nn.Dropout(p=p_dropout),
            nn.Conv2d(in_channels=n_ch[3], out_channels=1, kernel_size=(1,last_kernel_size), padding=(0,0), stride=(1,1)),
            nn.Sigmoid()
        )
    def forward(self, x):
        x_norm = self.layernorm(x.transpose(1, 2)).transpose(1, 2)
        conv1_lrelu = self.conv1(x_norm)
        conv2_lrelu = self.conv2(conv1_lrelu)
        conv3_lrelu = self.conv3(conv2_lrelu)
        y_pred = self.conv4(conv3_lrelu)
        return y_pred


class basic_cnn_pool(torch.nn.Module):
    """
    Basic CNN as in Johannes Zeitler's report, for HCQT input with 75 (-1) context frames
    and variable amount of pitch bins (num octaves * 12 * 3 bins per semitone)
    The number of input channels, channels in the hidden layers, and output
    dimensions (e.g. for pitch output) can be parameterized.
    This "pool" variant replaces strided convolutions by longer max pooling in
    order to achieve better shift invariance in time
    Layer normalization is only performed over frequency and channel dimensions,
    not over time (in order to work with variable length input).

    Args (Defaults: BasicCNN by Johannes Zeitler but with 6 input channels):
        n_chan_input:   Number of input channels (harmonics in HCQT)
        n_chan_layers:  Number of channels in the hidden layers (list)
        n_bins_in:      Number of input bins (12 * number of octaves)
        n_bins_out:     Number of output bins (12 for pitch class, 72 for pitch)
        a_lrelu:        alpha parameter (slope) of LeakyReLU activation function
        p_dropout:      Dropout probability
    """
    def __init__(self, n_chan_input=6, n_chan_layers=[20,20,10,1], n_bins_in=216, n_bins_out=12, a_lrelu=0.3, p_dropout=0.2):
        super(basic_cnn_pool, self).__init__()

        n_in = n_chan_input
        n_ch = n_chan_layers
        last_kernel_size = n_bins_in//3 + 1 - n_bins_out

        # Layer normalization over frequency and channels (harmonics of HCQT)
        self.layernorm = nn.LayerNorm(normalized_shape=[n_in, n_bins_in])
        # Prefiltering
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=n_in, out_channels=n_ch[0], kernel_size=(15,15), padding=(7,7), stride=(1,1)),
            nn.LeakyReLU(negative_slope=a_lrelu),
            nn.MaxPool2d(kernel_size=(8,1), stride=(8,1), padding=(0,0)),
            nn.Dropout(p=p_dropout)
        )
        # Binning to MIDI pitches
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=n_ch[0], out_channels=n_ch[1], kernel_size=(3,3), padding=(1,1), stride=(1,1)),
            nn.LeakyReLU(negative_slope=a_lrelu),
            nn.MaxPool2d(kernel_size=(3,3), stride=(3,3), padding=(0,0)),
            nn.Dropout(p=p_dropout)
        )
        # Time reduction
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=n_ch[1], out_channels=n_ch[2], kernel_size=(3,1), padding=(0,0), stride=(1,1)),
            nn.LeakyReLU(negative_slope=a_lrelu),
            nn.Dropout(p=p_dropout)
        )
        # Chroma reduction
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=n_ch[2], out_channels=n_ch[3], kernel_size=(1,1), padding=(0,0), stride=(1,1)),
            nn.LeakyReLU(negative_slope=a_lrelu),
            nn.Dropout(p=p_dropout),
            nn.Conv2d(in_channels=n_ch[3], out_channels=1, kernel_size=(1,last_kernel_size), padding=(0,0), stride=(1,1)),
            nn.Sigmoid()
        )
    def forward(self, x):
        x_norm = self.layernorm(x.transpose(1, 2)).transpose(1, 2)
        conv1_lrelu = self.conv1(x_norm)
        conv2_lrelu = self.conv2(conv1_lrelu)
        conv3_lrelu = self.conv3(conv2_lrelu)
        y_pred = self.conv4(conv3_lrelu)
        return y_pred
