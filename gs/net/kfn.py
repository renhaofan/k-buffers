import torch
import torch.nn as nn
from net.npbgunet import RefinerUNet

class KFN(nn.Module):
    def __init__(self, in_dim=32, chan_embed=48) -> None:
        super(KFN, self).__init__()

        # self.apt_conv = nn.Conv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=1)

        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv1 = nn.Conv2d(in_dim, chan_embed, 3, padding=1)
        self.conv2 = nn.Conv2d(chan_embed, chan_embed, 3, padding=1)
        self.conv3 = nn.Conv2d(chan_embed, in_dim, 1)

    def forward(self, x):
        # 9ms
        xt = self.act(self.conv1(x))
        xt = self.act(self.conv2(xt))
        xt = self.conv3(xt)
        mask = torch.softmax(xt, dim=1)
        # mask = xt

        # 3ms
        # ret = self.apt_conv(x.mul(mask))
        ret = x.mul(mask)

        return ret, mask


class NN_comp(nn.Module):

    def __init__(self, in_channels, out_channels=3) -> None:
        super(NN_comp, self).__init__()

        # self.kfn = KFN(in_dim=gs_dim, out_dim=in_channels)

        self.kfn = KFN(in_dim=in_channels)
        self.unet = RefinerUNet(num_input_channels=in_channels, num_output_channels=out_channels)

    def forward(self, x):
        # ret = self.kfn(x)
        # return ret

        denoise_fm, mask = self.kfn(x)
        ret = self.unet(denoise_fm)
        return ret, mask