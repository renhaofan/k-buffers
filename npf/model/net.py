import torch.nn as nn
import torch
from torchsummary import summary


def positional_encoding(tensor, num_encoding_functions=6, include_input=False, log_sampling=True):
    encoding = [tensor] if include_input else []
    frequency_bands = None
    if log_sampling:
        frequency_bands = 2.0 ** torch.linspace(
            0.0,
            num_encoding_functions - 1,
            num_encoding_functions,
            dtype=tensor.dtype,
            device=tensor.device,
        )
    else:
        frequency_bands = torch.linspace(
            2.0 ** 0.0,
            2.0 ** (num_encoding_functions - 1),
            num_encoding_functions,
            dtype=tensor.dtype,
            device=tensor.device,
        )

    for freq in frequency_bands:
        for func in [torch.sin, torch.cos]:
            # print(func(tensor * freq).shape)
            encoding.append(func(tensor * freq))

    # Special case, for no positional encoding
    return torch.cat(encoding, dim=-1)

def frebpcr_positional_encoding(tensor, num_encoding_functions=6, include_input=False, log_sampling=True, factor=1):
    encoding = [tensor] if include_input else []
    frequency_bands = None
    if log_sampling:
        frequency_bands = 2.0 ** torch.linspace(
            0.0,
            num_encoding_functions - 1,
            num_encoding_functions * factor,
            dtype=tensor.dtype,
            device=tensor.device,
        )
    else:
        frequency_bands = torch.linspace(
            2.0 ** 0.0,
            2.0 ** (num_encoding_functions - 1),
            num_encoding_functions,
            dtype=tensor.dtype,
            device=tensor.device,
        )

    for freq in frequency_bands:
        for func in [torch.sin, torch.cos]:
            encoding.append(func(tensor * freq * 3.1415926))  

    # Special case, for no positional encoding
    return torch.cat(encoding, dim=-1)


def fourier_encoding(tensor, choice='xyz', gaussian_scale=10, xyz_size=30, dirs_size=12, sigma=0.4):
    # https://github.com/tancik/fourier-feature-networks/blob/master/Experiments/3d_simple_nerf.ipynb
    # raw embeding size -> 256 performs very bad
    if choice == 'xyz':
        bvals_xyz = torch.normal(
            mean=0, std=sigma, size=[xyz_size, 3], device=tensor.device) * gaussian_scale
        avals_xyz = torch.ones((bvals_xyz.shape[0]), device=tensor.device)
        ab_xyz = [avals_xyz, bvals_xyz]
    elif choice == 'dirs':
        bvals_dirs = torch.normal(
            mean=0, std=sigma, size=[dirs_size, 3], device=tensor.device) * gaussian_scale
        avals_dirs = torch.ones((bvals_dirs.shape[0]), device=tensor.device)
        ab_dirs = [avals_dirs, bvals_dirs]
    else:
        raise NotImplementedError

    def input_encoder(x, a, b): return (torch.cat([a * torch.sin(torch.mm((2.*torch.pi*x), b.T)),
                                                   a * torch.cos(torch.mm((2.*torch.pi*x), b.T))], axis=-1) / torch.norm(a))
    ab = ab_xyz if choice == 'xyz' else ab_dirs
    encoding = input_encoder(tensor, *ab)
    return encoding


class MLP(nn.Module):

    def __init__(self, dim, use_fourier):
        super(MLP, self).__init__()

        # fourier_encoding init
        self.use_fourier = use_fourier

        # note: 60 is the embedding dimension after positional encoding
        self.l1 = nn.Linear(60, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(280, 256)
        self.l4 = nn.Linear(256, 128)
        self.l5 = nn.Linear(128, dim)

        self.ac = nn.ReLU()

    def forward(self, xyz, dirs):
        if not self.use_fourier:
            xyz = positional_encoding(xyz, 10)
            dirs = positional_encoding(dirs, 4)
        else:
            xyz = fourier_encoding(xyz, 'xyz')
            dirs = fourier_encoding(dirs, 'dirs')

        # layer 1
        x = self.l1(xyz)
        x = self.ac(x)

        x = self.l2(x)
        x = self.ac(x)

        x = torch.cat([x, dirs], dim=-1)

        x = self.l3(x)
        x = self.ac(x)

        x = self.l4(x)
        x = self.ac(x)

        x = self.l5(x)
        return x


class AFNet(nn.Module):

    def __init__(self, dim):
        super(AFNet, self).__init__()

        ##############################################
        # self.l1 = nn.Linear(120,256)
        # self.l1 = nn.Linear(60,256)
        # self.l2 = nn.Linear(256,256) 
        # self.l3 = nn.Linear(256,256) 
        # self.l4 = nn.Linear(280,128)
        # self.l5 = nn.Linear(128,dim)

        self.l1 = nn.Linear(60, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(280, 256)
        self.l4 = nn.Linear(256, 128)
        self.l5 = nn.Linear(128, dim)

        # self.ac = nn.ReLU()

        self.hyper = nn.Sequential(
            nn.Linear(60,256),
            nn.ReLU(),
            nn.Linear(256,8),
        )


    def forward(self, xyz, dirs):
        # xyz = positional_encoding(xyz, 10, factor=2)
        # dirs = positional_encoding(dirs, 2, factor=2)
        xyz = positional_encoding(xyz, 10)
        dirs = positional_encoding(dirs, 4)


        ##############################################
        # freq = self.hyper(xyz).unsqueeze(-1)
        
        # x = self.l1(xyz)
        # x = x * torch.sin(x * freq[:,0] + freq[:,1])

        # x = self.l2(x)
        # x = x * torch.sin(x * freq[:,2] + freq[:,3])


        # x = self.l3(x)
        # x = x * torch.sin(x * freq[:,4] + freq[:,5])

        # x = torch.cat([x, dirs], dim=-1)

        # x = self.l4(x)
        # x = x * torch.sin(x * freq[:,6] + freq[:,7])

        # x = self.l5(x)
        # return x
        
        
        freq = self.hyper(xyz).unsqueeze(-1)
        
        x = self.l1(xyz)
        x = x * torch.sin(x * freq[:,0] + freq[:,1])

        x = self.l2(x)
        x = x * torch.sin(x * freq[:,2] + freq[:,3])

        x = torch.cat([x, dirs], dim=-1)

        x = self.l3(x)
        x = x * torch.sin(x * freq[:,4] + freq[:,5])

        x = self.l4(x)
        x = x * torch.sin(x * freq[:,6] + freq[:,7])

        x = self.l5(x)
        return x


class GatedBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1, padding_mode='reflect', act_fun=nn.ELU, normalization=nn.InstanceNorm2d):
        super().__init__()
        self.pad_mode = padding_mode
        self.filter_size = kernel_size
        self.stride = stride
        self.dilation = dilation

        n_pad_pxl = int(self.dilation * (self.filter_size - 1) / 2)

        # this is for backward campatibility with older model checkpoints
        self.block = nn.ModuleDict(
            {
                'conv_f': nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, dilation=dilation, padding=n_pad_pxl),
                'act_f': act_fun(),
                'conv_m': nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, dilation=dilation, padding=n_pad_pxl),
                'act_m': nn.Sigmoid(),
                'norm': normalization(out_channels)
            }
        )

        self.ac = nn.ReLU()

    def forward(self, x, *args, **kwargs):
        features = self.block.act_f(self.block.conv_f(x))
        mask = self.block.act_m(self.block.conv_m(x))
        output = features * mask
        output = self.block.norm(output)
        output = self.ac(output)

        return output


class DownsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, conv_block=GatedBlock):
        super().__init__()

        self.conv = conv_block(in_channels, out_channels)
        self.down = nn.AvgPool2d(2, 2)

    def forward(self, inputs, mask=None):
        outputs = self.down(inputs)
        outputs = self.conv(outputs, mask=mask)
        return outputs


class UpsampleBlock(nn.Module):
    def __init__(self, out_channels, upsample_mode, num_filt, conv_block=GatedBlock):
        super().__init__()

        #  = out_channels if same_num_filt else out_channels * 2
        if upsample_mode == 'deconv':
            self.up = nn.ConvTranspose2d(
                num_filt, out_channels, 4, stride=2, padding=1)
            self.conv = conv_block(out_channels * 2, out_channels)
        elif upsample_mode == 'bilinear' or upsample_mode == 'nearest':
            self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode=upsample_mode),
                                    nn.Conv2d(num_filt, out_channels,
                                              3, padding=1)
                                    )
            self.conv = conv_block(out_channels * 2, out_channels)
        else:
            assert False

    def forward(self, inputs1, inputs2):
        in1_up = self.up(inputs1)
        output = self.conv(torch.cat([in1_up, inputs2], 1))

        return output


class UNet(nn.Module):
    def __init__(self, args, out_dim=3, upsample_mode='nearest'):
        super().__init__()

        in_dim = args.dim * args.points_per_pixel

        if args.udim == 'pp':
            filters = [16, 32, 48, 64, 80]
        elif args.udim == 'npbg':
            filters = [64, 128, 256, 512, 1024]
            filters = [x // 4 for x in filters]
        elif args.udim == '4xnpbg':
            filters = [64, 128, 256, 512, 1024]
        else:
            assert False

        self.start = GatedBlock(in_dim, filters[0])

        self.down1 = DownsampleBlock(filters[0], filters[1])
        self.down2 = DownsampleBlock(filters[1], filters[2])

        if args.U == 4:
            self.down3 = DownsampleBlock(filters[2], filters[3])
            self.down4 = DownsampleBlock(filters[3], filters[4])

            self.up4 = UpsampleBlock(filters[3], upsample_mode, filters[4])
            self.up3 = UpsampleBlock(filters[2], upsample_mode, filters[3])
        self.up2 = UpsampleBlock(filters[1], upsample_mode, filters[2])
        self.up1 = UpsampleBlock(filters[0], upsample_mode, filters[1])

        self.final = nn.Sequential(
            nn.Conv2d(filters[0], out_dim, 1),
        )
        self.U = args.U

    def forward(self, x):

        in64 = self.start(x)

        down1 = self.down1(in64)
        down2 = self.down2(down1)
        if self.U == 4:
            down3 = self.down3(down2)
            down4 = self.down4(down3)

            up4 = self.up4(down4, down3)
            up3 = self.up3(up4, down2)
            up2 = self.up2(up3, down1)
        else:
            up2 = self.up2(down2, down1)
        up1 = self.up1(up2, in64)

        return self.final(up1)


class UNet_bpcr(nn.Module):
    def __init__(self, args, out_dim=3, upsample_mode='nearest', U=4, udim='npbg'):
        super().__init__()

        self.pre = nn.Conv2d(in_channels=args.dim * args.points_per_pixel, out_channels=args.dim, kernel_size=1)
        in_dim = args.dim

        if udim == 'pp':
            filters = [16, 32, 48, 64, 80]
        elif udim == 'npbg':
            filters = [64, 128, 256, 512, 1024]
            filters = [x // 4 for x in filters]
        elif udim == '4xnpbg':
            filters = [64, 128, 256, 512, 1024]
        else:
            assert False

        self.start = GatedBlock(in_dim, filters[0])

        self.down1 = DownsampleBlock(filters[0], filters[1])
        self.down2 = DownsampleBlock(filters[1], filters[2])

        if U == 4:
            self.down3 = DownsampleBlock(filters[2], filters[3])
            self.down4 = DownsampleBlock(filters[3], filters[4])

            self.up4 = UpsampleBlock(filters[3], upsample_mode, filters[4])
            self.up3 = UpsampleBlock(filters[2], upsample_mode, filters[3])
        self.up2 = UpsampleBlock(filters[1], upsample_mode, filters[2])
        self.up1 = UpsampleBlock(filters[0], upsample_mode, filters[1])

        self.final = nn.Sequential(
            nn.Conv2d(filters[0], out_dim, 1),
        )
        self.U = U

    def forward(self, x):
        
        in64 = self.start(self.pre(x))

        down1 = self.down1(in64)
        down2 = self.down2(down1)
        if self.U == 4:
            down3 = self.down3(down2)
            down4 = self.down4(down3)

            up4 = self.up4(down4, down3)
            up3 = self.up3(up4, down2)
            up2 = self.up2(up3, down1)
        else:
            up2 = self.up2(down2, down1)
        up1 = self.up1(up2, in64)

        return self.final(up1)


class UNet_color(nn.Module):
    def __init__(self, args, out_dim=3, upsample_mode='nearest'):
        super().__init__()

        in_dim = 3 * args.points_per_pixel

        if args.udim == 'pp':
            filters = [16, 32, 48, 64, 80]
        elif args.udim == 'npbg':
            filters = [64, 128, 256, 512, 1024]
            filters = [x // 4 for x in filters]
        elif args.udim == '4xnpbg':
            filters = [64, 128, 256, 512, 1024]
        else:
            assert False

        self.start = GatedBlock(in_dim, filters[0])

        self.down1 = DownsampleBlock(filters[0], filters[1])
        self.down2 = DownsampleBlock(filters[1], filters[2])

        if args.U == 4:
            self.down3 = DownsampleBlock(filters[2], filters[3])
            self.down4 = DownsampleBlock(filters[3], filters[4])

            self.up4 = UpsampleBlock(filters[3], upsample_mode, filters[4])
            self.up3 = UpsampleBlock(filters[2], upsample_mode, filters[3])
        self.up2 = UpsampleBlock(filters[1], upsample_mode, filters[2])
        self.up1 = UpsampleBlock(filters[0], upsample_mode, filters[1])

        self.final = nn.Sequential(
            nn.Conv2d(filters[0], out_dim, 1),
        )
        self.U = args.U

    def forward(self, x):

        in64 = self.start(x)

        down1 = self.down1(in64)
        down2 = self.down2(down1)
        if self.U == 4:
            down3 = self.down3(down2)
            down4 = self.down4(down3)

            up4 = self.up4(down4, down3)
            up3 = self.up3(up4, down2)
            up2 = self.up2(up3, down1)
        else:
            up2 = self.up2(down2, down1)
        up1 = self.up1(up2, in64)

        return self.final(up1)

if __name__ == '__main__':
    device = torch.device("cuda")
    # moda_num = 64
    # batch_size = 1
    # H = 800
    # W = 800
    # input_size = (batch_size, moda_num, H, W)

    # ----------------------------------------------------------------
    # Total params: 187,272
    # Trainable params: 187,272
    # Non-trainable params: 0
    # ----------------------------------------------------------------
    # Input size (MB): 3076241.94
    # Forward/backward pass size (MB): 4110.75
    # Params size (MB): 0.71
    # Estimated Total Size (MB): 3080353.41
    # ----------------------------------------------------------------
    data = torch.randn((299336, 3)).to(device)
    mlp = MLP(8, False).to(device)
    print(summary(mlp, [(299336, 3), (299336, 3)]))