import torch
import torch.nn as nn

from ..loss_functions import HSLoss, LPLoss, MaxwellLoss
from ._base import BaseModel


class FNO(BaseModel):
    def __init__(
        self,
        modes,
        width,
        blocks,
        padding,
        out_channels,
        lr,
        weight_decay,
        epochs,
        scheduler,
        lambda0,
        dl,
        with_maxwell_loss,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.save_hyperparameters({"model": self.name})
        self.l1_rel = LPLoss(1)
        self.l2_rel = LPLoss(2)
        if "2d" in self.name:
            self.h1_rel = HSLoss(1)
            self.h2_rel = HSLoss(2)
            self.mloss_l1 = MaxwellLoss(lambda0, dl, loss_fn=self.l1_rel)
            self.mloss_l2 = MaxwellLoss(lambda0, dl, loss_fn=self.l2_rel)
        if "3d" in self.name and with_maxwell_loss:
            raise RuntimeError("fno3d does not support maxwell loss!")
        self.model = self.get_model(modes, width, blocks, padding, out_channels)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("FNO")
        parser.add_argument("--modes", type=int, default=12)
        parser.add_argument("--width", type=int, default=32)
        parser.add_argument("--blocks", type=int, default=10)
        parser.add_argument("--padding", type=int, default=2)
        return parent_parser


class FNO2d(FNO):
    name = "fno2d"

    @staticmethod
    def get_model(modes, width, blocks, padding, out_channels):
        return FNOModel2d(modes, width, blocks, padding, out_channels)


class FNO3d(FNO):
    name = "fno3d"

    @staticmethod
    def get_model(modes, width, blocks, padding, out_channels):
        return FNOModel3d(modes, width, blocks, padding, out_channels)


class FNOModel(nn.Module):
    def __init__(self, modes, width, blocks, padding, out_channels):
        super().__init__()
        self.fno_blocks = self.get_fno_blocks(blocks, modes, width)
        self.conv_in = self.get_conv(1, width)
        self.conv_out = self.get_conv(width, out_channels)
        self.pad_in = self.get_pad(padding)
        self.pad_out = self.get_pad(-padding)

    def get_fno_blocks(self, blocks, modes, width):
        fno_blocks = nn.Sequential(
            *[self.get_fno_block(modes, width) for _ in range(blocks)]
        )
        return fno_blocks

    def forward(self, x):
        x = self.conv_in(x)
        x = self.pad_in(x)
        x = self.fno_blocks(x)
        x = self.pad_out(x)
        x = self.conv_out(x)
        return x


class FNOModel2d(FNOModel):
    @staticmethod
    def get_fno_block(modes, width):
        return FNOBlock2d(modes, width)

    @staticmethod
    def get_conv(in_channels, out_channels):
        return nn.Conv2d(in_channels, out_channels, 1, bias=True)

    @staticmethod
    def get_pad(padding):
        return nn.ConstantPad2d(padding, 0.0)


class FNOModel3d(FNOModel):
    @staticmethod
    def get_fno_block(modes, width):
        return FNOBlock3d(modes, width)

    @staticmethod
    def get_conv(in_channels, out_channels):
        return nn.Conv3d(in_channels, out_channels, 1, bias=True)

    @staticmethod
    def get_pad(padding):
        return nn.ConstantPad3d(padding, 0.0)


class FNOBlock(nn.Module):
    def __init__(self, modes, width):
        super().__init__()
        self.act = nn.GELU()
        self.fftconv = self.get_fft_conv(modes, width)
        self.conv = self.get_real_conv(width)
        self.bn = self.get_batch_norm(width)

    def forward(self, x):
        x = self.fftconv(x) + self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class FNOBlock2d(FNOBlock):
    @staticmethod
    def get_fft_conv(modes, width):
        return FFTConv2d(width, width, modes)

    @staticmethod
    def get_real_conv(width):
        return nn.Conv2d(width, width, 1, bias=False)

    @staticmethod
    def get_batch_norm(width):
        return nn.BatchNorm2d(width)


class FNOBlock3d(FNOBlock):
    @staticmethod
    def get_fft_conv(modes, width):
        return FFTConv3d(width, width, modes)

    @staticmethod
    def get_real_conv(width):
        return nn.Conv3d(width, width, 1, bias=False)

    @staticmethod
    def get_batch_norm(width):
        return nn.BatchNorm3d(width)


class FFTConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes):
        super().__init__()

        self.out_channels = out_channels
        self.modes = modes

        self.w = nn.Parameter(
            torch.rand(in_channels, out_channels, 2 * modes, modes, 2)
            / (in_channels * out_channels)
        )

    @staticmethod
    def cmul2d(a, b):
        return torch.einsum("bixy,ioxy->boxy", a, b)

    def forward(self, x):
        xs = x.shape
        device = x.device

        x = torch.fft.rfft2(x)

        out_ft = torch.zeros(
            xs[0],
            self.out_channels,
            xs[-2],
            xs[-1] // 2 + 1,
            dtype=torch.cfloat,
            device=device,
        )

        m = self.modes
        cx = xs[-2] // 2
        x = torch.fft.fftshift(x, -2)
        out_ft[..., cx - m : cx + m, :m] = self.cmul2d(
            x[..., cx - m : cx + m, :m],
            torch.view_as_complex(self.w),
        )
        out_ft = torch.fft.ifftshift(out_ft, -2)

        x = torch.fft.irfft2(out_ft, xs[-2:])

        return x


class FFTConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, modes):
        super().__init__()

        self.out_channels = out_channels
        self.modes = modes

        self.w = nn.Parameter(
            torch.rand(in_channels, out_channels, 2 * modes, 2 * modes, modes, 2)
            / (in_channels * out_channels)
        )

    @staticmethod
    def cmul3d(a, b):
        return torch.einsum("bixyz,ioxyz->boxyz", a, b)

    def forward(self, x):
        xs = x.shape
        device = x.device

        x = torch.fft.rfftn(x, dim=[-3, -2, -1])

        out_ft = torch.zeros(
            xs[0],
            self.out_channels,
            xs[-3],
            xs[-2],
            xs[-1] // 2 + 1,
            dtype=torch.cfloat,
            device=device,
        )

        m = self.modes
        cx = xs[-3] // 2
        cy = xs[-2] // 2
        x = torch.fft.fftshift(x, (-3, -2))
        out_ft[..., cx - m : cx + m, cy - m : cy + m, :m] = self.cmul3d(
            x[..., cx - m : cx + m, cy - m : cy + m, :m],
            torch.view_as_complex(self.w),
        )
        out_ft = torch.fft.ifftshift(out_ft, (-3, -2))

        x = torch.fft.irfftn(out_ft, s=xs[-3:])

        return x
