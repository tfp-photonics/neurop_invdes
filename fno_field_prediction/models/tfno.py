from neuralop.models.fno_block import FactorizedSpectralConv2d, FactorizedSpectralConv3d

import torch.nn as nn

from ..loss_functions import HSLoss, LPLoss, MaxwellLoss
from ._base import BaseModel


class TFNO(BaseModel):
    def __init__(
        self,
        modes,
        width,
        blocks,
        padding,
        factorization,
        joint_factorization,
        rank,
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
        self.model = self.get_model(
            modes,
            width,
            blocks,
            padding,
            factorization,
            joint_factorization,
            rank,
            out_channels,
        )

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("TFNO")
        parser.add_argument("--factorization", type=str, default="tucker")
        parser.add_argument("--joint_factorization", action="store_true")
        parser.add_argument("--rank", type=float, default=0.5)
        return parent_parser


class TFNO2d(TFNO):
    name = "tfno2d"

    @staticmethod
    def get_model(
        modes,
        width,
        blocks,
        padding,
        factorization,
        joint_factorization,
        rank,
        out_channels,
    ):
        return TFNOModel2d(
            modes,
            width,
            blocks,
            padding,
            factorization,
            joint_factorization,
            rank,
            out_channels,
        )


class TFNO3d(TFNO):
    name = "tfno3d"

    @staticmethod
    def get_model(
        modes,
        width,
        blocks,
        padding,
        factorization,
        joint_factorization,
        rank,
        out_channels,
    ):
        return TFNOModel3d(
            modes,
            width,
            blocks,
            padding,
            factorization,
            joint_factorization,
            rank,
            out_channels,
        )


class TFNOModel2d(nn.Module):
    def __init__(
        self,
        modes,
        width,
        blocks,
        padding,
        factorization,
        joint_factorization,
        rank,
        out_channels,
    ):
        super().__init__()
        self.n_layers = blocks
        self.fft_convs = self.get_fft_convs(
            modes, width, factorization, joint_factorization, rank
        )
        self.skips = self.get_skips(width)
        self.norms = self.get_norms(width)
        self.conv_in = self.get_conv(1, width)
        self.conv_out = self.get_conv(width, out_channels)
        self.pad_in = self.get_pad(padding)
        self.pad_out = self.get_pad(-padding)
        self.act = nn.GELU()

    def get_fft_convs(self, modes, width, factorization, joint_factorization, rank):
        fno_convs = FactorizedSpectralConv2d(
            width,
            width,
            modes,
            modes,
            n_layers=self.n_layers,
            implementation="factorized",
            fft_norm="forward",
            factorization=factorization,
            joint_factorization=joint_factorization,
            bias=False,
            rank=rank,
        )
        return fno_convs

    def get_skips(self, width):
        return nn.ModuleList(
            [nn.Conv2d(width, width, 1, bias=False) for _ in range(self.n_layers)]
        )

    def get_norms(self, width):
        return nn.ModuleList([nn.BatchNorm2d(width) for _ in range(self.n_layers)])

    @staticmethod
    def get_conv(in_channels, out_channels):
        return nn.Conv2d(in_channels, out_channels, 1, bias=True)

    @staticmethod
    def get_pad(padding):
        return nn.ConstantPad2d(padding, 0.0)

    def forward(self, x):
        x = self.conv_in(x)
        x = self.pad_in(x)
        for idx in range(self.n_layers):
            x_fno = self.fft_convs[idx](x)
            x_skip = self.skips[idx](x)
            x = x_fno + x_skip
            x = self.norms[idx](x)
            x = self.act(x)

        x = self.pad_out(x)
        x = self.conv_out(x)
        return x


class TFNOModel3d(nn.Module):
    def __init__(
        self,
        modes,
        width,
        blocks,
        padding,
        factorization,
        joint_factorization,
        rank,
        out_channels,
    ):
        super().__init__()
        self.n_layers = blocks
        self.fft_convs = self.get_fft_convs(
            modes, width, factorization, joint_factorization, rank
        )
        self.skips = self.get_skips(width)
        self.norms = self.get_norms(width)
        self.conv_in = self.get_conv(1, width)
        self.conv_out = self.get_conv(width, out_channels)
        self.pad_in = self.get_pad(padding)
        self.pad_out = self.get_pad(-padding)
        self.act = nn.GELU()

    def get_fft_convs(self, modes, width, factorization, joint_factorization, rank):
        fno_convs = FactorizedSpectralConv3d(
            width,
            width,
            modes,
            modes,
            modes,
            n_layers=self.n_layers,
            implementation="factorized",
            fft_norm="forward",
            factorization=factorization,
            joint_factorization=joint_factorization,
            bias=False,
            rank=rank,
        )
        return fno_convs

    def get_skips(self, width):
        return nn.ModuleList(
            [nn.Conv3d(width, width, 1, bias=False) for _ in range(self.n_layers)]
        )

    def get_norms(self, width):
        return nn.ModuleList([nn.BatchNorm3d(width) for _ in range(self.n_layers)])

    @staticmethod
    def get_conv(in_channels, out_channels):
        return nn.Conv3d(in_channels, out_channels, 1, bias=True)

    @staticmethod
    def get_pad(padding):
        return nn.ConstantPad3d(padding, 0.0)

    def forward(self, x):
        x = self.conv_in(x)
        x = self.pad_in(x)

        for idx in range(self.n_layers):
            x_fno = self.fft_convs[idx](x)
            x_skip = self.skips[idx](x)
            x = x_fno + x_skip
            x = self.norms[idx](x)
            x = self.act(x)

        x = self.pad_out(x)
        x = self.conv_out(x)
        return x
