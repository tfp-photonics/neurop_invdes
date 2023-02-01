import torch
import torch.nn as nn

from ._base import VAEBase


class VAE2d(VAEBase):
    name = "vae2d"

    @staticmethod
    def get_encoder(input_dim, channels, output_dim):
        return Encoder2d(input_dim, channels, output_dim)

    @staticmethod
    def get_decoder(input_dim, channels, output_dim):
        return Decoder2d(input_dim, channels, output_dim)


class VAE3d(VAEBase):
    name = "vae3d"

    @staticmethod
    def get_encoder(input_dim, channels, output_dim):
        return Encoder3d(input_dim, channels, output_dim)

    @staticmethod
    def get_decoder(input_dim, channels, output_dim):
        return Decoder3d(input_dim, channels, output_dim)


class Encoder(nn.Module):
    def __init__(self, input_dim, channels, output_dim):
        super().__init__()

        self.act = nn.SELU()

        self.conv_in = self.get_conv(
            1, channels[0], kernel_size=1, stride=1, padding=0, bias=True
        )

        blocks = []
        for ii, channel in enumerate(channels[:-1]):
            blocks += [
                self.get_conv(
                    channel,
                    channels[ii + 1],
                    kernel_size=5,
                    stride=2,
                    padding=2,
                    bias=False,
                ),
                self.get_bn(channels[ii + 1]),
                self.act,
            ]
        self.conv_blocks = nn.Sequential(*blocks)

        self.linear = nn.Sequential(
            nn.Linear(
                channels[-1] * int(input_dim / 2 ** (len(channels) - 1)) ** self.dim,
                output_dim,
            ),
            self.act,
        )

    def forward(self, x):
        x = self.conv_in(x)
        x = self.conv_blocks(x)
        x = torch.flatten(x, 1)
        x = self.linear(x)
        return x


class Encoder2d(Encoder):
    dim = 2

    @staticmethod
    def get_conv(in_channels, out_channels, kernel_size, stride, padding, bias):
        return nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )

    @staticmethod
    def get_bn(features):
        return nn.BatchNorm2d(features)


class Encoder3d(Encoder):
    dim = 3

    @staticmethod
    def get_conv(in_channels, out_channels, kernel_size, stride, padding, bias):
        return nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )

    @staticmethod
    def get_bn(features):
        return nn.BatchNorm3d(features)


class Decoder(nn.Module):
    def __init__(self, input_dim, channels, output_dim):
        super().__init__()

        self.act = nn.SELU()

        self.linear = nn.Sequential(
            nn.Linear(
                input_dim,
                channels[0] * int(output_dim / 2 ** (len(channels) - 1)) ** self.dim,
            ),
            self.act,
        )

        self.unflatten = self.get_unflatten(channels, output_dim)

        blocks = []
        for ii, channel in enumerate(channels[:-1]):
            blocks += [
                self.get_conv_transpose(
                    channel,
                    channels[ii + 1],
                    kernel_size=5,
                    stride=2,
                    padding=2,
                    output_padding=1,
                    bias=False,
                ),
                self.get_bn(channels[ii + 1]),
                self.act,
            ]
        self.conv_blocks = nn.Sequential(*blocks)

        self.conv_out = self.get_conv(channels[-1], 1)

    def forward(self, x):
        x = self.linear(x)
        x = self.unflatten(x)
        x = self.conv_blocks(x)
        x = self.conv_out(x)
        x = torch.sigmoid(x)
        return x


class Decoder2d(Decoder):
    dim = 2

    @staticmethod
    def get_conv_transpose(
        in_channels, out_channels, kernel_size, stride, padding, output_padding, bias
    ):
        return nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            bias=bias,
        )

    @staticmethod
    def get_conv(in_channels, out_channels):
        return nn.Conv2d(in_channels, out_channels, kernel_size=1)

    @staticmethod
    def get_bn(features):
        return nn.BatchNorm2d(features)

    @staticmethod
    def get_unflatten(channels, output_dim):
        return nn.Unflatten(
            dim=1,
            unflattened_size=(
                channels[0],
                int(output_dim / 2 ** (len(channels) - 1)),
                int(output_dim / 2 ** (len(channels) - 1)),
            ),
        )


class Decoder3d(Decoder):
    dim = 3

    @staticmethod
    def get_conv_transpose(
        in_channels, out_channels, kernel_size, stride, padding, output_padding, bias
    ):
        return nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            bias=bias,
        )

    @staticmethod
    def get_conv(in_channels, out_channels):
        return nn.Conv3d(in_channels, out_channels, kernel_size=1)

    @staticmethod
    def get_bn(features):
        return nn.BatchNorm3d(features)

    @staticmethod
    def get_unflatten(channels, output_dim):
        return nn.Unflatten(
            dim=1,
            unflattened_size=(
                channels[0],
                int(output_dim / 2 ** (len(channels) - 1)),
                int(output_dim / 2 ** (len(channels) - 1)),
                int(output_dim / 2 ** (len(channels) - 1)),
            ),
        )
