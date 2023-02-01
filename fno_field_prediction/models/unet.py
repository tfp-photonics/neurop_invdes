import torch
import torch.nn as nn
import torch.nn.functional as F

from ..loss_functions import HSLoss, LPLoss, MaxwellLoss
from ._base import BaseModel


class UNet(BaseModel):
    name = "unet"

    def __init__(
        self,
        alpha,
        num_down_conv,
        hidden_dim,
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
        self.h1_rel = HSLoss(1)
        self.h2_rel = HSLoss(2)
        self.mloss_l1 = MaxwellLoss(lambda0, dl, loss_fn=self.l1_rel)
        self.mloss_l2 = MaxwellLoss(lambda0, dl, loss_fn=self.l2_rel)
        self.model = UNetModel(alpha, num_down_conv, hidden_dim, out_channels)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("UNet")
        parser.add_argument("--alpha", type=float, default=0.3)
        parser.add_argument("--num-down-conv", type=int, default=6)
        parser.add_argument("--hidden-dim", type=int, default=16)
        return parent_parser


class UNetModel(nn.Module):
    """Straight from https://github.com/jonfanlab/waveynet"""

    def __init__(self, alpha, num_down_conv, hidden_dim, outc):
        super().__init__()
        config = []
        for block in range(num_down_conv):

            kernel_size = 3
            out_channels = (2**block) * hidden_dim
            if block == 0:
                config += [
                    (
                        "conv2d",
                        [out_channels, 1, kernel_size, kernel_size, 1, 1, True],
                    )
                ]
            else:
                config += [
                    (
                        "conv2d",
                        [
                            out_channels,
                            out_channels // 2,
                            kernel_size,
                            kernel_size,
                            1,
                            1,
                            True,
                        ],
                    ),
                ]
            config += [
                ("bn", [out_channels]),
                ("leakyrelu", [alpha, False]),
            ]
            config += [
                (
                    "conv2d",
                    [out_channels, out_channels, kernel_size, kernel_size, 1, 1, False],
                ),
                ("bn", [out_channels]),
                ("leakyrelu", [alpha, False]),
            ]
            config += [
                (
                    "conv2d",
                    [out_channels, out_channels, kernel_size, kernel_size, 1, 1, False],
                ),
                ("bn", [out_channels]),
            ]
            config += [("residual", []), ("leakyrelu", [alpha, False])]
            config += [
                (
                    "conv2d",
                    [out_channels, out_channels, kernel_size, kernel_size, 1, 1, True],
                ),
                ("bn", [out_channels]),
                ("leakyrelu", [alpha, False]),
            ]
            config += [
                (
                    "conv2d",
                    [out_channels, out_channels, kernel_size, kernel_size, 1, 1, False],
                ),
                ("bn", [out_channels]),
                ("leakyrelu", [alpha, False]),
            ]
            config += [
                (
                    "conv2d",
                    [out_channels, out_channels, kernel_size, kernel_size, 1, 1, False],
                ),
                ("bn", [out_channels]),
            ]
            config += [("residual", [])]
            config += [("leakyrelu", [alpha, False])]
            if block < 2:
                config += [("max_pool2d", [(1, 2), (1, 2), 0])]  # kernel_size, padding
            else:
                config += [("max_pool2d", [(2, 2), (2, 2), 0])]
        for block in range(num_down_conv - 1):
            out_channels = (2 ** (num_down_conv - block - 2)) * hidden_dim
            in_channels = out_channels * 3
            config += [("upsample", [2])]
            config += [
                ("conv2d", [out_channels, in_channels, 3, 3, 1, 1, True]),
                ("bn", [out_channels]),
                ("leakyrelu", [alpha, False]),
            ]
            config += [
                ("conv2d", [out_channels, out_channels, 3, 3, 1, 1, False]),
                ("bn", [out_channels]),
                ("leakyrelu", [alpha, False]),
            ]
            config += [
                ("conv2d", [out_channels, out_channels, 3, 3, 1, 1, False]),
                ("bn", [out_channels]),
            ]
            config += [("residual", []), ("leakyrelu", [alpha, False])]
            config += [
                ("conv2d", [out_channels, out_channels, 3, 3, 1, 1, True]),
                ("bn", [out_channels]),
                ("leakyrelu", [alpha, False]),
            ]
            config += [
                ("conv2d", [out_channels, out_channels, 3, 3, 1, 1, False]),
                ("bn", [out_channels]),
                ("leakyrelu", [alpha, False]),
            ]
            config += [
                ("conv2d", [out_channels, out_channels, 3, 3, 1, 1, False]),
                ("bn", [out_channels]),
            ]
            config += [("residual", [])]
            config += [("leakyrelu", [alpha, False])]

        # all the conv2d before are without bias, and this conv_b is with bias
        config += [("conv2d_b", [outc, hidden_dim, 3, 3, 1, 1])]
        self.config = config

        # this dict contains all tensors needed to be optimized
        self.vars = nn.ParameterList()
        # running_mean and running_var
        self.vars_bn = nn.ParameterList()
        self.loss_fn = nn.L1Loss()
        self.optimizer = None  # will be initialized in waveynet_trainer.py
        self.lr_scheduler = None  # will be initialized in waveynet_trainer.py
        self.residual_terms = None  # to store the residual connect for addition later

        for i, (name, param) in enumerate(self.config):
            if name == "conv2d":
                # [ch_out, ch_in, kernelsz, kernelsz]
                w = nn.Parameter(torch.ones(*param[:4]), requires_grad=True)
                # gain=1 according to cbfin's implementation
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
            elif name == "conv2d_b":
                # [ch_out, ch_in, kernelsz, kernelsz]
                w = nn.Parameter(torch.ones(*param[:4]), requires_grad=True)
                # gain=1 according to cbfin's implementation
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                # [ch_out]
                self.vars.append(
                    nn.Parameter(torch.zeros(param[0]), requires_grad=True)
                )

            elif name == "bn":
                # [ch_out]
                w = nn.Parameter(torch.ones(param[0]), requires_grad=True)
                self.vars.append(w)
                # [ch_out]
                self.vars.append(
                    nn.Parameter(torch.zeros(param[0]), requires_grad=True)
                )

                # must set requires_grad=False
                running_mean = nn.Parameter(torch.zeros(param[0]), requires_grad=False)
                running_var = nn.Parameter(torch.ones(param[0]), requires_grad=False)
                self.vars_bn.extend([running_mean, running_var])

            elif name in [
                "tanh",
                "relu",
                "upsample",
                "avg_pool2d",
                "max_pool2d",
                "flatten",
                "reshape",
                "leakyrelu",
                "sigmoid",
                "residual",
            ]:
                continue
            else:
                raise NotImplementedError

    def forward(self, x, vars=None, bn_training=True):
        """
        Defining how the data flows through the components initialized in the
        __init__ function, defining the model
        """
        if vars is None:
            vars = self.vars

        idx = 0
        bn_idx = 0
        first_upsample = True

        blocks = []
        for name, param in self.config:
            if name == "conv2d":
                w = vars[idx]
                # remember to keep synchrozied of forward_encoder and forward_decoder!
                x = F.conv2d(x, w, stride=param[4], padding="same")
                idx += 1
                if param[6]:
                    self.residual_terms = torch.clone(x)

            elif name == "conv2d_b":
                w, b = vars[idx], vars[idx + 1]
                # remember to keep synchrozied of forward_encoder and forward_decoder!

                x = F.conv2d(x, w, b, stride=param[4], padding="same")
                idx += 2

            elif name == "bn":
                w, b = vars[idx], vars[idx + 1]
                running_mean, running_var = (
                    self.vars_bn[bn_idx],
                    self.vars_bn[bn_idx + 1],
                )
                x = F.batch_norm(
                    x, running_mean, running_var, weight=w, bias=b, training=bn_training
                )
                idx += 2
                bn_idx += 2

            elif name == "leakyrelu":
                x = F.leaky_relu(x, negative_slope=param[0], inplace=param[1])

            elif name == "upsample":
                if first_upsample:
                    first_upsample = False
                    x = blocks.pop()
                shortcut = blocks.pop()
                x = F.interpolate(
                    x, size=(shortcut.shape[2], shortcut.shape[3]), mode="nearest"
                )
                x = torch.cat([shortcut, x], dim=1)  # batch, channels, h, w

            elif name == "residual":
                x = x + self.residual_terms

            elif name == "max_pool2d":
                blocks.append(x)
                x = F.max_pool2d(x, param[0], stride=param[1], padding=param[2])

            else:
                raise NotImplementedError(name)

        # make sure variable is used properly
        assert idx == len(vars)
        assert bn_idx == len(self.vars_bn)

        return x

    def zero_grad(self, vars=None):
        with torch.no_grad():
            if vars is None:
                for p in self.vars:
                    if p.grad is not None:
                        p.grad.zero_()
            else:
                for p in vars:
                    if p.grad is not None:
                        p.grad.zero_()

    def parameters(self):
        """
        override parameters since initial parameters will return with a generator.
        """
        return self.vars
