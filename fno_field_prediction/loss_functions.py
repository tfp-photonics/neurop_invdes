import math

import torch
import torch.nn.functional as F
from scipy.constants import epsilon_0, mu_0


def relative_error(x, y, p, reduction="mean"):
    x = x.reshape(x.shape[0], -1)
    y = y.reshape(y.shape[0], -1)
    diff_norms = torch.linalg.norm(x - y, p, 1)
    y_norms = torch.linalg.norm(y, p, 1)
    if reduction == "mean":
        return torch.mean(diff_norms / y_norms)
    elif reduction == "none":
        return diff_norms / y_norms
    else:
        raise ValueError("Only reductions 'mean' and 'none' supported!")


class LPLoss(torch.nn.Module):
    def __init__(self, p=2, reduction="mean"):
        super().__init__()
        self.p = p
        self.reduction = reduction

    @torch.jit.ignore
    def forward(self, x, y):
        return relative_error(x, y, self.p, self.reduction)


class HSLoss(torch.nn.Module):
    """
    Sobolev norm, essentially the difference between numerical derivatives.
    This is not used in the paper, but we played around with it a bit and you can, too!
    """

    def __init__(self, p=2):
        super().__init__()
        self.p = p

    @torch.jit.ignore
    def forward(self, x, y):
        s = x.shape[0]
        nx, ny = x.shape[-2:]

        x = x.view(s, nx, ny, -1)
        y = y.view(s, nx, ny, -1)

        k_x = (
            torch.cat(
                (
                    torch.arange(nx // 2, device=x.device),
                    torch.arange(nx // 2, 0, -1, device=x.device),
                ),
            )
            .view(nx, 1)
            .repeat(1, ny)
            .view(1, nx, ny, 1)
        )
        k_y = (
            torch.cat(
                (
                    torch.arange(ny // 2, device=x.device),
                    torch.arange(ny // 2, 0, -1, device=x.device),
                ),
            )
            .view(1, ny)
            .repeat(nx, 1)
            .view(1, nx, ny, 1)
        )

        x = torch.fft.fftn(x, dim=[1, 2])
        y = torch.fft.fftn(y, dim=[1, 2])

        w = torch.sqrt(1 + k_x**2 + k_y**2)

        return relative_error(w * x, w * y, self.p)


class MaxwellLoss(torch.nn.Module):
    """
    This is more or less a copy of the Maxwell loss implemented in
    https://github.com/jonfanlab/waveynet, except we add a way to calculate it via a
    direct convolution with a Laplacian, which works just as well.

    The loss is not used in the paper and mostly included for convenience and
    completeness, as we do have some pre-trained models for this loss.
    Note however that this loss does not work with Meep fields, but only for fields
    generated from fdfdpy (https://github.com/fancompute/fdfdpy) and related solvers
    due to the choice of units.
    """

    def __init__(self, lambda0, dl, loss_fn=F.l1_loss, method="explicit"):
        super().__init__()
        c0 = math.sqrt(1 / epsilon_0 / mu_0)
        omega = 2 * math.pi * c0 / lambda0
        self.const_fac = 1 / dl**2 / omega**2 / mu_0 / epsilon_0
        self.loss_fn = loss_fn
        if method == "explicit":
            self.ez_to_ez_fun = self.ez_to_ez_explicit
        elif method == "conv":
            self.ez_to_ez_fun = self.ez_to_ez_conv
        else:
            raise RuntimeError(f"Unknown method: {method}")

    @torch.jit.ignore
    def forward(self, ez, eps):
        """Note that this expects real input like (B, 2, H, W)"""
        ez_ = torch.view_as_complex(ez.permute(0, 2, 3, 1).contiguous())
        ez_hat = self.ez_to_ez_fun(ez_, eps)
        loss = self.loss_fn(ez_[..., 1:-1, 1:-1], ez_hat[..., 1:-1, 1:-1])
        return loss

    @staticmethod
    def dxb(x):
        x_ = F.pad(x, (0, 0, 1, 0))
        return x_[..., 1:, :] - x_[..., :-1, :]

    @staticmethod
    def dyb(x):
        x_ = F.pad(x, (1, 0, 0, 0))
        return x_[..., :, 1:] - x_[..., :, :-1]

    @staticmethod
    def dxf(x):
        x_ = F.pad(x, (0, 0, 0, 1))
        return x_[..., :-1, :] - x_[..., 1:, :]

    @staticmethod
    def dyf(x):
        x_ = F.pad(x, (0, 1, 0, 0))
        return x_[..., :, :-1] - x_[..., :, 1:]

    def ez_to_hx(self, ez):
        return 1j * self.dyb(ez)

    def ez_to_hy(self, ez):
        return -1j * self.dxb(ez)

    def hxhy_to_ez(self, hx, hy, eps):
        return 1j * (self.dxf(hy) - self.dyf(hx)) / eps

    def ez_to_ez_explicit(self, ez, eps):
        hx = self.ez_to_hx(ez)
        hy = self.ez_to_hy(ez)
        ez_hat = self.hxhy_to_ez(hx, hy, eps) * self.const_fac
        return ez_hat

    def ez_to_ez_conv(self, ez, eps):
        k = torch.tensor(
            [[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=ez.dtype, device=ez.device
        )
        ez_conv = F.conv2d(ez[:, None], k[None, None], padding=1).squeeze()
        ez_conv = -ez_conv / eps * self.const_fac
        return ez_conv
