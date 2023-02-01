import pytorch_lightning as pl
import torch
import torch.distributions
import torch.nn as nn


class BaseModel(pl.LightningModule):
    def forward(self, x):
        return self.model(x)

    @torch.jit.ignore
    def training_step(self, batch, batch_idx):
        x, y = batch.values()
        y_pred = self(x.unsqueeze(1))
        loss = self.l2_rel(y_pred, y)
        if self.hparams.with_maxwell_loss:
            loss = loss + self.mloss_l2(y_pred, x)
        self.log("train_loss", loss)
        return loss

    @torch.jit.ignore
    def validation_step(self, batch, batch_idx):
        x, y = batch.values()
        y_pred = self(x.unsqueeze(1))
        loss = self.l2_rel(y_pred, y)
        if self.hparams.with_maxwell_loss:
            loss = loss + self.mloss_l2(y_pred, x)

        l1_rel = self.l1_rel(y_pred, y)
        l2_rel = self.l2_rel(y_pred, y)
        self.log("val_loss", loss, sync_dist=True)
        self.log("l1_loss", l1_rel, sync_dist=True)
        self.log("l2_loss", l2_rel, sync_dist=True)

        if "2d" in self.name:
            h1_rel = self.h1_rel(y_pred, y)
            h2_rel = self.h2_rel(y_pred, y)
            mloss_l1 = self.mloss_l1(y_pred, x)
            mloss_l2 = self.mloss_l2(y_pred, x)
            self.log("h1_loss", h1_rel, sync_dist=True)
            self.log("h2_loss", h2_rel, sync_dist=True)
            self.log("l1_mwloss", mloss_l1, sync_dist=True)
            self.log("l2_mwloss", mloss_l2, sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

        conf = {"optimizer": optimizer}

        if self.hparams.scheduler == "none":
            pass
        elif self.hparams.scheduler == "onecycle":
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=10 * self.hparams.lr,
                total_steps=self.hparams.epochs,
                base_momentum=0.85,
                max_momentum=0.95,
            )
            conf.update(
                {
                    "lr_scheduler": {
                        "scheduler": scheduler,
                        "interval": "epoch",
                        "frequency": 1,
                    }
                }
            )

        elif self.hparams.scheduler == "exponential":
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
            conf.update(
                {
                    "lr_scheduler": {
                        "scheduler": scheduler,
                        "interval": "epoch",
                        "frequency": 1,
                    }
                }
            )
        else:
            raise ValueError(f"Invalid scheduler: {self.hparams.scheduler}")

        return conf


class VAEBase(pl.LightningModule):
    def __init__(
        self,
        latent_dim,
        channels_encode,
        channels_decode,
        input_dim,
        output_dim,
        kld_weight,
        kld_weight_annealing,
        bin_weight,
        bin_cutoff,
        bin_weight_annealing,
        lr,
        weight_decay,
        steps,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.encoder = self.get_encoder(input_dim, channels_encode, latent_dim)
        self.decoder = self.get_decoder(latent_dim, channels_decode, output_dim)

        self.fc_mu = nn.Linear(latent_dim, latent_dim)
        self.fc_log_var = nn.Linear(latent_dim, latent_dim)
        self.log_scale = nn.Parameter(torch.Tensor([0.0]))

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("VAE")
        parser.add_argument("--latent-dim", type=int, default=256)
        parser.add_argument(
            "--channels-encode", type=int, nargs="+", default=[8, 16, 32, 64, 128, 256]
        )
        parser.add_argument(
            "--channels-decode", type=int, nargs="+", default=[256, 128, 64, 32, 16, 8]
        )
        parser.add_argument(
            "--kld-weight",
            type=float,
            default=0.2,
            help="KLD loss term weight. If annealing is used, this defines the maximum.",
        )
        parser.add_argument(
            "--kld-weight-annealing",
            type=float,
            nargs=3,
            default=None,
            help="""
            Annealing parameters [min, slope, offset (fraction of steps)]
            Good defaults to try are [1e-3, 2e-3, 0.4].,
            """,
        )
        parser.add_argument(
            "--bin-cutoff",
            type=float,
            default=3.0,
            help="Binarization log cutoff. Defines up to which decimal binarization should apply.",
        )
        parser.add_argument(
            "--bin-weight",
            type=float,
            default=0.0,
            help="Binarization term weight. If annealing is used, this defines the maximum.",
        )
        parser.add_argument(
            "--bin-weight-annealing",
            type=float,
            nargs=3,
            default=None,
            help="""
            Annealing parameters [min, slope, offset (fraction of steps)].
            Good defaults to try are [0.0, 5e-3, 0.75].
            """,
        )
        return parent_parser

    @torch.jit.export
    def encode(self, x):
        return self.encoder(x)

    @torch.jit.export
    def decode(self, x):
        return self.decoder(x)

    @torch.jit.ignore
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.encode(x)
        p, q, z = self._sample(x)
        return self.decode(z)

    @staticmethod
    def _gaussian_likelihood(x_hat, x, logscale):
        scale = torch.exp(logscale)
        dist = torch.distributions.Normal(x_hat, scale)
        log_pxz = dist.log_prob(x)
        return torch.flatten(log_pxz, 1).mean(1)

    @staticmethod
    def _kl_divergence(p, q, z):
        log_qzx = q.log_prob(z)
        log_pz = p.log_prob(z)
        kld = log_qzx - log_pz
        return kld.mean(-1)

    @staticmethod
    def _binarization(x, cutoff):
        b = -torch.log10(torch.sum(4 * x * (1 - x)) / float(x.numel()))
        return torch.minimum(b, cutoff) / cutoff

    @staticmethod
    def _sigmoid_anneal(x, minimum, maximum, slope, offset):
        return minimum + (maximum - minimum) / (
            1 + torch.exp(torch.tensor(-slope * (x - offset)))
        )

    def _get_annealed_weight(self, batch_idx, weight, params):
        if params is None or weight == 0.0:
            return weight
        else:
            minimum, slope, offset = params
            offset *= self.hparams.steps
            return self._sigmoid_anneal(batch_idx, minimum, weight, slope, offset)

    def _sample(self, x):
        mu = self.fc_mu(x)
        log_var = self.fc_log_var(x)
        log_var = torch.clamp(log_var, -10, 10)
        std = torch.exp(log_var / 2)
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()
        return p, q, z

    def step(self, batch, batch_idx):
        x = batch.unsqueeze(1)
        x_enc = self.encoder(x)
        p, q, z = self._sample(x_enc)
        x_hat = self.decoder(z)

        # Note: Reconstruction loss is scaled by input size, KL divergence is
        # scaled by latent size. We get rid of this scaling by taking the mean
        # instead of the sum, which would correspond to the mathematical definition.
        # This makes the weighting of the terms independent of latent and input sizes.
        reconstruction_loss = -self._gaussian_likelihood(x_hat, x, self.log_scale)

        kld = self._kl_divergence(p, q, z)
        kld_weight = self._get_annealed_weight(
            batch_idx, self.hparams.kld_weight, self.hparams.kld_weight_annealing
        )

        cutoff = torch.tensor(self.hparams.bin_cutoff).type_as(x)
        binarization = self._binarization(x_hat, cutoff)
        bin_weight = self._get_annealed_weight(
            batch_idx, self.hparams.bin_weight, self.hparams.bin_weight_annealing
        )

        elbo = kld_weight * kld + reconstruction_loss - bin_weight * binarization
        elbo = elbo.mean()  # batch mean

        logs = {
            "reconstruction_loss": reconstruction_loss.mean(),
            "kld": kld.mean(),
            "binarization": binarization,
            "kld_weight": kld_weight,
            "bin_weight": bin_weight,
            "elbo": elbo,
        }
        return elbo, logs

    def training_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict(logs, on_step=True, on_epoch=False)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=10 * self.hparams.lr,
            total_steps=self.hparams.steps,
            base_momentum=0.85,
            max_momentum=0.95,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }
