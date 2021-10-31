from udons.models.modules.shuffle_transformer import ShuffleTransformer
from typing import Any, List

import itertools
import numpy as np
import torch
from pytorch_lightning import LightningModule
from einops.layers.torch import Rearrange


class ShuffleTransformerModel(LightningModule):
    """Transformer Model that predicts the classification task but applies it via soft-indexing to the original input"""
    def __init__(
        self,
        nb_classes: int = 120,
        nb_channels: int = 1,
        nb_patches: int = 5,
        n_mels: int = 256,
        patch_len: int = 32,
        depth: int = 3,
        heads: int = 3,
        pool = 'cls',
        dim_head = 64,
        model_dim = 1024,
        mlp_dim = 512,
        dropout = 0.,
        emb_dropout = 0.,
        lr: float = 0.001,
        weight_decay: float = 0.0005,
    ):
        super().__init__()

        # this line ensures params passed to LightningModule will be saved to ckpt
        # it also allows to access params with 'self.hparams' attribute
        self.save_hyperparameters()
        self.model = ShuffleTransformer(hparams=self.hparams)
        self.unbatch = Rearrange('(b p) c f t -> b p c f t', p=self.hparams["nb_patches"])
        self.patch = Rearrange('b p c f t -> b p (c f t)')
        self.unpatch = Rearrange('b p (c f t) -> b p c f t', c=self.hparams["nb_channels"], f=self.hparams["n_mels"])
        self.permutations = torch.tensor(
            np.array(
                list(itertools.permutations(list(range(self.hparams["nb_patches"]))))
            )
        )
        # loss function
        self.criterion = torch.nn.MSELoss()

    def forward(self, x: torch.Tensor, y: None):
        return self.model(x, y)

    def train_step(self, batch: Any):
        x, permutation = batch
        x = self.unbatch(x)
        x = self.patch(x)
        pred = self.forward(x, permutation)
        pred = self.unpatch(pred)
        loss = self.criterion(pred, x)
        return loss, pred, permutation

    def test_step(self, batch: Any):
        x, permutation = batch
        x = self.unbatch(x)
        x = self.patch(x)
        pred = self.forward(x)
        pred = self.unpatch(pred)
        loss = self.criterion(pred, x)
        return loss, pred, permutation

    def training_step(self, batch: Any, batch_idx: int):
        x, permutation = batch
        x = self.unbatch(x)
        x = self.patch(x)
        pred = self.forward(x, permutation)
        pred = self.unpatch(pred)
        loss = self.criterion(pred, x)
        return loss, pred, permutation)

        # we can return here dict with any tensors
        # and then read it in some callback or in training_epoch_end() below
        # remember to always return loss from training_step, or else backpropagation will fail!
        return {"loss": loss, "pred": pred, "permutation": permutation}

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        loss, pred, permutation = self.step(batch)

        return {"loss": loss, "pred": pred, "permutation": permutation}

    def validation_epoch_end(self, outputs: List[Any]):
        pass

    def test_step(self, batch: Any, batch_idx: int):
        loss, pred, permutation = self.step(batch)

        return {"loss": loss, "pred": pred, "permutation": permutation}

    def test_epoch_end(self, outputs: List[Any]):
        pass

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        See examples here:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        return torch.optim.Adam(
            params=self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )
