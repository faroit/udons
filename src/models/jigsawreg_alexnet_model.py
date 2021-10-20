from src.models.modules.audiojigsaw_net import AlexNetJigsaw
from typing import Any, List

import itertools
import numpy as np
import torch
from pytorch_lightning import LightningModule
from torchmetrics.classification.accuracy import Accuracy
from einops.layers.torch import Rearrange


class JigsawRegAlexnetModel(LightningModule):
    def __init__(
        self,
        nb_patches: int = 5,
        nb_classes: int = 120,
        nb_channels: int = 1,
        n_mels: int = 256,
        hidden_size: int = 256,
        lr: float = 0.001,
        weight_decay: float = 0.0005,
    ):
        super().__init__()

        # this line ensures params passed to LightningModule will be saved to ckpt
        # it also allows to access params with 'self.hparams' attribute
        self.save_hyperparameters()
        self.model = AlexNetJigsaw(hparams=self.hparams)
        self.unpatch = Rearrange('(b p) 1 f t -> b p f t', p=self.hparams["nb_patches"])
        self.permutations = torch.tensor(
            np.array(
                list(itertools.permutations(list(range(self.hparams["nb_patches"]))))
            )
        )
        # loss function
        self.criterion = torch.nn.MSELoss()

        # use separate metric instance for train, val and test step
        # to ensure a proper reduction over the epoch
        self.train_accuracy = Accuracy()
        self.val_accuracy = Accuracy()
        self.test_accuracy = Accuracy()
        self.train_top3_accuracy = Accuracy(top_k=3)
        self.val_top3_accuracy = Accuracy(top_k=3)
        self.test_top3_accuracy = Accuracy(top_k=3)
        self.train_topp_accuracy = Accuracy(top_k=self.hparams["nb_patches"])
        self.val_topp_accuracy = Accuracy(top_k=self.hparams["nb_patches"])
        self.test_topp_accuracy = Accuracy(top_k=self.hparams["nb_patches"])

    def forward(self, x: torch.Tensor):
        return self.model(x)

    def step(self, batch: Any):
        x, permutation = batch
        unpatched_x = self.unpatch(x)

        logits = self.forward(x)

        # get unshuffled x
        softmaxed = torch.softmax(logits, -1)
        unshuffled_x = torch.sum(
            softmaxed[:, torch.arange(len(self.permutations)), None, None, None]
            * unpatched_x[:, self.permutations],
            1,
        )
        ordered_x = unpatched_x[torch.arange(unpatched_x.shape[0])[:, None], self.permutations[permutation], ...]
        loss = self.criterion(unshuffled_x, ordered_x)
        return loss, logits, permutation

    def training_step(self, batch: Any, batch_idx: int):
        loss, logits, targets = self.step(batch)

        # log train metrics
        acc = self.train_accuracy(logits, targets)
        acc3 = self.train_top3_accuracy(logits, targets)
        accp = self.train_topp_accuracy(logits, targets)

        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/acc3", acc3, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/accp", accp, on_step=False, on_epoch=True, prog_bar=True)

        # we can return here dict with any tensors
        # and then read it in some callback or in training_epoch_end() below
        # remember to always return loss from training_step, or else backpropagation will fail!
        return {"loss": loss, "logits": logits, "targets": targets}

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        loss, logits, targets = self.step(batch)

        # log val metrics
        acc = self.val_accuracy(logits, targets)
        acc3 = self.val_top3_accuracy(logits, targets)
        accp = self.val_topp_accuracy(logits, targets)

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc3", acc3, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/accp", accp, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "logits": logits, "targets": targets}

    def validation_epoch_end(self, outputs: List[Any]):
        pass

    def test_step(self, batch: Any, batch_idx: int):
        loss, logits, targets = self.step(batch)

        # log test metrics
        acc = self.test_accuracy(logits, targets)
        acc3 = self.test_top3_accuracy(logits, targets)
        accp = self.test_topp_accuracy(logits, targets)

        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.log("test/acc", acc, on_step=False, on_epoch=True)
        self.log("test/acc3", acc3, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/accp", accp, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "logits": logits, "targets": targets}

    def test_epoch_end(self, outputs: List[Any]):
        pass

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        See examples here:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        return torch.optim.SGD(
            params=self.parameters(),
            lr=self.hparams.lr,
            nesterov=True,
            momentum=0.9,
            weight_decay=self.hparams.weight_decay,
        )
