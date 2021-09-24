
from torch import nn
import torch

class SiameseConcatView(nn.Module):
    """
    This head is useful for dealing with Siamese models which have multiple towers.
    For an input of type (N * num_towers) x C, this head can convert the output
    to N x (num_towers * C).
    This head is used in case of PIRL https://arxiv.org/abs/1912.01991 and
    Jigsaw https://arxiv.org/abs/1603.09246 approaches.

    code from https://github.com/facebookresearch/vissl/blob/aa3f7cc33b3b7806e15593083aedc383d85e4a53/vissl/models/heads/siamese_concat_view.py
    MIT Licensed
    """

    def __init__(self, num_towers: int):
        """
        Args:
            model_config (AttrDict): dictionary config.MODEL in the config file
            num_towers (int): number of towers in siamese model
        """
        super().__init__()
        self.num_towers = num_towers

    def forward(self, batch: torch.Tensor):
        """
        Args:
            batch (torch.Tensor): 2D torch tensor `(N * num_towers) x C` or 4D tensor of
                                  shape `(N * num_towers) x C x 1 x 1`
        Returns:
            out (torch.Tensor): 2D output torch tensor `N x (C * num_towers)`
        """
        # batch dimension = (N * num_towers) x C x H x W
        siamese_batch_size = batch.shape[0]
        assert (
            siamese_batch_size % self.num_towers == 0
        ), f"{siamese_batch_size} not divisible by num_towers {self.num_towers}"
        batch_size = siamese_batch_size // self.num_towers
        out = batch.view(batch_size, -1)
        return out



class AlexNetJigsaw(nn.Module):
    def __init__(self, num_patches=5, num_classes: int = 1000) -> None:
        super(AlexNetJigsaw, self).__init__()
        self.num_patches = num_patches
        self.patch_model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.AdaptiveAvgPool2d((6, 6)),
            nn.Flatten(1),
        )
        self.unpatch = SiameseConcatView(self.num_patches)
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, patched_x: torch.Tensor) -> torch.Tensor:
        patched_x = self.patch_model(patched_x)
        x = self.unpatch(patched_x)
        x = self.classifier(x)
        return x

