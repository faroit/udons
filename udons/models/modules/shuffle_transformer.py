from torch import nn
import numpy as np
import torch
from torch import nn

def generate_square_subsequent_mask(sz: int) -> torch.Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)


class ShuffleTransformer(nn.Module):
    """Transformer model that unshuffles the sequence"""
    def __init__(self, hparams: dict):
        super().__init__()

        dim = hparams["model_dim"]
        patch_dim = hparams["patch_len"] * hparams["n_mels"] * hparams["nb_channels"]
        self.patch_model = nn.Linear(patch_dim, dim)

        # make target sequence causal
        self.tgt_mask = generate_square_subsequent_mask(hparams["nb_patches"])
        # use absolute positional encodings
        self.pos_embedding = nn.Parameter(torch.randn(hparams["nb_patches"], dim))
        # vanilla transformer
        self.transformer = nn.Transformer(d_model=dim, batch_first=True)
        self.generator = nn.Linear(dim, patch_dim)

    def forward(self, patches, permutation=None):
        x = self.patch_model(patches)

        # do not add pos embedding to the source sequence
        # but add it to the target sequence and deshuffle sequence
        if permutation is None:
            tgt = x
            tgt_mask = None
        else:
            tgt = x[:, permutation, ...]
            tgt_mask = self.tgt_mask
    
        tgt += self.pos_embedding
        out = self.transformer(src=x, tgt=tgt, tgt_mask=tgt_mask)
        x = self.generator(out)
        return x