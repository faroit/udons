from typing import Optional, Tuple
from pathlib import Path


from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.transforms import transforms

from src.datamodules.datasets.jigsawaudio_dataset import JigsawAudioDataset


class JigsawAudioDataModule(LightningDataModule):
    """
    Example of LightningDataModule dataset.

    A DataModule implements 5 key methods:
        - prepare_data (things to do on 1 GPU/TPU, not on every GPU/TPU in distributed mode)
        - setup (things to do on every accelerator in distributed mode)
        - train_dataloader (the training dataloader)
        - val_dataloader (the validation dataloader(s))
        - test_dataloader (the test dataloader(s))

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html
    """

    def __init__(
        self,
        data_dir: str = "data/",
        train_dir: str = "train",
        test_dir: str = "test",
        valid_dir: str = "valid",
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        sample_rate: int = 44100,
        nb_timesteps: int = 44100 * 3,
        nb_channels: int = 1,
    ):
        super().__init__()

        self.data_dir = data_dir
        self.train_dir = train_dir
        self.valid_dir = valid_dir
        self.test_dir = test_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.sample_rate = sample_rate

        self.nb_channels = nb_channels
        self.nb_timesteps = nb_timesteps

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        # self.dims is returned when you call datamodule.size()
        self.dims = None

    def prepare_data(self):
        """Download data if needed. This method is called only from a single GPU.
        Do not use it to assign state (self.x = y)."""
        pass

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: self.data_train, self.data_val, self.data_test."""
        self.train_set = JigsawAudioDataset(
            root=Path(self.data_dir, self.train_dir),
            min_chunk_length=self.nb_timesteps,
            random_chunk_length=self.nb_timesteps,
            nb_channels=self.nb_channels,
        )
        self.test_set = JigsawAudioDataset(
            root=Path(self.data_dir, self.train_dir),
            min_chunk_length=self.nb_timesteps,
            random_chunk_length=self.nb_timesteps,
            nb_channels=self.nb_channels,
        )
        self.valid_set = JigsawAudioDataset(
            root=Path(self.data_dir, self.train_dir),
            min_chunk_length=self.nb_timesteps,
            random_chunk_length=self.nb_timesteps,
            nb_channels=self.nb_channels,
        )

        self.dims = self.train_set[0][0].shape

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.valid_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )
