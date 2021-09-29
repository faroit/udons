from typing import Callable, List, Tuple, Union
from pathlib import Path
import random
import tqdm

import torch
import torch.utils.data as data
import torchaudio
from torchvision import transforms

import numpy as np
import itertools

from typing import List, Tuple, Union


class TimePatcher(object):
    """Sample patches over last axis of 3d Tensor"""

    def __init__(self, patch_len=36, nb_patches=5, patch_jitter_min=5):
        self.patch_len = patch_len
        self.nb_patches = nb_patches
        self.patch_jitter_min = patch_jitter_min

    def __call__(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = torch.atleast_3d(x)
        patch_jitter_max = (
            x.shape[1] - (self.nb_patches // self.patch_len)
        ) // self.nb_patches
        jitter = np.random.randint(
            self.patch_jitter_min, patch_jitter_max, (self.nb_patches,)
        )
        patches = []
        for i in range(self.nb_patches):
            offset = i * self.patch_len
            patch = x[..., offset + jitter[i] : offset + self.patch_len + jitter[i]]
            patches.append(np.copy(patch))

        return patches


class AudioFolderJigsawDataset(data.Dataset):
    def __init__(
        self,
        root: Union[str, Path],
        transform: Callable = None,
        sample_rate: float = 44100.0,
        random_chunk_length: int = 44100 * 5,
        extensions: List[str] = [".wav", ".flac", ".ogg"],
        nb_channels: int = 1,
        patch_len: int = 32,
        nb_patches: int = 5,
        n_fft: int = 2048,
        hop_length: int = 512,
        f_min: float = 27.5,
        f_max: int = 16000,
        n_mels: int = 256,
        patch_jitter_min: int = 5,
    ):
        self.root = root

        # set sample rate. Files not matching sample rate will be discarded
        self.sample_rate = sample_rate
        self.random_chunk_length = random_chunk_length
        self.nb_channels = nb_channels
        self.extensions = extensions
        self.audio_data = list(tqdm.tqdm(self.get_audio_data()))
        self.nb_patches = nb_patches
        self.patch_len = patch_len
        self.patch_jitter_min = patch_jitter_min
        if transform is None:
            self.transform = transforms.Compose(
                [
                    torchaudio.transforms.MelSpectrogram(
                        sample_rate=self.sample_rate,
                        n_fft=n_fft,
                        hop_length=hop_length,
                        f_min=f_min,
                        f_max=f_max,
                        n_mels=n_mels,
                    ),
                    TimePatcher(
                        patch_len=self.patch_len,
                        nb_patches=self.nb_patches,
                        patch_jitter_min=self.patch_jitter_min,
                    ),
                ]
            )
        else:
            self.tranform = transform

        self.permutations = np.array(
            list(itertools.permutations(list(range(self.nb_patches))))
        )

    def __getitem__(self, index: int) -> torch.Tensor:
        """Get dataset item

        Returns:
            Tuple(torch.Tensor, torch.long): Tuple of audio data and label. Label is the index of the permutation array.
                The audio data is a tensor of shape (nb_patches, nb_channels, nb_mels, nb_frames)
        """
        info = self.audio_data[index]["metadata"]
        if self.random_chunk_length is not None:
            total_samples = info["samples"]
            frame_offset = random.randint(0, total_samples - self.random_chunk_length)
            num_frames = self.random_chunk_length
            audio, _ = torchaudio.load(
                self.audio_data[index]["path"],
                frame_offset=frame_offset,
                num_frames=num_frames,
            )
        else:
            # load full audio signals
            audio, _ = torchaudio.load(self.audio_data[index]["path"])

        # make sure mono signals have channel dim
        audio = torch.atleast_2d(audio)

        if audio.shape[0] > self.nb_channels:
            # apply downmix
            audio = torch.mean(audio, dim=0, keepdim=True)
        elif audio.shape[0] < self.nb_channels:
            return self[index - 1]

        audio = self.transform_audio(audio)
        return self.shuffle_and_get_label(audio, self.permutations)

    def __len__(self) -> int:
        return len(self.audio_data)

    def transform_audio(self, audio: torch.Tensor) -> torch.Tensor:
        if self.transform is not None:
            audio = self.transform(audio)
        return audio

    def get_audio_data(self):
        # iterate over root/split and get all audio files
        # also extract metadata from audio to be used for chunking
        p = Path(self.root)
        for extension in self.extensions:
            for audio_path in p.glob(f"**/*{extension}"):
                info = self.load_info(audio_path)
                if not info["samplerate"] == self.sample_rate:
                    continue
                if info["samples"] < self.random_chunk_length:
                    continue

                yield ({"path": audio_path, "metadata": info})

    def load_info(self, path: str) -> dict:
        """Load audio metadata
        this is a backend_independent wrapper around torchaudio.info
        Args:
            path: Path of filename
        Returns:
            Dict: Metadata with
            `samplerate`, `samples` and `duration` in seconds
        """

        info = {}
        si = torchaudio.info(str(path))
        info["samplerate"] = si.sample_rate
        info["samples"] = si.num_frames
        info["channels"] = si.num_channels
        info["duration"] = info["samples"] / info["samplerate"]
        return info

    def shuffle_and_get_label(self, patches, permutations):
        """modified from
        https://github.com/facebookresearch/vissl/blob/aa3f7cc33b3b7806e15593083aedc383d85e4a53/vissl/data/ssl_transforms/shuffle_img_patches.py

        shuffles list of patches
        Args:
            input_patches (List[torch.tensor]): list of torch tensors
        """
        perm_index = np.random.randint(permutations.shape[0])
        shuffled_patches = [
            torch.FloatTensor(patches[i]) for i in permutations[perm_index]
        ]
        # num_towers x C x H x W
        input_data = torch.stack(shuffled_patches)
        out_label = torch.Tensor([perm_index]).long()
        return {"data": input_data, "label": out_label}


class ConcatDataset(data.Dataset):
    def __init__(self, *datasets: tuple):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i % len(d)] for d in self.datasets)

    def __len__(self):
        return max(len(d) for d in self.datasets)


if __name__ == "__main__":
    import argparse
    import os
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="data/", help="root directory")
    args = parser.parse_args()

    data = AudioFolderJigsawDataset(args.root)
    for audio in data:
        print(audio["data"].shape)