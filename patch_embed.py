from src.models.jigsaw_transformer_model import JigsawTransformerModel
from einops.layers.torch import Rearrange

import torch
import torchaudio
from hearbaseline.util import frame_audio


# Number of frames to batch process for timestamp embeddings
BATCH_SIZE = 512

class JigSawEmbedder(torch.nn.Module):
    """
    JigSaw Embedder class
    """
    def __init__(self, model: torch.nn.Module):
        """
        Initialize JigSaw Embedder
        """
        super().__init__()
        self.sample_rate = 16000
        self.transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=400,
            hop_length=100,
            f_min=27.5,
            f_max=16000,
            n_mels=128,
        )
        modules = [Rearrange('b c f t -> b (c f t)')]
        modules.extend(list(model.patch_encoding.children())[1:])
        self.model = torch.nn.Sequential(*modules)
    
    def get_activation(self, name):
        activation = {}
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook

    def forward(self, x):
        """
        Forward pass
        """

        audio = self.transform(x)
        return self.model(audio)

def load_model(model_file_path: str = "") -> torch.nn.Module:
    """
    Returns a torch.nn.Module that produces embeddings for audio.
    Args:
        model_file_path: Load model checkpoint from this file path. For this baseline,
            if no path is provided then the default random init weights for the
            linear projection layer will be used.
    Returns:
        Model
    """
    jigsaw_model = JigsawTransformerModel.load_from_checkpoint(checkpoint_path=model_file_path)
    model = JigSawEmbedder(jigsaw_model.model)
    model.eval()
    model.sample_rate = 16000  # sample rate
    model.embedding_size = 768  # model_dim  TODO: get from configs

    return model


def get_timestamp_embeddings(
    audio: torch.Tensor,
    model: torch.nn.Module,
):
    """
    This function returns embeddings at regular intervals centered at timestamps. Both
    the embeddings and corresponding timestamps (in milliseconds) are returned.
    Args:
        audio: n_sounds x n_samples of mono audio in the range [-1, 1].
        model: Loaded model.

    Returns:
        - Tensor: embeddings, A float32 Tensor with shape (n_sounds, n_timestamps,
            model.timestamp_embedding_size).
        - Tensor: timestamps, Centered timestamps in milliseconds corresponding
            to each embedding in the output. Shape: (n_sounds, n_timestamps).
    """

    # Send the model to the same device that the audio tensor is on.
    model = model.to(audio.device)

    # Split the input audio signals into frames and then flatten to create a tensor
    # of audio frames that can be batch processed.
    frames, timestamps = frame_audio(
        audio,
        frame_size=1599,
        hop_size=50,
        sample_rate=16000,
    )
    audio_batches, num_frames, frame_size = frames.shape
    frames = frames.flatten(end_dim=1)
    

    # We're using a DataLoader to help with batching of frames
    dataset = torch.utils.data.TensorDataset(frames)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False
    )

    # Put the model into eval mode, and not computing gradients while in inference.
    # Iterate over all batches and accumulate the embeddings for each frame.
    model.eval()
    with torch.no_grad():
        embeddings_list = [model(batch[0][:, None]) for batch in loader]

    # Concatenate mini-batches back together and unflatten the frames
    # to reconstruct the audio batches
    embeddings = torch.cat(embeddings_list, dim=0)
    embeddings = embeddings.unflatten(0, (audio_batches, num_frames))

    return embeddings, timestamps


def get_scene_embeddings(
    audio: torch.Tensor,
    model: torch.nn.Module,
) -> torch.Tensor:
    """
    This function returns a single embedding for each audio clip. In this baseline
    implementation we simply summarize the temporal embeddings from
    get_timestamp_embeddings() using torch.mean().
    Args:
        audio: n_sounds x n_samples of mono audio in the range [-1, 1]. All sounds in
            a batch will be padded/trimmed to the same length.
        model: Loaded model.
    Returns:
        - embeddings, A float32 Tensor with shape
            (n_sounds, model.scene_embedding_size).
    """
    embeddings, _ = get_timestamp_embeddings(audio, model)
    embeddings = torch.mean(embeddings, dim=1)
    return embeddings


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=str, help="model directory")
    args = parser.parse_args()
    model = load_model(args.model_path)
    x = torch.rand(16, 1, 1599)
    get_timestamp_embeddings(torch.rand(8, 44100), model)
