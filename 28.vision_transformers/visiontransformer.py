"""Implementation of Vision Transformers."""

## Necessary imports ##

import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image

from torchvision.transforms import transforms
from torchvision.utils import make_grid

import matplotlib.pyplot as plt


## Scaled Dot Product Attention ##


class ScaledDotProductAttention(nn.Module):
    """Implementation of the scaled dot product attention."""

    def __init__(self, embedding_dim):
        """Constructor.

        Parameters
        ----------
        embedding_dim : int
            Dimension of the channels for the query, key and values.
        """

        super().__init__()

        # Query mapping
        self.query_map = nn.LazyLinear(out_features=embedding_dim, bias=False)

        # Key mapping
        self.key_map = nn.LazyLinear(out_features=embedding_dim, bias=False)

        # Value mapping
        self.value_map = nn.LazyLinear(out_features=embedding_dim, bias=False)

    def forward(self, x, return_wt=False):
        """Forward Pass.

        Parameters
        ----------
        x : torch.tensor
            Input tensor

        return_wt : bool [Default : False]
            Flag when set to True, returns the weight,
            i.e., softmax(Q @ K.T / sqrt(embedding_dim)).

        Returns
        -------
        torch.tensor
            The result of the matrix multiplication of weight and value.

        [Optional]
        torch.tensor
            Weight (Only when return_wt flag is set to True).
        """

        # Query, key, value
        query, key, value = self.query_map(x), self.key_map(x), self.value_map(x)

        B, N, C = key.shape  # B -> batch, N -> Number of patches, C -> Each patch shape

        # Weight
        wei = (query @ key.transpose(-2, -1)) / C**0.5
        wei = F.softmax(wei, dim=-1)

        if return_wt:
            return wei @ value, wei

        return wei @ value


## Multihead Attention ##


class MultiheadAttention(nn.Module):
    """Implements Multihead Attention module."""

    def __init__(self, embedding_dim, num_heads):
        """Constructor.

        Parameters
        ----------
        embedding_dim : int
            Dimension of the final output channel.

        num_heads : int
            Number of heads for multi-head attention.
        """

        super().__init__()

        # Setting each head output dim
        each_head_dim = embedding_dim // num_heads

        self.multi_head = nn.ModuleList(
            [ScaledDotProductAttention(each_head_dim) for _ in range(num_heads)]
        )

    def forward(self, x):
        """Forward Pass.

        Parameters
        ----------
        x : torch.tensor
            Input tensor

        Returns
        -------
        torch.tensor
            Output value.
        """

        return torch.cat([head(x) for head in self.multi_head], dim=-1)


## MLP ##


class MLP(nn.Module):
    """Implements the Pointwise FeedForward Network."""

    def __init__(self, embedding_dim):
        """Constructor

        Parameters
        ----------
        embedding_dim : int
            Dimension of the final output.
        """

        super().__init__()

        self.mlp = nn.Sequential(
            nn.LazyLinear(4 * embedding_dim, bias=False),
            nn.GELU(),
            nn.LazyLinear(embedding_dim, bias=False),
        )

    def forward(self, x):
        """Forward Pass.

        Parameters
        ----------
        x : torch.tensor
            Input tensor.

        Returns
        -------
        torch.tensor
            Output of the feedforward network.
        """

        return self.mlp(x)


## Residual Connection ##


class NormDropoutResidual(nn.Module):
    """Implements LayernNorm, then Sublayer passthrough, dropout
    and then Residual Connection."""

    def __init__(self, embedding_dim, sublayer, dropout_rate=0.1):
        """Constructor.

        Parameters
        ----------
        embedding_dim : int
            Dimension of the internal channel dimension.

        sublayer : nn.Module
            The alternate path block of residual connection.

        dropout_rate : float
            Dropout rate of the Dropout layers. [Default : 0.1]
        """

        super().__init__()

        self.block = nn.Sequential(
            nn.LayerNorm(embedding_dim), sublayer, nn.Dropout(dropout_rate)
        )

    def forward(self, x):
        """Forward Pass.

        Parameters
        ----------
        x : torch.tensor
            Input tensor.

        Returns
        -------
        torch.tensor
            Output of the residual connection.
        """

        return x + self.block(x)


## Encoder Layer ##


class EncoderLayer(nn.Module):
    """Implements a single Encoder Layer."""

    def __init__(self, embedding_dim, num_heads, dropout_rate=0.1):
        """Constructor.

        Parameters
        ----------
        embedding_dim : int
            Dimension of the internal channel dimension.

        num_heads : int
            Number of heads for multi-head attention.

        dropout_rate : float
            Dropout rate of the Dropout layers. [Default : 0.1]
        """

        super().__init__()

        self.block = nn.Sequential(
            NormDropoutResidual(
                embedding_dim,
                MultiheadAttention(embedding_dim, num_heads),
                dropout_rate,
            ),
            NormDropoutResidual(embedding_dim, MLP(embedding_dim), dropout_rate),
        )

    def forward(self, x):
        """Forward Pass.

        Parameters
        ----------
        x : torch.tensor
            Input tensor.

        Returns
        -------
        torch.tensor
            Output of a single encoder layer.
        """
        return self.block(x)


## Positional Encoding ##


class PositionalEncoding(nn.Module):
    """Implementation of the Positional Encoding"""

    def __init__(self):
        """Constuctor"""

        super().__init__()

    def forward(self, sequence_length, embedding_dim, device):
        """Forward Pass.

        Parameters
        ----------
        sequence_length : int
            Sequence length or the total number of patches
            that has been created from a single image.

        embedding_dim : int
            Dimension of the internal channel dimension.
        """

        positions = torch.arange(sequence_length).unsqueeze(-1)
        i = torch.arange(embedding_dim)

        position_embedding = torch.empty((sequence_length, embedding_dim))

        position_embedding[:, ::2] = torch.sin(
            positions / 1000 ** (i[::2] / embedding_dim)
        )
        position_embedding[:, 1::2] = torch.cos(
            positions / 1000 ** (i[1::2] / embedding_dim)
        )

        return position_embedding.to(device)


## Patch Embedding ##


class PatchEmbedding(nn.Module):
    """Implements the Patch Embedding."""

    def __init__(self, patch_shape):
        """Constructor.

        Parameters
        ----------
        patch_shape : tuple
            The shape of each patch that is created
            from an image.

        """

        super().__init__()

        self.patch_shape = patch_shape

        self.patch_embed = nn.Unfold(kernel_size=patch_shape, stride=patch_shape)

    def forward(self, img):
        """Forward pass.

        Parameters
        ----------
        img : torch.tensor
            Input image.

        Returns
        -------
        torch.tensor
            Output patches all combined in [B , N , C x P x P],
            where P is patch size, C is channels, N is number
            of patches and B is batch size.
        """

        B, C, _, _ = img.shape

        out = self.patch_embed(img)

        out = out.transpose(-2, -1)

        return out


## Encoder ##


class Encoder(nn.Module):
    """Implementation of the Encoder Block."""

    def __init__(self, embedding_dim, num_heads, num_layers, dropout_rate=0.1):
        """Constructor.

        Parameters
        ----------
        embedding_dim : int
            Dimension of the internal channel dimension.

        num_heads : int
            Number of heads for multi-head attention.

        num_layers : int
            Number of EncoderLayers that are repeated in the
            Encoder.

        dropout_rate : float
            Dropout rate of the Dropout layers. [Default : 0.1]
        """

        super().__init__()

        self.encoder = nn.ModuleList(
            [
                EncoderLayer(embedding_dim, num_heads, dropout_rate)
                for _ in range(num_layers)
            ]
        )

    def forward(self, x):
        """Forward Pass.

        Parameters
        ----------
        x : torch.tensor
            Input tensor.

        Returns
        -------
        torch.tensor
            Returns the output of the encoder.
        """

        for layer in self.encoder:
            x = layer(x)

        return x


## Vision Transformer ##


class VisionTransformer(nn.Module):
    """Implements the Vision Transformer Module."""

    def __init__(
        self,
        patch_shape,
        embedding_dim,
        num_heads,
        num_layers,
        num_classes,
        dropout_rate=0.1,
    ):
        """Constructor.

        Parameters
        ----------
        patch_shape : tuple
            The shape of each patch that is created
            from an image.

        embedding_dim : int
            Dimension of the internal channel dimension.

        num_heads : int
            Number of heads for multi-head attention.

        num_layers : int
            Number of EncoderLayers that are repeated in the
            Encoder.

        num_classes : int
            Number of classes in the classifier.

        dropout_rate : float
            Dropout rate of the Dropout layers. [Default : 0.1]
        """

        super().__init__()

        self.patch_embedding = PatchEmbedding(patch_shape)

        self.linear_projection_patches = nn.LazyLinear(out_features=embedding_dim)

        self.class_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))

        self.position_embedding = PositionalEncoding()

        self.encoder = Encoder(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout_rate=dropout_rate,
        )

        self.classifer = nn.LazyLinear(num_classes)

    def forward(self, img, device=torch.device("cpu")):
        """Forward Pass.

        Parameter
        ---------
        img : torch.tensor
            Image Tensor.

        Returns
        -------
        torch.tensor
            Output logits.
        """

        img_patches = self.patch_embedding(img)

        img_patches = self.linear_projection_patches(img_patches)

        # Adding class token

        B, L, C = img_patches.shape

        img_patches = torch.cat(
            [self.class_token.expand(B, -1, -1), img_patches], dim=1
        )

        positional_embedding = self.position_embedding(
            sequence_length=L + 1, embedding_dim=C, device=device
        )

        img_patches += positional_embedding

        encoder_out = self.encoder(img_patches)

        encoder_out_class_token = encoder_out[:, 0, :]

        return self.classifer(encoder_out_class_token)
