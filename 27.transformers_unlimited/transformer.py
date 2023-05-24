## Importing necessary packages ##

import torch
import torch.nn as nn
import torch.nn.functional as F

## Scaled Dot Product Attention ##


class ScaledDotProductAttention(nn.Module):
    """Implements the scaled dot product attention."""

    def __init__(
        self,
        d_embed: int,
        d_k: int,
        d_v: int = None,
        mask: bool = False,
    ):
        """Constructor"""

        super().__init__()

        # If d_v is not specified set to same as d_k #
        if d_v == None:
            d_v = d_k

        # Query, key and values linear layers #
        self.query_ffn = nn.Linear(d_embed, d_k, bias=False)
        self.key_ffn = nn.Linear(d_embed, d_k, bias=False)
        self.value_ffn = nn.Linear(d_embed, d_v, bias=False)

        self.mask = mask

    def forward(
        self,
        x: torch.tensor,
        y: torch.tensor = None,
        z: torch.tensor = None,
        return_wt: bool = False,
    ):
        """Applies a forward pass through the Scaled Dot Product Attention."""

        # Getting the key, query and the value
        key = self.key_ffn(x)

        if y == None:
            y = x.clone()

        if z == None:
            z = x.clone()

        query = self.query_ffn(y)
        value = self.value_ffn(z)

        # Getting d_k from query
        d_k = key.shape[-1]

        # Calculating weights
        weight = query @ key.transpose(-2, -1) / d_k**0.5

        # Setting mask
        if self.mask:
            weight = torch.tril(weight)
            weight = weight.masked_fill(weight == 0, float("-inf"))

        # Pass through softmax
        weight = F.softmax(weight, dim=-1)

        # Finally product with the values and return
        if return_wt:
            return weight @ value, weight

        return weight @ value


## Multihead Attention ##


class MultiHeadAttention(nn.Module):
    """Implements the Multihead Attention."""

    def __init__(
        self,
        d_embed: int,
        num_heads: int,
        d_k: int = None,
        mask: bool = False,
    ):
        """Constructor"""

        super().__init__()

        d_v = d_embed // num_heads
        if d_k == None:
            d_k = d_v

        self.multi_heads = nn.ModuleList(
            [
                ScaledDotProductAttention(
                    d_embed=d_embed,
                    d_k=d_k,
                    d_v=d_v,
                    mask=mask,
                )
                for _ in range(num_heads)
            ]
        )

        self.mask = mask

        # self.register_buffer('d_v', torch.tensor([d_v]))

    def forward(self, x: torch.tensor, y: torch.tensor = None, z: torch.tensor = None):
        """Forward Pass"""
        # d_v = self.d_v.item()

        return torch.cat([head(x, y, z) for head in self.multi_heads], dim=-1)


## Position wise Feed Forward Networks ##


class PositionWiseFFN(nn.Module):
    """Implements the Position Wise Feed Forward Networks"""

    def __init__(self, d_embed: int):
        """Constructor"""

        super().__init__()

        self.pwffn = nn.Sequential(
            nn.Linear(in_features=d_embed, out_features=4 * d_embed, bias=False),
            nn.ReLU(),
            nn.Linear(in_features=4 * d_embed, out_features=d_embed, bias=False),
        )

    def forward(self, x):
        """Forward Pass"""

        return self.pwffn(x)


## Postion Encoding ##


def PositionalEncoding(seq_length: int, d_embed: int, device="cpu"):
    """Positional Embedding"""

    pos = torch.arange(seq_length).unsqueeze(-1)
    i = torch.arange(d_embed)

    position_embedding = torch.empty((seq_length, d_embed))

    position_embedding[:, ::2] = torch.sin(pos / 1000 ** (i[::2] / d_embed))
    position_embedding[:, 1::2] = torch.cos(pos / 1000 ** (i[1::2] / d_embed))

    return position_embedding.to(device)


## Residual Dropout Norm ##


class ResidualDropoutNorm(nn.Module):
    """Implements the Residual Dropout Norm Layer."""

    def __init__(
        self, sublayer: nn.Module, d_embed: int = 512, dropout_rate: float = 0.1
    ):
        """Constructor."""

        super().__init__()

        self.sublayer = nn.ModuleList([sublayer, nn.Dropout(dropout_rate)])
        self.layer_norm = nn.LayerNorm(d_embed)

    def forward(self, x, y: torch.tensor = None, z: torch.tensor = None):
        """Forward Pass."""

        out = x.clone()

        for layer in self.sublayer:
            if type(layer) == MultiHeadAttention and layer.mask:
                out = layer(out, y, z)
            else:
                out = layer(out)
        x = x + out

        return self.layer_norm(x)


## Encoder Layer ##


class EncoderLayer(nn.Module):
    """Encoder Layer."""

    def __init__(
        self,
        d_embed: int,
        num_heads: int,
        dropout_rate: float,
        d_k: int = None,
        mask: bool = False,
    ):
        """Constructor."""

        super().__init__()

        self.encoder_layer = nn.ModuleList(
            [
                ResidualDropoutNorm(
                    d_embed=d_embed,
                    sublayer=MultiHeadAttention(
                        d_embed=d_embed,
                        num_heads=num_heads,
                        d_k=d_k,
                        mask=mask,
                    ),
                    dropout_rate=dropout_rate,
                ),
                ResidualDropoutNorm(
                    d_embed=d_embed,
                    sublayer=PositionWiseFFN(d_embed),
                    dropout_rate=dropout_rate,
                ),
            ]
        )

    def forward(self, x, y: torch.tensor = None, z: torch.tensor = None):
        """Forward pass through a single encoder layer"""

        for layer in self.encoder_layer:
            out = layer(x, y, z)

        return out


## Decoder Layer ##


class DecoderLayer(nn.Module):
    """Implements a single decoder layer."""

    def __init__(
        self,
        d_embed: int,
        num_heads: int,
        dropout_rate: float,
        d_k_sa: int = None,
        use_ca: bool = False,
        d_k_ca: int = None,
    ):

        """Constructor."""

        super().__init__()

        self.masked_attn = ResidualDropoutNorm(
            sublayer=MultiHeadAttention(
                d_embed=d_embed,
                num_heads=num_heads,
                d_k=d_k_sa,
                mask=True,
            ),
            d_embed=d_embed,
            dropout_rate=dropout_rate,
        )

        self.cross_attn = (
            ResidualDropoutNorm(
                sublayer=MultiHeadAttention(
                    d_embed=d_embed,
                    num_heads=num_heads,
                    d_k=d_k_ca,
                    mask=False,
                ),
                d_embed=d_embed,
                dropout_rate=dropout_rate,
            )
            if use_ca
            else None
        )

        self.pwfn = ResidualDropoutNorm(
            d_embed=d_embed,
            sublayer=PositionWiseFFN(d_embed),
            dropout_rate=dropout_rate,
        )

    def forward(self, z: torch.tensor, x: torch.tensor = None, y: torch.tensor = None):
        """Forward Pass through a single decoder layer."""

        z = self.masked_attn(x=z)

        if self.cross_attn != None:
            z = self.cross_attn(x, y, z)

        return self.pwfn(z)


## Encoder block ##


class Encoder(nn.Module):
    """Encoder Block"""

    def __init__(
        self,
        d_embed: int,
        num_heads: int,
        dropout_rate: float,
        d_k: int = None,
        mask: bool = False,
        num_layers: int = 6,
    ):
        """Constructor"""

        super().__init__()
        self.encoder = nn.ModuleList(
            [
                EncoderLayer(d_embed, num_heads, dropout_rate, d_k, mask)
                for _ in range(num_layers)
            ]
        )

    def forward(self, inp, return_intermediate=False):
        """Forward Pass"""

        intermediate_val = torch.tensor([])

        for layer in self.encoder:
            inp = layer(inp)
            intermediate_val = torch.cat([intermediate_val, inp.unsqueeze(0)], dim=0)

        if return_intermediate:
            return intermediate_val[-1], intermediate_val

        return intermediate_val[-1]


## Decoder Block ##


class Decoder(nn.Module):
    """Decoder Block"""

    def __init__(
        self,
        d_embed: int,
        num_heads: int,
        dropout_rate: float,
        d_k_sa: int = None,
        use_ca: bool = False,
        d_k_ca: int = None,
        num_layers: int = 6,
    ):

        super().__init__()

        self.decoder = nn.ModuleList(
            [
                DecoderLayer(
                    d_embed,
                    num_heads,
                    dropout_rate,
                    d_k_sa,
                    use_ca,
                    d_k_ca,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, z, intermediate_value=None):
        """Forward Pass"""

        for i, layer in enumerate(self.decoder):
            if intermediate_value != None:
                z = layer(z=z, x=intermediate_value[i])

            else:
                z = layer(z=z)

        return z


## The transformer module ##


class Transformer(nn.Module):
    """Transformer module."""

    def __init__(
        self,
        vocab_size: int,
        sequence_length: int,
        d_embed: int,
        use_encoder: bool,
        use_decoder: bool,
        num_heads: int,
        d_k_encoder: int = None,
        encoder_mask: bool = False,
        encoder_num_layers: int = 6,
        decoder_num_layers: int = 6,
        dropout_rate: float = 0.1,
        classification: bool = False,
        num_classes: int = None,
        use_ca: bool = False,
        d_k_sa: int = None,
        d_k_ca: int = None,
        mask: bool = False,
        device: torch.device = torch.device("cpu"),
    ):
        """Constructor"""

        super().__init__()

        self.embed = nn.Embedding(vocab_size, d_embed)

        # positional encoding network #
        # self.pos_embed = nn.Embedding(
        #     num_embeddings=sequence_length, embedding_dim=d_embed
        # )

        # self.pos_embed = PositionalEncoding(sequence_length, d_embed, device)

        self.final_output = (
            nn.Linear(d_embed, num_classes)
            if classification
            else nn.Linear(d_embed, vocab_size)
        )

        self.encoder = None
        self.decoder = None

        if use_encoder:
            self.encoder = Encoder(
                d_embed,
                num_heads,
                dropout_rate,
                d_k_encoder,
                mask,
                num_layers=encoder_num_layers,
            )

        if use_decoder:
            self.decoder = Decoder(
                d_embed,
                num_heads,
                dropout_rate,
                d_k_sa,
                use_ca,
                d_k_ca,
                num_layers=decoder_num_layers,
            )

        self.device = device
        self.register_buffer("use_ca", torch.tensor([use_ca]))
        self.register_buffer("decoder_num_layers", torch.tensor([decoder_num_layers]))
        self.register_buffer("sequence_length", torch.tensor([sequence_length]))
        self.register_buffer("classification", torch.tensor([classification]))

    def forward(self, inputs=None, outputs=None):
        """Forward Pass."""

        intermediate = None

        if self.encoder != None:
            inputs = self.embed(inputs)
            B, L, C = inputs.shape
            pos_embed = PositionalEncoding(L, C, self.device)
            # positional embedding #
            # pos = torch.arange(L, device=device).repeat(B, 1)
            # pos_embed = self.pos_embed(pos)
            inputs = inputs + pos_embed
            inputs, intermediate = self.encoder(inputs, return_intermediate=True)

            if self.decoder == None or self.classification.item() == True:
                return self.final_output(inputs)

        if self.decoder != None:
            outputs = self.embed(outputs)
            B, L, C = outputs.shape
            pos_embed = PositionalEncoding(L, C, self.device)
            # positional embedding #
            # pos = torch.arange(L, device=self.device).repeat(B, 1)
            # pos_embed = self.pos_embed(pos)
            outputs = outputs + pos_embed

            if (intermediate != None and self.use_ca.item() == True) or (
                intermediate != None
            ):
                outputs = self.decoder(outputs, intermediate)

            elif self.use_ca:
                intermediate = torch.cat(
                    [outputs.unsqueeze(0) for _ in range(self.decoder_num_layers)],
                    dim=0,
                )
                outputs = self.decoder(outputs, intermediate)

            else:
                outputs = self.decoder(outputs)

        return self.final_output(outputs)

    ## Utility function to show data ##

    def _show_data(self, data, idx_2_char_map, verbose=True):
        """Given a data tensor, maps them to string and prints them."""
        str_data = [idx_2_char_map[each_word.item()] for each_word in data.data]
        if verbose:
            print(str_data)
        else:
            return str_data

    def generate(
        self,
        max_length: int,
        idx_2_char_map: dict,
        device: torch.device = torch.device("cpu"),
    ):
        """Generative bit."""

        idx = torch.zeros((1, 1), device=device).int()

        assert self.decoder != None, "Decoder must be present!!"

        for _ in range(max_length):
            logits = self(outputs=idx[:, -self.sequence_length :])
            logits = logits[:, -1, :]
            prob = torch.nn.functional.softmax(logits, dim=-1)
            next_idx = torch.multinomial(prob, 1).int()
            idx = torch.cat([idx, next_idx], dim=1)

        print("".join(self._show_data(idx[0, 1:].data, idx_2_char_map, verbose=False)))
