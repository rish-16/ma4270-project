import torch, math
import torch.nn as nn
import torch.nn.functional as F
    
class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Adds positional encoding to the given tensor.

        Args:
            x: tensor to add PE to [bs, seq_len, embed_dim]

        Returns:
            torch.Tensor: tensor with PE [bs, seq_len, embed_dim]
        """
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)
    
class TransformerWithPE(torch.nn.Module):
    def __init__(self, in_dim: int, out_dim: int, embed_dim: int, num_heads: int, num_layers: int, 
                 layer_norm_eps: float = 1e-5, bias: bool = True):
        super().__init__()

        self.positional_encoding = PositionalEncoding(embed_dim)

        self.encoder_embedding = torch.nn.Linear(
            in_features=in_dim, out_features=embed_dim
        )

        encoder_layer = torch.nn.TransformerEncoderLayer(
            embed_dim,
            nhead=num_heads,
            batch_first=True
        )
        encoder_norm = torch.nn.LayerNorm(embed_dim, eps=layer_norm_eps, bias=bias)
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers, encoder_norm)
        
        # self.decoder_embedding = torch.nn.Linear(
        #     in_features=out_dim, out_features=embed_dim
        # )
        self.output_layer = torch.nn.LazyLinear(
            out_features=out_dim
        )

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        src = self.encoder_embedding(src)
        src = self.positional_encoding(src)
        
        # Get prediction from transformer and map to output dimension [bs, tgt_seq_len, embed_dim]
        pred = self.transformer_encoder(src)
        pred = torch.reshape(pred, (pred.size(0), -1))
        pred = self.output_layer(pred)

        return pred    