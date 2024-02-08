import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch


class ASTEncoder(nn.Module):
    def __init__(self, embed_dim, hidden_dim, output_dim, num_layers, num_heads, dropout=0.1):
        super(ASTEncoder, self).__init__()

        self.dropout = nn.Dropout(p=dropout)

        # 编码器
        encoder_layers = TransformerEncoderLayer(embed_dim, num_heads, hidden_dim, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)

        # 添加一个额外的线性层
        self.linear_layer = nn.Linear(embed_dim, output_dim)

    def forward(self, embedded_ast):
        embedded_ast = self.dropout(embedded_ast)

        # Transformer Encoding
        encoded_vec = self.transformer_encoder(embedded_ast)

        # layer
        encoded_vec = self.linear_layer(encoded_vec)

        # Sum-pooling
        encoded_vec = torch.sum(encoded_vec, dim=1)

        return encoded_vec
