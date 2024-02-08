import torch
from torch import nn
from Model.ASTEncoder import ASTEncoder
from configs.config import FEATURES

class SiameseNetwork(nn.Module):
    def __init__(self, siamese_embed_size, embed_size, hidden_size, output_size, num_layers, num_heads, dropout=0.1):
        super(SiameseNetwork, self).__init__()

        # 定义一个共享的AST编码器
        self.ast_encoder = ASTEncoder(embed_size, hidden_size, output_size, num_layers, num_heads, dropout)


        # 定义一个用于计算相似度的全连接层
        self.similarity_layer = nn.Sequential(
            nn.Linear(siamese_embed_size, embed_size),
            nn.ReLU(),
            nn.Linear(embed_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, embed_l, embed_r, features_l=None, features_r=None):
        # 编码AST1和AST2
        # ast_l_encoded = self.ast_encoder(embed_l)
        # ast_r_encoded = self.ast_encoder(embed_r)
        if features_l and features_r:
            feature_l_tensors = [features_l[feature] for feature in FEATURES]
            feature_r_tensors = [features_r[feature] for feature in FEATURES]
            # ast_l_encoded = torch.cat([ast_l_encoded] + feature_l_tensors, dim=1).to(torch.float)
            # ast_r_encoded = torch.cat([ast_r_encoded] + feature_r_tensors, dim=1).to(torch.float)

            embed_l = torch.cat([embed_l] + feature_l_tensors, dim=1).to(torch.float)
            embed_r = torch.cat([embed_r] + feature_r_tensors, dim=1).to(torch.float)

        # 计算AST1和AST2的相似度得分
        # diff = torch.cat((ast_l_encoded, ast_r_encoded), dim=1)
        # ast_l_encoded = torch.nn.functional.normalize(ast_l_encoded, p=2, dim=1)
        # ast_r_encoded = torch.nn.functional.normalize(ast_r_encoded, p=2, dim=1)
        # diff = torch.abs(ast_l_encoded - ast_r_encoded)

        embed_l = torch.nn.functional.normalize(embed_l, p=2, dim=1)
        embed_r = torch.nn.functional.normalize(embed_r, p=2, dim=1)
        diff = torch.abs(embed_l - embed_r)

        similarity_score = self.similarity_layer(diff)
        return similarity_score.squeeze()
