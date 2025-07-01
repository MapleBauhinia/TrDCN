import torch
from torch import nn
import math
from Hyperparameters import parameters

# 本文件使用Vision Transformer(ViT) 作为编码器，提取图像信息

class PatchEmbedding(nn.Module):
    # 将图像进行分割固定大小的patch并嵌入
    def __init__(self):
        super().__init__()
        self.ImageSize = parameters['ImageSize']
        self.PatchSize = parameters['ViT_Encoder']['PatchSize']
        self.in_channels = parameters['ViT_Encoder']['in_channels']

        # 输入Transformer的嵌入维度
        self.embedding_dim = parameters['ViT_Encoder']['embedding_dim']
        
        # 分割后的块数
        self.grid_size = (self.ImageSize[0] // self.PatchSize[0], self.ImageSize[1] // self.PatchSize[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
       
        # 通过卷积实现分块和线性投影
        # 对图像特征[B, C, H, W]进行投影
        self.project_image = nn.Conv2d(self.in_channels, self.embedding_dim, kernel_size=self.PatchSize, stride=self.PatchSize)
        # 对掩码特征[B, 1, H, W]进行投影
        self.project_mask = nn.Conv2d(1, self.embedding_dim, kernel_size=self.PatchSize, stride=self.PatchSize)
        
    def forward(self, x):
        """
        Args:
            x (torch.tensor):         图像特征/掩码特征    ->[B, C, H, W] / [B, 1, H, W]
        Returns:
            x (torch.tensor):         投影后的特征         ->[B, N, D]
            self.grid_size (tuple):   分块后的块数         ->(Nh, Nw)
        """
        B, C, H, W = x.shape
        assert H == 100 and W == 500, "Image size must be 100x500!"
        
        if C == 3:
            # 对图像特征[B, C, H, W]进行投影
            # [B, C, H, W] -> [B, D, H//PatchSize[0], W//PatchSize[1]]
            x = self.project_image(x)
        elif C == 1:
            # 对掩码特征[B, 1, H, W]进行投影
            # [B, 1, H, W] -> [B, D, H//PatchSize[0], W//PatchSize[1]]
            x = self.project_mask(x)
        else:
            raise ValueError("Input channel must be 1 or 3!")

        # [B, D, H//PatchSize[0], W//PatchSize[1]] -> [B, D, H//PatchSize[0]*W//PatchSize[1] = N]
        x = x.flatten(2)

        # [B, D, H//PatchSize[0]*W//PatchSize[1] = N] -> [B, N, D]
        x = x.transpose(1, 2)

        # 分别后的结果[B, N, D], (Nh, Nw)积为N
        return x, self.grid_size
    
class PositionalEncoding(nn.Module):
    # 向图像中注入位置编码：Transformer的正弦余弦位置编码
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        """
        Args:
            num_hiddens (int): 隐藏单元数
            dropout (float):   Dropout
            max_len (int):     最大序列长度
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)  # 注册为缓冲区（不参与训练）

    def forward(self, x):
        # x: [B, num_patches, d_model]
        x = x + self.pe[:x.size(1)]  # 仅添加前num_patches个位置编码
        return self.dropout(x)
    
class ViT(nn.Module):
    # 使用Vision Transformer(ViT) 作为编码器，提取图像信息
    def __init__(self):
        super().__init__()
        self.device = parameters['device']
        self.num_patches = (parameters['ImageSize'][0] // parameters['ViT_Encoder']['PatchSize'][0]) * \
                           (parameters['ImageSize'][1] // parameters['ViT_Encoder']['PatchSize'][1])

        self.PatchEmbedding = PatchEmbedding().to(self.device)

        self.PositionalEncoding = PositionalEncoding(
            d_model=parameters['ViT_Encoder']['embedding_dim'],
            dropout=parameters['ViT_Encoder']['dropout'],
            max_len=self.num_patches  # 最大长度设为分块总数
        ).to(self.device)

        encoder_layer = nn.TransformerEncoderLayer(d_model=parameters['ViT_Encoder']['embedding_dim'], 
                                                   nhead=parameters['ViT_Encoder']['num_heads'], 
                                                   dim_feedforward=parameters['ViT_Encoder']['embedding_dim'] * 4, 
                                                   dropout=parameters['ViT_Encoder']['dropout'],
                                                   device=self.device)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=parameters['ViT_Encoder']['num_layers'])

    def forward(self, x):
        """
        Args:
            x (torch.tensor):         图像特征/掩码特征    ->[B, C, H, W] / [B, 1, H, W]
        Returns:
            x (torch.tensor):         投影后的特征         ->[B, D, H, W]
        """
        x = x.to(self.device)

        x, self.grid_size = self.PatchEmbedding(x)
        
        # self.PositionalEncoding = PositionalEncoding(self.grid_size[0] * self.grid_size[1]).to(self.device)
        
        # ViT输出形状为[B, N, D]
        x = self.PositionalEncoding(x)
        x = self.encoder(x)

        # 转成4-D张量[B, D, H, W]
        x = x.reshape(x.shape[0], self.grid_size[0], self.grid_size[1], x.shape[-1])
        x = x.permute(0, 3, 1, 2)
        return x

