import torch
from torch import nn
import torch.nn.functional as F
from Hyperparameters import *

# 本文件所有动态卷积的 MSCM(Multi-Scale Counting Module)进行符号计数统计，作为解码器的辅助输入
# 灵感来源：https://arxiv.org/pdf/2207.11463，致谢！

class ChannelAttention(nn.Module):
    # 通道注意力机制 GAP -> liner -> relu -> liner -> sigmoid
    def __init__(self, channel, reduction):
        """
        通道注意力机制
        Args:
            channel (int):    输入通道数
            reduction (int):  衰减系数
        """
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(channel, channel // reduction),
                                nn.ReLU(),
                                nn.Linear(channel // reduction, channel),
                                nn.Sigmoid())

    def forward(self, x):
        """
        前向传播
        Args:
            x (tensor):      输入的图像特征信息         -> [Batch, Channel, Height, Width]
        Return:
            x * y (tensor):  通道注意力机制后的结果     -> [Batch, Channel, Height, Width]
        """
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y
    

class DynamicConv2d(nn.Module):
    # 动态卷积层：根据输入动态生成卷积核权重
    def __init__(self, in_channels, out_channels, kernel_list, num_experts=4):
        """
        Args:
            in_channels (int):   输入通道数
            out_channels (int):  输出通道数
            kernel_list (list):  候选卷积核大小列表(如[3, 5, 7])
            num_experts (int):   专家数量（动态权重生成的基数）
        """
        super(DynamicConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_list = kernel_list
        self.num_experts = num_experts
        
        # 为每个候选核大小生成专家权重
        self.weights = nn.ParameterList([
            nn.Parameter(torch.randn(num_experts, out_channels, in_channels, kernel_size, kernel_size), requires_grad=True)
            for kernel_size in kernel_list
            ])
        
        # 动态权重生成器（轻量级网络）
        self.router = nn.Sequential(nn.AdaptiveAvgPool2d(1), 
                                    nn.Flatten(), 
                                    nn.Linear(in_channels, num_experts),  # 输出维度与专家数量匹配
                                    nn.Softmax(dim=1))
        
    def forward(self, x):
        """
        Args:
            x (torch.Tensor):     输入的图像特征信息         -> [B, in_channels, H, W]
        Returns:
            sum (torch.Tensor):   多尺度结果求和累加         -> [B, out_channels, H, W]
        """
        B, _, H, W = x.shape

        # 生成动态的权重[B, num_experts]
        routing_weight = self.router(x)

        # 对每个候选核大小计算动态卷积结果
        outputs = []
        for idx, kernel_size in enumerate(self.kernel_list):
            # 合并多专家权重[B, out_channels, in_channels, kernel_size, kernel_size]
            combined_weight = torch.einsum("be,eocij->bocij", routing_weight, self.weights[idx])

            # 动态卷积计算
            x_reshaped = x.reshape(1, -1, H, W)
            combined_weight = combined_weight.view(-1, self.in_channels, kernel_size, kernel_size)

            output = F.conv2d(x_reshaped, 
                              combined_weight, 
                              stride=1, 
                              padding=(kernel_size // 2, kernel_size // 2),  # 根据实际核大小调整填充
                              groups=B)
            
            outputs.append(output.view(B, -1, H, W))

        # 多尺度结果求和
        return sum(outputs)

class CountingDecoder(nn.Module):
    # 动态MSCM模块支持自适应卷积核
    def __init__(self, in_channels, out_channels, kernel_list: list):
        """
        Args:
            in_channels (int):  输入通道数
            out_channels (int): 输出通道数(符号类别数)
            kernel_list (list): 候选卷积核大小列表(如[3, 5, 7])
        """
        super(CountingDecoder, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # 动态卷积层 -> 通道注意力层 -> 1x1卷积层 -> sigmoid激活 -> sum pooling求和累加
        self.DynamicConv = DynamicConv2d(in_channels, 512, kernel_list=kernel_list, num_experts=parameters['MSCM']['num_experts'])
        self.channel_att = ChannelAttention(512, 16)
        self.mask_proj = nn.Conv2d(parameters['ViT_Encoder']['embedding_dim'], 1, kernel_size=1) 

        self.pred_layer = nn.Sequential(nn.Conv2d(512, out_channels, kernel_size=1, bias=False), nn.Sigmoid())

        self.device = parameters['device']
        self.to(self.device)
        
    def forward(self, x, mask=None):
        """
        Args:
            x (torch.tensor):               图像特征信息      -> [B, 128, 10, 10]
            mask (torch.tensor):            掩码特征信息      -> B, 128, 10, 10]
        Returns:
            count_vector (torch.tensor):    符号统计向量      -> [B, out_channels]
            density_map (torch.tensor):     密度图           -> [B, out_channels, H, W]
        """
        x = x.to(self.device)
        B, C, H, W = x.shape

        x = self.DynamicConv(x)
        x = self.channel_att(x)

        # 处理掩码信息，用掩码加权特征
        if mask is not None:
            mask = mask.to(self.device)
            mask_feature = self.mask_proj(mask)  # [B, 1, 10, 10]
            x = x * mask_feature                 # [B, 512, 10, 10] * [B, 1, 10, 10]

        x = self.pred_layer(x)

        # [B, C] 在所有空间维度上求和
        count_vector = torch.sum(x, dim=(-1, -2))  
        # [B, C, H, W]
        density_map = x.view(B, self.out_channels, H, W)
        return count_vector, density_map
