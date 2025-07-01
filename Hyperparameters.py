import torch
import torch.optim

# 数据集元素的索引
IMG_IDX = 0          # 图片（numpy.arrya/torch.tensor）
MASK_IDX = 1         # 掩码（numpy.array/torch.tensor）
LATEX_IDX = 2        # LaTeX文本（str）
TOKEN_IDS_IDX = 2    # 词元索引（numpy.array/torch.tensor）
LENGTH_IDX = 3       # 实际长度（numpy.array/torch.tensor）

# 我们在这里定义模型所需的所有超参数
parameters = {
    # 根据实际需求和设备条件，加载样本的最大数
    'Training_max_number': 1000, 
    'Testing_max_number': 500, 

    # 图像预处理 transforms.Normalize
    'mean': [0.485, 0.456, 0.406],
    'std': [0.229, 0.224, 0.225],

    # 训练图像数据随机翻转概率
    'flip_prob': 0.5,
     
    'device': 'cuda:0' if torch.cuda.is_available() else 'cpu',

    # 图像和掩码统一大小
    'ImageSize': (100, 500),
    'MaskSize': (500, 100),

    # 训练和预测的最大长度
    'max_len': 150,

    'BatchSize': 64,

    # 训练周期以及每一轮参与训练最大批量数
    'epochs': 10,
    'max_batches': 50,
    
    # 优化器以及学习率调度器
    'optimizer': torch.optim.AdamW,
    'scheduler': torch.optim.lr_scheduler.CosineAnnealingLR,
    'lr': 1e-4,
    'weight_decay': 1e-2,

    # 损失函数中，两个损失项的平衡参数
    'lamda': 0.5,
    
    'ViT_Encoder': {
        'PatchSize': (50, 50),
        'in_channels': 3, 
        'embedding_dim': 128,
        'num_layers': 6,
        'num_heads': 8,
        'dropout': 0.1
    },
    
    'MSCM': {
       'num_experts': 4, 
       'kernel_list': [3, 5]
    },

    'Decoder': {
        'input_size': 128,
        'hidden_size': 128,
        'num_heads': 8,
        'dropout': 0.1,
        'num_layers': 4,
    }
    
}

