import torch
import torch.nn as nn
import torch.nn.functional as F
from Encoder import ViT
from DynamicCounting import CountingDecoder
from Decoder import TransformerDecoder
from UtilityFunctions import Generating_CountingLabel
from Hyperparameters import parameters

# 本文件定义整个模型
# TrDCN: "Transformer with Dynamic Counting Network for Mathematical Expression Recognition"

class TrDCN(nn.Module):
    def __init__(self, vocab):
        super(TrDCN, self).__init__()
        self.vocab = vocab
        self.device = parameters['device']
        
        # 1.Vision Transformer提取图像信息
        self.encoder = ViT()
 
        # 2.动态卷积计数网络
        self.in_channels = parameters['ViT_Encoder']['embedding_dim']
        self.out_channels = len(vocab)
        self.CountingNet = CountingDecoder(self.in_channels, self.out_channels, kernel_list=parameters['MSCM']['kernel_list'])

        # 3.Transformer解码器
        self.decoder =  TransformerDecoder(vocab)

        # 4. 损失函数
        self.cross = nn.CrossEntropyLoss(ignore_index=vocab['<pad>'])
        self.counting_loss = nn.SmoothL1Loss(reduction='mean')
 
        self.to(self.device)
        
        # 5. 初始化权重
        def init_weights(m):
            if isinstance(m, nn.Linear | nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
        
        self.apply(init_weights)
       
        # 6. 可视化模型参数数量
        self.encoder_params_nums  = 0
        self.counting_params_nums = 0
        self.decoder_params_nums  = 0
        
        for _, param in self.encoder.named_parameters():
            self.encoder_params_nums += param.numel()
        for _, param in self.CountingNet.named_parameters():
            self.counting_params_nums += param.numel()
        for  _, param in self.decoder.named_parameters():
            self.decoder_params_nums += param.numel()

        print(f"Encoder params: {self.encoder_params_nums / (1000 ** 2)} M")
        print(f"Counting params: {self.counting_params_nums / (1000 ** 2)} M")
        print(f"Decoder params: {self.decoder_params_nums / (1000 ** 2)} M")
        print(f"Total params: {(self.encoder_params_nums + self.counting_params_nums + self.decoder_params_nums) / (1000 ** 2)} M")

    def forward(self, images, masks, labels, lengths, is_train=True):
        """
        Args:
            images (torch.tensor):            图像特征                  -> [B, 3, 100, 500]
            masks (torch.tensor):             掩码特征                  -> [B, 1, 100, 500]
            labels (torch.tensor):            词元索引                  -> [B, max_len]
            lengths (torch.tensor):           实际长度                  -> [B]
            is_train (bool):                  是否为训练模式
        Returns:
            word_probs (torch.tensor):        词元概率                   -> [B, max_len, vocab_size]
            count_vector (torch.tensor):      符号统计向量               -> [B, out_channels]
            word_loss (torch.tensor):         词元损失                   -> [1]
            counting_loss (torch.tensor):     符号计数损失               -> [1]
            attention_weights (torch.tensor | None): 注意力权重          -> [B, num_heads, max_len, max_len] | None
        """
        images = images.to(self.device)
        masks = masks.to(self.device)
        labels = labels.to(self.device)
        lengths = lengths.to(self.device)
        
        #——————1. ViT特征提取————————————————————————————————————————————————————————————
        image_features = self.encoder(images)    # 图像特征 [B, 128, 10, 10]
        mask_features = self.encoder(masks)    # 掩码特征 [B, 128, 10, 10]

        #——————2. 计数标签生成———————————————————————————————————————————————————————————
        counting_labels = Generating_CountingLabel(labels, self.out_channels, tag=True, vocab=self.vocab)
        
        #———————3.多分支计数预测——————————————————————————————————————————————————————————
        count_vector, _ = self.CountingNet(image_features, mask_features)
        count_vector = count_vector / len(parameters['MSCM']['kernel_list'])

        #————————4.Transformer解码器预测—————————————————————————————————————————————————
        word_probs, attention_weights = self.decoder(image_features, mask_features, labels, counting_labels, lengths, is_train)

        #————————5.损失计算——————————————————————————————————————————————————————————————
        counting_loss = self.counting_loss(count_vector, counting_labels)
        word_loss = self.cross(word_probs.reshape(-1, word_probs.shape[-1]), labels.reshape(-1))
        
        # 返回预测结果和损失
        return word_probs, count_vector, word_loss, counting_loss, attention_weights
    
    def predict(self, images, masks):
        """
        Args:
            images (torch.tensor):            图像特征        -> [B, 3, 100, 500]
            masks (torch.tensor):             掩码特征        -> [B, 1, 100, 500]
        Returns:
            predicted_seq (torch.tensor):     预测结果        -> [B, max_len]
        """
        images = images.to(self.device)
        masks = masks.to(self.device)
        
        #——————1. ViT特征提取————————————————————————————————————————
        image_features = self.encoder(images * masks)
        mask_features = self.encoder(masks)

        #——————2.多分支计数预测——————————————————————————————————————
        count_vector, _ = self.CountingNet(image_features, mask_features)
        count_vector = count_vector / len(parameters['MSCM']['kernel_list'])

        #——————3.Transformer解码器预测———————————————————————————————
        predicted_seq = self.decoder.predict(image_features, mask_features, counting_preds=count_vector)

        # 返回预测结果[B, max_len]
        return predicted_seq
