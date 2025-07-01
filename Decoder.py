import torch 
from torch import nn
import math
from Hyperparameters import *

# 本文件定义模型解码器：Transformer Decoder，将编码器输出以及动态卷积 MSCM 的角色统计输出作为解码器输入 

class PositionalEncoding(nn.Module):
    # 仍使用Transformer的正弦余弦位置编码
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
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1)]
        return self.dropout(x)
    
class TransformerDecoder(nn.Module):
    def __init__(self, vocab):
        """
        Args:
            vocab (Vocab):  词表, 因为需要进行词嵌入操作
        """
        super().__init__()
        self.input_size = parameters['Decoder']['input_size']
        self.hidden_size = parameters['Decoder']['hidden_size']
        self.vocabsize = len(vocab)
        self.device = parameters['device']
        self.bos_idx = 1  # 开始<bos>索引为1
        self.eos_idx = 2  # 结束<eos>索引为2
        self.pad_idx = 3  # 填充<pad>索引为3

        # 词嵌入层
        self.embedding = nn.Embedding(self.vocabsize ,self.input_size)
        self.pos_encoder = PositionalEncoding(self.input_size, parameters['Decoder']['dropout'])

        # Transformer 解码器
        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=self.hidden_size,
                                       nhead=parameters['Decoder']['num_heads'],
                                       dim_feedforward=self.hidden_size * 4,
                                       dropout=parameters['Decoder']['dropout']),
                                       num_layers=parameters['Decoder']['num_layers'])

        # 特征投影层（将 CNN 特征适配到 Transformer）
        self.feature_proj = nn.Linear(parameters['ViT_Encoder']['embedding_dim'], self.hidden_size)

        # 计数模块融合层（保留）
        self.counting_context = nn.Linear(len(vocab), self.hidden_size)

        # 输出层
        self.word_convert = nn.Linear(self.hidden_size, self.vocabsize)

        # 统一移动所有模块到指定设备
        self.to(self.device)

    def forward(self, image_features, mask_features, labels, counting_preds, seq_lens=None, is_train=True):
        """
        前向传播
        Args:
            image_features (tensor):     图像特征信息       -> [Batch, Channel, Height, Width]
            mask_features (tensor):      掩码特征信息       -> [Batch, Channel, Height, Width]
            labels (tensor):             实际的标签         -> [Batch, Num_steps]
            counting_preds (tensor):     MSCM 符号计数输出  -> [Batch, Channel]
            seq_lens (tensor):           序列长度           -> [Batch]
            is_train (bool):             是否是训练阶段
        Return:
            word_probs (tensor):         词元的概率         -> [Batch, Num_steps, Word_num]
            None                         (暂时不返回注意力权重)
        """
        batch_size, num_steps = labels.shape
        image_features = image_features.to(self.device)
        mask_features = mask_features.to(self.device)
        labels = labels.to(self.device)
        counting_preds = counting_preds.to(self.device)

        # ——————————1. 生成填充掩码 (padding mask)—————————————————————————————————
        if seq_lens is not None:
            # 基于实际长度生成掩码
            seq_lens = seq_lens.to(self.device)
            pad_mask = (torch.arange(num_steps, device=self.device).expand(batch_size, num_steps) >= seq_lens.unsqueeze(1))  # [B, seq_len]
        else:
            # 如果没有提供长度，则自动检测填充token<pad> = 3
            pad_mask = (labels == self.pad_idx)  # [B, seq_len]

        # ——————————2. 词嵌入 + 位置编码———————————————————————————————————————————
        if is_train:
            # 如果是训练，则使用真实的词元索引，同时在最前面加上开始词元<bos>
            bos = torch.ones(batch_size, 1, dtype=torch.long, device=self.device)
            input = torch.cat([bos, labels[:, :-1]], dim=1)  # [B, seq_len]
        else:
            # 否则将输入开始词元<bos>的索引作为自回归的开始
            input = torch.ones(batch_size, num_steps, dtype=torch.long, device=self.device)

        word_embeds = self.embedding(input)
        word_embeds = self.pos_encoder(word_embeds.transpose(0, 1)).transpose(0, 1)  # [seq_len, B, input_size] -> [B, seq_len, input_size]

        # ——————————3. 编码图像特征（展平空间维度并投影）——————————————————————————
        image_features = image_features * mask_features

        image_features = image_features.flatten(2).transpose(1, 2)  # [B, H*W, C]
        memory = self.feature_proj(image_features)  # [B, H*W, hidden_size]

        # ——————————4. 生成自回归掩码（测试时阻止偷看未来信息）———————————
        tgt_mask = self.generate_square_subsequent_mask(num_steps).to(self.device)

        # ——————————5. Transformer 解码———————————————————————————————————————————
        hidden_states = self.transformer_decoder(
            tgt=word_embeds.transpose(0, 1),    # [seq_len, B, hidden_size]
            memory=memory.transpose(0, 1),      # [H*W, B, hidden_size]
            tgt_mask=tgt_mask,                  # 自回归掩码 [seq_len, seq_len]
            tgt_key_padding_mask=pad_mask       # 填充掩码 [B, seq_len]
        ).transpose(0, 1)  # [B, seq_len, hidden_size]

        # ——————————6. 融合计数信息———————————————————————————————————————————————
        counting_context = self.counting_context(counting_preds).unsqueeze(1) # [B, 1, hidden_size]
        output = hidden_states + counting_context

        # ——————————7. 输出logits（无Softmax）———————————————————————————————————
        word_probs = self.word_convert(output)  # [B, seq_len, vocabsize]

        # 训练时直接使用 nn.CrossEntropyLoss（内置 Softmax）
        return word_probs, None # 暂时不返回注意力权重
    
    def predict(self, image_features, mask_features, counting_preds):
        """
        预测阶段专用前向传播
        Args:
            image_features (tensor):     图像特征信息         -> [Batch, Channel, Height, Width]
            mask_features (tensor):      掩码特征信息         -> [Batch, Channel, Height, Width]
            counting_preds (tensor):     MSCM 符号计数输出    -> [Batch, Channel]
        Return:
            predicted_words (tensor):    预测词元            -> [Batch, max_len]
        """
        image_features = image_features.to(self.device)
        counting_preds = counting_preds.to(self.device)
        mask_features = mask_features.to(self.device)
    
        batch_size = image_features.size(0)
        outputs = []
        unfinished = torch.ones(batch_size, dtype=torch.bool, device=self.device)  # 标记未完成的样本
    
        # 初始输入：<bos>（假设索引为1）
        # current_input = torch.ones(batch_size, 1, dtype=torch.long, device=self.device)  # [B, 1]
        current_input = torch.full((batch_size, 1), self.bos_idx, dtype=torch.long, device=self.device)
    
        # --- 预计算图像特征和计数信息（避免每次循环重复计算）---
        # 1. 编码图像特征
        image_features = image_features * mask_features
        memory = self.feature_proj(image_features.flatten(2).transpose(1, 2)).transpose(0, 1)  # [H*W, B, hidden_size]

        # 2. 预计算计数上下文
        counting_context = self.counting_context(counting_preds).unsqueeze(1)  # [B, 1, hidden_size]

        for _ in range(parameters['max_len']):
            #————————————————1. 词嵌入——————————————————————————————————————————————————————
            word_embeds = self.pos_encoder(self.embedding(current_input).transpose(0, 1)).transpose(0, 1)  # [B, seq_len, input_size]
 
            #————————————————2. 生成自回归掩码———————————————————————————————————————————————
            tgt_mask = self.generate_square_subsequent_mask(current_input.size(1)).to(self.device)
        
            #————————————————3. Transformer 解码—————————————————————————————————————————————
            output = self.transformer_decoder(
                word_embeds.transpose(0, 1),  # [seq_len, B, input_size]
                memory=memory,                # 预计算的memory [H*W, B, hidden_size]
                tgt_mask=tgt_mask
            )[-1:]  # 只取最后一步输出 [1, B, hidden_size]

            #————————————————4. 融合计数信息——————————————————————————————————————————————————
            output = output.transpose(0, 1) + counting_context  # [B, 1, hidden_size]

            #————————————————5. 输出logits（无Softmax）———————————————————————————————————————
            logits = self.word_convert(output.squeeze(1)) # [B, vocabsize]
            next_word = logits.argmax(-1)                 # [B]
            
            #————————————————6. 记录输出（仅未完成的样本）——————————————————————————————————————
            outputs.append(next_word * unfinished)

            #————————————————7. 更新未完成标记—————————————————————————————————————————————————
            unfinished &= (next_word != self.eos_idx)
        
            #—————————————————8. 提前终止检查——————————————————————————————————————————————————
            if not unfinished.any():
                break

            #—————————————————9. 更新输入（仅未完成的样本）—————————————————————————————————————
            current_input = torch.cat([current_input[unfinished], next_word[unfinished].unsqueeze(1)], dim=1)

            #—————————————————10. 更新memory/counting_context（仅保留未完成的）—————————————————
            memory = memory[:, unfinished, :]
            counting_context = counting_context[unfinished]

        max_len = len(outputs)
        outputs = torch.stack(outputs, dim=1)
        eos_pos = (outputs == self.eos_idx).float().argmax(dim=1)
        eos_pos[eos_pos == 0] = max_len
        final_outputs = []
        for i in range(batch_size):
            final_outputs.append(outputs[i, :eos_pos[i]+1])
        return torch.stack(final_outputs, dim=0)
    
    def generate_square_subsequent_mask(self, seq_len):
        """生成自回归掩码（下三角矩阵）"""
        return torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()

    def init_hidden(self, batch_size):
        """兼容性方法（实际不需要，可以保持使用 RNN 时接口一致）"""
        return torch.zeros(batch_size, self.hidden_size).to(self.device)
    
# 如果需要可视化注意力权重
# 修改 TransformerDecoderLayer 的 forward
class CustomTransformerDecoderLayer(nn.TransformerDecoderLayer):
    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        attn_output, attn_weights = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,need_weights=True)
        return attn_output, attn_weights
