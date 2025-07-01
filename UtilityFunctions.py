import torch
import numpy as np
from Hyperparameters import *
from PIL import Image
import matplotlib.pyplot as plt

# 本文件存储所有的辅助函数，包括可视化等函数

def Visualize_Sample(data, index):
    """
    可视化指定样本（未预处理）
    Args:
        data (List[tuple]):     tuple:[img(PIL.Image), latex(str)]
        index (int):            样本索引
    """
    sample = data[index]
    plt.imshow(sample[IMG_IDX])
    plt.title(sample[LATEX_IDX])
    plt.axis('off')
    plt.show()
 
def Visualize_ProcessedSample(data, index, mode='both'):
    """
    可视化预处理完毕的图像（经过反归一化）
    Args:
        data (List[tuple]):      tuple:[img(torch.tensor), mask(torch.tensor), latex(str)]
        index (int):             样本索引
        mode (str):              可视化图像和掩码的模式'both', 'img', 'mask'
    """
    sample = data[index]
    img_tensor = sample[IMG_IDX]
    mask_tensor = sample[MASK_IDX]
    
    # 转换为numpy.array
    img = img_tensor.clone().detach().cpu().numpy()
    mask = mask_tensor.clone().detach().cpu().numpy()
    # 反归一化
    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    img = img * std + mean  # 逆归一化
    # 调整通道顺序 [C, H, W] -> [H, W, C]
    img = np.transpose(img, (1, 2, 0))
    mask = np.transpose(mask, (1, 2, 0))
    # 裁剪到[0,1]范围（可能因增强超出范围）
    img = np.clip(img, 0, 1)
    # 显示图像
    if mode == 'both':
        plt.imshow(img)
        plt.title(sample[TOKEN_IDS_IDX])
        plt.axis('off')
        plt.show()
        plt.imshow(mask, alpha=0.3, cmap='jet')
        plt.title(sample[TOKEN_IDS_IDX])
        plt.axis('off')
        plt.show()
    
    elif mode == 'img':
        plt.imshow(img)
        plt.title(sample[TOKEN_IDS_IDX])
        plt.axis('off')
        plt.show()
    
    elif mode == 'mask':
        plt.imshow(mask, alpha=0.3, cmap='jet')
        plt.title(sample[TOKEN_IDS_IDX])
        plt.axis('off')
        plt.show()
    
    else:
        raise ValueError("mode must be 'both', 'img' or 'mask'!")

def Generating_CountingLabel(labels, type, tag, vocab):
    """
    将符号序列转为计数序列
    Args:
        labels (tensor):           符号序列[Batch, num_step]
        type (int):                符号类别总数
        tag (bool):                是否忽略<bos>, <eos>, <pad>, ^, _, {, }
        vocab (Vocab)              词表
    Return:
        counting_labels (tensor):  计数序列[Batch, type]
    """
    batch, num_step = labels.size()
    device = labels.device
    counting_labels = torch.zeros((batch, type))

    # 若 tag=True，忽略<bos>, <eos>, <pad>, ^, -, {, }
    if tag:
        ignore = [vocab['<bos>'], vocab['<eos>'], vocab['<pad>'], vocab['^'], vocab['_'], vocab['{'], vocab['}']]
    else:
        ignore = []

    for i in range(batch):
        for j in range(num_step):
            k = labels[i][j]

            if k in ignore:
                continue  # 这些符号不计入真实数量，其计数标签被强制设为0
            else:
                counting_labels[i][k] += 1  # 对于其他符号，计数标签加1

    return counting_labels.to(device)

def Visualize_Prediction(model, dataloader, max_display=5):
    """
    可视化模型的预测结果
    Args:
        model (TrDCN):            训练好的模型实例
        dataloader (DataLoader):  数据加载器
        vocab (Vocab):            词表对象
        max_display (int):        最多显示多少个样本
    """
    # 获取数据
    images, masks, texts = next(iter(dataloader))
    images = images.to(model.device)
    masks = masks.to(model.device)

    # 模型预测 - 移除显式的is_train参数
    with torch.no_grad():
        predicted_seq =  model.predict(images, masks) # 返回预测结果[B, max_len]
    
    vocab = model.vocab
    eos_idx = vocab['<eos>']
    
    # 可视化预测结果
    for i in range(min(max_display, images.shape[0])):
        # 获取原始图像和真实文本
        img = images[i].cpu().numpy()
        true_text = texts[i]

        # 反归一化图像
        mean = np.array(parameters['mean']).reshape(3, 1, 1)
        std = np.array(parameters['std']).reshape(3, 1, 1)
        img = img * std + mean
        img = np.clip(img, 0, 1)
        img = np.transpose(img, (1, 2, 0))

        # 处理预测序列 - 找到第一个<eos>的位置
        pred_seq = predicted_seq[i].cpu().numpy()
        eos_pos = np.where(pred_seq == eos_idx)[0]
        if len(eos_pos) > 0:
            pred_seq = pred_seq[:eos_pos[0]]  # 取<eos>之前的部分

        # 转换为token
        pred_tokens = vocab.to_tokens(pred_seq.tolist())
        pred_text = ' '.join(pred_tokens)
        
        # 显示结果
        print(f"Pred: {pred_text}")
        plt.imshow(img)
        plt.title(f"True: {true_text}")
        plt.axis('off')
        plt.show()
