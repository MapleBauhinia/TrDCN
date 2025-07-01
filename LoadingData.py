import os
import json
import re
from Vocabulary import *
from Hyperparameters import *
from PIL import Image
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

# 本文件存储数据集加载函数

def Loading_HME100K(data_root, mode='train'): 
    """
    加载图片和对应LaTeX标签
    Args:
        data_root (str):       数据集根目录
        mode (str):            模式'train' 或 'test'
    Returns:
        data (List[tuple]):    tuple:[img(PIL.Image), latex(str)]
    """
    # 检查mode规格
    if mode not in ['train', 'test']:
        raise ValueError("mode must be 'train' or 'test'!")
    
    # 构建文件路径
    img_dir = os.path.join(data_root, mode, f"{mode}_images")
    lable_dir = os.path.join(data_root, mode, f"{mode}_labels.txt")
    
    # 读取标签文本
    with open(lable_dir, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 解析每一行的数据
    data = []
    for i, line in enumerate(lines):
        # 若指定最大样本数，则超过最大样本数时跳出循环
        if mode == 'train' and i >= parameters['Training_max_number']:
            break
        elif mode == 'test' and i >= parameters['Testing_max_number']:
            break

        filename, latex = line.strip().split('\t', 1)
        img_path = os.path.join(img_dir, filename)

        try:
            img = Image.open(img_path).convert('RGB')
            data.append((img, latex))  # 使用元组而非列表

        except FileNotFoundError:
            print(f"Error: {img_path} not found!!!")

    return data

def Loading_Subset(data_root, difficulty='easy'):
    """
    加载难度分级子集
    Args:
        data_root (str):        数据集根目录
        difficulty (str):       难度系数'easy'/'medium'/'hard'
    Returns:
        data (List[int]):       对应样本的名称列表(test_idx)
    """
    subset_path = os.path.join(data_root, 'subset', f"{difficulty}.json")
    with open(subset_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def Preprocess_Data(data, mode='train'):
    """
    数据预处理
    Args:
        data (List[tuple]):     [img(PIL.Image), latex(str)]
        mode (str):             模式'train' 或 'test'
    Returns:
        data (List[tuple]):     tuple:[img(torch.tensor), mask(torch.tensor), latex(str)]
    """
    # 图片预处理
    transform = transforms.Compose([
        transforms.Resize(parameters['ImageSize']),
        transforms.ToTensor(),
        transforms.Normalize(mean=parameters['mean'], std=parameters['std'])])
    
    def generate_soft_mask(img: Image):
        """
        通过图像生成对应的软掩码
        Args:
            img (PIL.Image):       RGB图像
        Returns:
            mask (torch.Tensor):   归一化的单通道掩码 (公式区域≈1, 背景≈0)
        """
        img_gray = img.resize(parameters['MaskSize'])
        img_gray = img_gray.convert('L')
        mask = transforms.ToTensor()(img_gray).float()
        mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-6) 
        mask = 1.0 - mask 
        return mask
    
    # LaTex文本的预处理
    def clean_latex(text):
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\\[a-z]+\{([^}]+)\}', r'\1', text)
        return text.strip()
    
    # 预处理并整合所有样本
    processed_data = []

    if mode == 'train':
        # 训练模式，随机翻转图像，数据增强
        for img, latex in data:
            try:
                # 生成掩码
                mask = generate_soft_mask(img)
                
                # 随机决定是否翻转
                if torch.rand(1) < parameters['flip_prob']:
                    img = img.transpose(Image.FLIP_LEFT_RIGHT)
                    mask = mask.flip(2) 
                
                # 应用图像变换
                img_tensor = transform(img)
                processed_data.append((img_tensor, mask, clean_latex(latex)))
                
            except Exception as e:
                print(f"Error: {e}")
    else:
        # 测试模式，不进行翻转
        for img, latex in data:
            try:
                img_tensor = transform(img)
                mask = generate_soft_mask(img)
                processed_data.append((img_tensor, mask, clean_latex(latex)))
                
            except Exception as e:
                print(f"Error: {e}")

    return processed_data

class HME100KDataset(torch.utils.data.Dataset):
    # 数据集类
    # 将应用于torch.utils.data.DataLoader
    def __init__(self, data, vocab=None):
        """
        Args:
            data (List[tuple]):      tuple:[img(torch.tensor), mask(torch.tensor), latex(str)] -> 已经完成预处理
            vocab (Vocab):           数据集产生的词表
        """
        self.data = data
        self.vocab = vocab
        self.max_len = parameters['max_len']

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        返回指定样本
        Args:
            idx (int):                样本索引
        Returns:
            img (torch.tensor):       图像
            mask (torch.tensor):      掩码
            token_ids (torch.tensor): 词元索引
            length (torch.tensor):    实际长度
            或
            img (torch.tensor):       图像
            mask (torch.tensor):      掩码
            latex (str):              原始LaTeX文本
        """
        img, mask, latex = self.data[idx]

        # 如果是训练数据集，则已经构建了词表
        if self.vocab:
            # 分词后，加上结束词元<eos>，将LaTeX文本转换为词元索引
            tokens = Tokenize_Word_Level(latex) + ['<eos>']
            token_ids = self.vocab[tokens]
            length = len(token_ids)
            token_ids = Truncate_Padding(line=token_ids, nums_step=self.max_len, padding_token_idx=self.vocab['<pad>'])
            return (img, mask, torch.LongTensor(token_ids), torch.LongTensor([length]))
        else:
            return (img, mask, latex)
        
def collate_fn(batch):
    """
    本函数将应用于torch.utils.data.DataLoader的collate_fn参数
    将batch中的样本组合成一个batch
    Args:
        batch (List[tuple]):          tuple:[img(torch.tensor), mask(torch.tensor), token_ids(torch.tensor), length(torch.tensor)] 或
                                            [img(torch.tensor), mask(torch.tensor), latex(str)]
    Returns:
        images (torch.tensor):        图像
        masks (torch.tensor):         掩码
        token_ids (torch.tensor):     词元索引
        lengths (torch.tensor):       实际长度
        或
        images (torch.tensor):        图像
        masks (torch.tensor):         掩码
        texts (List[str]):            原始LaTeX文本
    """
    if len(batch[0]) == 4:  
        # 训练模式
        images = torch.stack([sample[IMG_IDX] for sample in batch])
        masks = torch.stack([sample[MASK_IDX] for sample in batch])
        token_ids = torch.stack([sample[TOKEN_IDS_IDX] for sample in batch])
        lengths = torch.cat([sample[LENGTH_IDX] for sample in batch])
        return (images, masks, token_ids, lengths)
    else:
        # 测试模式
        images = torch.stack([sample[IMG_IDX] for sample in batch])
        masks = torch.stack([sample[MASK_IDX] for sample in batch])
        texts = [sample[LATEX_IDX] for sample in batch]
        return (images, masks, texts)
    
def Loading_HME100K_Dataset(data_root, mode='train'):
    """
    加载数据集至批量迭代器，同时预处理数据集，并构建词表
    Args:
        data_root (str):              数据集根目录
        mode (str):                   模式'train' 或 'test'
    Returns:
        vocab (Vocab):                数据集产生的词表
        data_loader (DataLoader):     批量迭代器 (batch, (images, masks, token_ids, lengths)) 或 (batch, (images, masks, texts))
    """
    # 加载数据集，同时预处理数据集
    data = Loading_HME100K(data_root=data_root, mode=mode)
    data = Preprocess_Data(data=data, mode=mode)
    
    # 分词，同时构建词表(训练样本时)
    tokenized_samples = [Tokenize_Word_Level(sample[LATEX_IDX]) for sample in data]
    vocab = Vocab(tokenized_samples) if mode == 'train' else None
    
    # 加载批量迭代器
    dataset = HME100KDataset(data=data, vocab=vocab)
    return vocab, DataLoader(dataset=dataset, batch_size=parameters['BatchSize'], 
                             shuffle=True if mode == 'train' else False, collate_fn=collate_fn)
