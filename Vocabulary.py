import re
import collections

# 本文件存储词表类和分词函数

def Tokenize_Word_Level(latex_str):
    # e.g:
    # \frac { 5 } { 1 4 } = \frac { 1 } { x }
    # ['\\frac', '{', '5', '}', '{', '1', '4', '}', '=', '\\frac', '{', '1', '}', '{', 'x', '}']
    """
    单词级分词
    Args:
        latex_str (str):    LaTeX 文本
    Returns:
        text (List[str]):   分词后的 token 列表
    """
    # pattern 是一个正则表达式，用于匹配 latex_str 中的所有字符
    pattern = r'(\\?[a-zA-Z]+|\{|\}|[^ \\{}]+)'
    # 返回 latex_str 中所有匹配 pattern 的字符串，并去除空字符串
    return [t for t in re.findall(pattern, latex_str) if t.strip()]

class Vocab:
    # 词表类，统计数据集的词元，并将词元转换为索引，同时支持反向查询
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=['<bos>', '<eos>', '<pad>']):
        # <unk> -> 未知词元 0 
        # <bos> -> 句子开始词元 1
        # <eos> -> 句子结束词元 2
        # <pad> -> 填充词元 3
        if tokens is None:
            tokens = []
        
        # 统计词元频率
        # 同时按出现频率排序
        counter =  self.count_corpus(tokens)
        self._token_freqs = sorted(counter.items(), key=lambda x : x[1], reverse=True)
        
        # idx_to_token: List[idx]
        # token_to_idx: {token: idx}
        self.idx_to_token = ['unk'] + reserved_tokens
        self.token_to_idx = {token : idx for idx, token in enumerate(self.idx_to_token)}
        
        # 根据出现频率从小到大，将词元添加到词表中
        # 如果词元的频率小于某一个阈值，则停止添加
        for token, freq in self._token_freqs:
            if freq < min_freq:
                break
            self.idx_to_token.append(token)
            self.token_to_idx[token] = len(self.idx_to_token) - 1

    def count_corpus(self, tokens):
        """
        计算语料库中每个 token 的频率
        Args:
            tokens (List[str]):        词元列表
        Returns:
            Counter ({token: count}):  词元-频率字典
        """
        if len(tokens) == 0 or isinstance(tokens[0], list):
            tokens = [token for line in tokens for token in line]
        return collections.Counter(tokens)
    
    def __getitem__(self, tokens):
        """
        将 tokens 转换为对应的索引
        Args:
            tokens (str 或 List[str]):      词元
        Returns:
            indices (int 或 List[int]):     对应的索引
        """
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        """
        将 indices 转换为对应的 token
        Args:
            indices (List[idx]):   索引列表
        Returns:
            tokens (List[str]):    对应的 token
        """
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]
    
    def __len__(self):
        return len(self.idx_to_token)
    
    def unk(self): 
        # 未知词元的索引为0
        return 0

    def token_freqs(self):
        return self._token_freqs
        
def Truncate_Padding(line, nums_step, padding_token_idx):
    """
    文本截断或补齐序列
    Args:
        line (List[idx]):         原始序列
        nums_step (int):          截断或补齐后的长度
        padding_token_idx (int):  填充词元的索引
    Returns:
        line (List[idx]):         截断或补齐后的序列
    """
    if len(line) > nums_step:
        return line[:nums_step]
    else:
        return line + [padding_token_idx] * (nums_step - len(line))
    
    