# -*- coding: utf-8 -*-
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import bz2

class MSRADataset(Dataset):
    def __init__(self, file_path, word_to_idx, tag_to_idx, max_len=100):
        self.sentences, self.tags = self.load_data(file_path)
        self.word_to_idx = word_to_idx
        self.tag_to_idx = tag_to_idx
        self.max_len = max_len

    def load_data(self, file_path):
        sentences = []
        tags = []
        with open(file_path, 'r', encoding='utf-8') as f:
            sentence = []
            tag = []
            for line in f:
                line = line.strip()
                if not line:
                    if sentence:
                        sentences.append(sentence)
                        tags.append(tag)
                        sentence = []
                        tag = []
                    continue
                parts = line.split()
                if len(parts) == 2:
                    char, label = parts
                    sentence.append(char)
                    tag.append(label)
            if sentence:
                sentences.append(sentence)
                tags.append(tag)
        return sentences, tags

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        chars = self.sentences[idx]
        tags = self.tags[idx]
        if len(chars) > self.max_len:
            chars = chars[:self.max_len]
            tags = tags[:self.max_len]
        char_ids = [self.word_to_idx.get(c, self.word_to_idx['<UNK>']) for c in chars]
        tag_ids = [self.tag_to_idx[t] for t in tags]
        length = len(char_ids)
        char_ids = char_ids + [self.word_to_idx['<PAD>']] * (self.max_len - length)
        tag_ids = tag_ids + [self.tag_to_idx['O']] * (self.max_len - length)
        return torch.tensor(char_ids, dtype=torch.long), \
               torch.tensor(tag_ids, dtype=torch.long), \
               torch.tensor(length, dtype=torch.long)

def build_vocab(train_path, min_freq=1):
    """构建字符词汇表"""
    word_count = {}
    tag_count = {}
    with open(train_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) == 2:
                char, tag = parts
                word_count[char] = word_count.get(char, 0) + 1
                tag_count[tag] = tag_count.get(tag, 0) + 1

    word_to_idx = {'<PAD>': 0, '<UNK>': 1}
    for word, count in word_count.items():
        if count >= min_freq:
            word_to_idx[word] = len(word_to_idx)

    tag_to_idx = {'O': 0}
    for tag in tag_count:
        if tag != 'O':
            tag_to_idx[tag] = len(tag_to_idx)
    idx_to_tag = {v: k for k, v in tag_to_idx.items()}
    return word_to_idx, tag_to_idx, idx_to_tag

def load_pretrained_embedding(emb_file, word_to_idx, embedding_dim=300):
    """
    从压缩的文本文件中加载预训练字向量，构建嵌入矩阵。
    文件格式：第一行为 vocab_size embedding_dim，之后每行为 word vec1 vec2 ...
    """
    embeddings = np.random.uniform(-0.25, 0.25, (len(word_to_idx), embedding_dim))
    embeddings = embeddings.astype(np.float32)
    print(f"正在加载预训练字向量: {emb_file}")
    matched = 0
    with bz2.open(emb_file, 'rt', encoding='utf-8') as f:
        # 第一行：词汇量 维度
        header = f.readline().strip().split()
        vocab_size = int(header[0])
        dim = int(header[1])
        print(f"词向量文件包含 {vocab_size} 个词，维度 {dim}")
        if dim != embedding_dim:
            print(f"警告: 预训练向量维度 ({dim}) 与模型设置 ({embedding_dim}) 不一致，将截断或填充。")
        for line in f:
            parts = line.rstrip().split(' ')
            if len(parts) < 10:
                continue
            word = parts[0]
            if word in word_to_idx:
                vec = np.array(parts[1:1+embedding_dim], dtype=np.float32)
                embeddings[word_to_idx[word]] = vec
                matched += 1
    print(f"成功匹配 {matched} 个字符 (总词汇量 {len(word_to_idx)})")
    return torch.tensor(embeddings)

def create_dataloader(file_path, word_to_idx, tag_to_idx, batch_size=32, shuffle=True, max_len=100):
    dataset = MSRADataset(file_path, word_to_idx, tag_to_idx, max_len)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)

def collate_fn(batch):
    char_ids, tag_ids, lengths = zip(*batch)
    char_ids = torch.stack(char_ids)
    tag_ids = torch.stack(tag_ids)
    lengths = torch.stack(lengths)
    lengths, perm_idx = lengths.sort(descending=True)
    char_ids = char_ids[perm_idx]
    tag_ids = tag_ids[perm_idx]
    return char_ids, tag_ids, lengths