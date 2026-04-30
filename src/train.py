# -*- coding: utf-8 -*-
import os
import torch
import torch.optim as optim
from tqdm import tqdm
from data_loader import build_vocab, create_dataloader, load_pretrained_embedding
from model import BiLSTMCRF

def train():
    BATCH_SIZE = 64
    EMBEDDING_DIM = 300
    HIDDEN_DIM = 256
    NUM_LAYERS = 2
    DROPOUT = 0.5
    #LEARNING_RATE = 0.001
    BILSTM_LR = 0.001  # BiLSTM 层的学习率
    CRF_LR = 0.01  # CRF 层的学习率
    EPOCHS = 100
    MAX_LEN = 100
    MODEL_SAVE_PATH = 'models/bilstm_crf_computer.pth'
    EMBEDDING_PATH = 'data/embeddings/sgns.baidubaike.bigram-char.bz2'

    RESUME = False                      # 改为 False 为全新训练
    RESUME_MODEL_PATH = 'models/bilstm_crf_msra.pth'  # 要加载的权重文件
    VOCAB_PATH = 'models/vocab.pth'     # 保存过的词汇表

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 构建/加载词表
    if RESUME and os.path.exists(VOCAB_PATH):
        print("从已有词汇表恢复...")
        vocab = torch.load(VOCAB_PATH, map_location='cpu')
        word_to_idx = vocab['word_to_idx']
        tag_to_idx = vocab['tag_to_idx']
        idx_to_tag = vocab['idx_to_tag']
    else:
        print("构建新词汇表...")
        word_to_idx, tag_to_idx, idx_to_tag = build_vocab('data/data/train.txt', min_freq=2)
        torch.save({
            'word_to_idx': word_to_idx,
            'tag_to_idx': tag_to_idx,
            'idx_to_tag': idx_to_tag
        }, VOCAB_PATH)

    vocab_size = len(word_to_idx)
    tagset_size = len(tag_to_idx)
    print(f"字符表大小: {vocab_size}, 标签表大小: {tagset_size}")

    # 预训练字向量
    pretrained_embeddings = None
    if os.path.exists(EMBEDDING_PATH):
        pretrained_embeddings = load_pretrained_embedding(EMBEDDING_PATH, word_to_idx, EMBEDDING_DIM)
    else:
        print(f"警告: 未找到预训练向量文件 {EMBEDDING_PATH}，将使用随机初始化。")

    # 数据加载器
    train_loader = create_dataloader('data/data/train.txt', word_to_idx, tag_to_idx,
                                     batch_size=BATCH_SIZE, shuffle=True, max_len=MAX_LEN)
    test_loader = create_dataloader('data/data/test.txt', word_to_idx, tag_to_idx,
                                    batch_size=BATCH_SIZE, shuffle=False, max_len=MAX_LEN)

    # 模型初始化
    model = BiLSTMCRF(vocab_size, tagset_size, pretrained_embeddings,
                      embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM,
                      num_layers=NUM_LAYERS, dropout=DROPOUT)
    model.to(device)

    # 优化器
    #optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    crf_params = list(map(id, model.crf.parameters()))  # 获取 CRF 层参数的 id
    other_params = filter(lambda p: id(p) not in crf_params, model.parameters())

    optimizer = optim.Adam([
        {'params': other_params, 'lr': BILSTM_LR},  # BiLSTM 等层使用基础学习率
        {'params': model.crf.parameters(), 'lr': CRF_LR}  # CRF 层使用更大的学习率
    ], weight_decay=0.0)

    start_epoch = 1
    best_test_loss = float('inf')

    # 恢复模型权重
    if RESUME and os.path.exists(RESUME_MODEL_PATH):
        print(f"加载模型权重: {RESUME_MODEL_PATH}")
        checkpoint = torch.load(RESUME_MODEL_PATH, map_location=device)
        model.load_state_dict(checkpoint)
        # 如果要恢复优化器状态及当前 epoch，需要之前保存成字典
        # 这里仅演示简单权重恢复，优化器重新开始
        print("模型权重已加载，从第1个epoch继续训练（优化器全新）。")
    else:
        print("未找到模型权重，从头开始训练。")

    # 训练循环
    for epoch in range(start_epoch, EPOCHS + 1):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{EPOCHS}')
        for batch in pbar:
            char_ids, tag_ids, lengths = batch
            char_ids = char_ids.to(device)
            tag_ids = tag_ids.to(device)
            lengths = lengths.to(device)

            optimizer.zero_grad()
            loss = model(char_ids, lengths, tag_ids)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})

        avg_train_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch} 平均训练损失: {avg_train_loss:.4f}")

        model.eval()
        total_test_loss = 0
        with torch.no_grad():
            for batch in test_loader:
                char_ids, tag_ids, lengths = batch
                char_ids = char_ids.to(device)
                tag_ids = tag_ids.to(device)
                lengths = lengths.to(device)
                loss = model(char_ids, lengths, tag_ids)
                total_test_loss += loss.item()
        avg_test_loss = total_test_loss / len(test_loader)
        print(f"Epoch {epoch} 测试集损失: {avg_test_loss:.4f}")

        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"保存最佳模型到 {MODEL_SAVE_PATH}")

    print("训练完成！")

if __name__ == '__main__':
    train()