# -*- coding: utf-8 -*-
import torch
import json
import re
from data_loader import build_vocab
from model import BiLSTMCRF

def load_model(model_path, vocab_path, device):
    checkpoint = torch.load(vocab_path, map_location=device)
    word_to_idx = checkpoint['word_to_idx']
    tag_to_idx = checkpoint['tag_to_idx']
    idx_to_tag = checkpoint['idx_to_tag']

    model = BiLSTMCRF(len(word_to_idx), len(tag_to_idx))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model, word_to_idx, idx_to_tag

def text_to_sentences_with_position(text):
    """将文本按句号等切分，并记录每个句子在原文中的起始位置"""
    sentences = []
    pos = 0
    for para in text.split('\n'):
        para = para.strip()
        if not para:
            pos += len('\n')
            continue
        parts = re.split(r'([。！？!?；;])', para)
        i = 0
        while i < len(parts):
            if parts[i].strip():
                sent = parts[i]
                if i + 1 < len(parts):
                    sent += parts[i + 1]
                    i += 2
                else:
                    i += 1
                start_idx = text.find(sent, pos)
                if start_idx != -1:
                    sentences.append((sent, start_idx))
                    pos = start_idx + len(sent)
                else:
                    start_idx = pos
                    sentences.append((sent, start_idx))
                    pos += len(sent)
            else:
                i += 1
    return sentences

def predict_sentence(model, sentence, start_offset, word_to_idx, idx_to_tag, device, max_len=200):
    """预测单个句子中的实体，返回带全局位置的实体列表"""
    chars = list(sentence)
    if len(chars) > max_len:
        chars = chars[:max_len]
    char_ids = [word_to_idx.get(c, word_to_idx['<UNK>']) for c in chars]
    length = len(char_ids)
    char_ids = char_ids + [word_to_idx['<PAD>']] * (max_len - length)

    char_tensor = torch.tensor([char_ids], dtype=torch.long).to(device)
    lengths = torch.tensor([length], dtype=torch.long).to(device)

    with torch.no_grad():
        pred_tags = model(char_tensor, lengths)
    tags = pred_tags[0][:length]
    tags = [idx_to_tag[t] for t in tags]
    return parse_bio_with_pos(chars, tags, start_offset)

def parse_bio_with_pos(chars, tags, start_offset):
    """BIO 解析，记录实体在全文中的位置，返回 (name, type, start, end) 元组列表"""
    entities = []
    current_entity = []
    current_type = None
    current_start = -1
    for i, (char, tag) in enumerate(zip(chars, tags)):
        if tag == 'O':
            if current_entity:
                entities.append((''.join(current_entity), current_type,
                                 current_start, current_start + len(current_entity)))
                current_entity = []
                current_type = None
        elif tag.startswith('B-'):
            if current_entity:
                entities.append((''.join(current_entity), current_type,
                                 current_start, current_start + len(current_entity)))
            current_entity = [char]
            current_type = tag[2:]
            current_start = start_offset + i
        elif tag.startswith('I-'):
            if current_entity and tag[2:] == current_type:
                current_entity.append(char)
            else:
                if current_entity:
                    entities.append((''.join(current_entity), current_type,
                                     current_start, current_start + len(current_entity)))
                current_entity = [char]
                current_type = tag[2:]
                current_start = start_offset + i
    if current_entity:
        entities.append((''.join(current_entity), current_type,
                         current_start, current_start + len(current_entity)))
    return entities

# ==================== 基于最大匹配的规则抽取 (仅保留 DATE 和 WORK) ====================
def extract_date_entities(text):
    candidates = []
    patterns = [
        (r'(\d{4}\s*年\s*[—－-]\s*\d{4}\s*年)', 'DATE'),
        (r'(\d{4}\s*年\s*\d{1,2}\s*月\s*\d{1,2}\s*日)', 'DATE'),
        (r'(\d{4}\s*年\s*\d{1,2}\s*月)', 'DATE'),
        (r'(\d{4}\s*年)', 'DATE'),
        (r'(\d{4}-\d{1,2}-\d{1,2})', 'DATE'),
        (r'(\d{4}-\d{1,2})', 'DATE'),
        (r'(\d{1,2}\s*月\s*\d{1,2}\s*日)', 'DATE'),
        (r'(\d{2}\s*世纪\s*\d{2}\s*年代)', 'DATE'),
        (r'(\d{2}\s*年代)', 'DATE'),
        (r'(\d{1,2}\s*时\s*\d{1,2}\s*分)', 'DATE'),
    ]
    for pat, label in patterns:
        for m in re.finditer(pat, text):
            candidates.append((m.group(1).strip(), label, m.start(1), m.end(1)))
    candidates.sort(key=lambda x: (x[2], x[3]))
    # 最大匹配过滤：保留最长的非重叠实体
    filtered = []
    last_end = -1
    for ent in candidates:
        name, etype, start, end = ent
        if start < last_end:
            prev = filtered[-1]
            if end - start > prev[3] - prev[2]:
                filtered[-1] = (name, etype, start, end)
                last_end = end
        else:
            filtered.append((name, etype, start, end))
            last_end = end
    return filtered

def extract_work_entities(text):
    """抽取《》内的作品名称，返回 (name, type, start, end) 列表"""
    candidates = []
    pattern = r'《(.*?)》'
    for m in re.finditer(pattern, text):
        work_name = m.group(1).strip()
        if len(work_name) >= 2 and not all(c in '，。；、：“”’‘！？' for c in work_name):
            candidates.append((work_name, 'WORK', m.start(1), m.end(1)))
    return candidates

def rule_based_extraction(text):
    """返回所有规则抽取的实体列表"""
    date_ents = extract_date_entities(text)
    work_ents = extract_work_entities(text)
    return date_ents + work_ents

# ==================== 合并逻辑 ====================
def merge_all_entities(model_entities, rule_entities):
    """
    合并模型预测和规则抽取的实体，按位置排序，处理重叠（最大匹配原则）。
    输入每个实体为 (name, type, start, end) 元组。
    输出合并后的字典列表，每个字典包含 entity, type, start, end。
    """
    all_ents = model_entities + rule_entities
    all_ents.sort(key=lambda x: (x[2], x[3]))
    merged = []
    for ent in all_ents:
        name, etype, start, end = ent
        if end <= start:
            continue
        if not merged:
            merged.append(ent)
            continue
        prev = merged[-1]
        prev_name, prev_type, prev_start, prev_end = prev
        # 完全包含的重叠：新实体在旧实体内，保留更长者
        if start >= prev_start and end <= prev_end:
            if len(name) > len(prev_name):
                merged[-1] = ent
        elif start < prev_end and end > prev_end:
            # 部分重叠，保留更长的实体
            if (end - start) > (prev_end - prev_start):
                merged[-1] = ent
        else:
            merged.append(ent)
    # 转换为最终列表，包含位置信息
    final = []
    for name, etype, start, end in merged:
        final.append({'entity': name, 'type': etype, 'start': start, 'end': end})
    return final

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, word_to_idx, idx_to_tag = load_model(
        model_path='models/bilstm_crf_msra_computer.pth',
        vocab_path='models/vocab.pth',
        device=device
    )

    with open('data/raw/Turing_Baidu.txt', 'r', encoding='utf-8') as f:
        text = f.read()

    # 1. 模型预测
    sentences_with_pos = text_to_sentences_with_position(text)
    model_entities = []
    for sent, start_offset in sentences_with_pos:
        if not sent.strip():
            continue
        ents = predict_sentence(model, sent, start_offset, word_to_idx, idx_to_tag, device)
        model_entities.extend(ents)

    # 2. 规则抽取
    rule_entities = rule_based_extraction(text)

    # 3. 合并（最终实体包含 start, end）
    final_entities = merge_all_entities(model_entities, rule_entities)

    print(f"模型预测实体数: {len(model_entities)}, 规则匹配实体数: {len(rule_entities)}, 合并后总数: {len(final_entities)}")
    print("最终实体列表（按原文顺序，带位置）:")
    for item in final_entities:
        print(f"{item['entity']}\t{item['type']}\t位置：[{item['start']}, {item['end']})")

    # 保存 JSON，包含 start, end 字段
    with open('data/output/turing_entities_with_rules_se.json', 'w', encoding='utf-8') as f:
        json.dump(final_entities, f, ensure_ascii=False, indent=2)

if __name__ == '__main__':
    main()