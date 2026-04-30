# finetune_custom.py
import json, random, torch, os
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from sklearn.metrics import classification_report
from tqdm import tqdm

# ---------- 配置 ----------
BERT_PATH = './bert-base-chinese'
PRETRAINED_MODEL_PATH = 'models/relation_bert_duie_fixed.pth'  # 预训练权重
DUIE_REL2ID_PATH = 'models/rel2id_duie_fixed.json'            # 预训练时保存的关系映射
CUSTOM_DATA_DIR = 'data/data/computer_scientists'
MODEL_SAVE_PATH = 'models/relation_bert_cs_finetuned.pth'
FINAL_REL2ID_PATH = 'models/rel2id_cs.json'
BATCH_SIZE = 8
EPOCHS = 10
MAX_LEN = 128
LEARNING_RATE = 1e-5
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#目标关系
TARGET_RELATIONS = ["出生地", "毕业院校", "任职", "著作", "亲属"]

def load_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

def mark_text_with_entities(text, subj, obj):
    pos1 = text.find(subj)
    pos2 = text.find(obj)
    if pos1 == -1 or pos2 == -1:
        marked = text.replace(subj, f'[E1]{subj}[/E1]', 1)
        marked = marked.replace(obj, f'[E2]{obj}[/E2]', 1)
        return marked
    replacements = [(pos1, pos1+len(subj), subj, 'E1'),
                    (pos2, pos2+len(obj), obj, 'E2')]
    replacements.sort(key=lambda x: x[0], reverse=True)
    chars = list(text)
    for start, end, original, tag in replacements:
        chars[start:end] = list(f'[{tag}]{original}[/{tag}]')
    return ''.join(chars)

def build_samples(data, tokenizer, rel2id, max_len, neg_ratio=1):
    samples = []
    for item in data:
        text = item['text']
        spo_list = item.get('spo_list', [])
        if not spo_list:
            continue
        entities = set()
        for spo in spo_list:
            entities.add(spo['subject'])
            entities.add(spo['object'])
        entities_list = list(entities)
        positive_pairs = set((spo['subject'], spo['object']) for spo in spo_list)

        for spo in spo_list:
            pred = spo['predicate']
            if pred not in rel2id:
                continue
            marked = mark_text_with_entities(text, spo['subject'], spo['object'])
            encoding = tokenizer(marked, truncation=True, padding='max_length',
                                 max_length=max_len, return_tensors='pt')
            sample = {k: v.squeeze(0) for k, v in encoding.items()}
            sample['labels'] = torch.tensor(rel2id[pred], dtype=torch.long)
            samples.append(sample)

        if len(entities_list) < 2:
            continue
        neg_count = len(positive_pairs) * neg_ratio
        for _ in range(neg_count):
            s, o = random.sample(entities_list, 2)
            if (s, o) in positive_pairs or s == o:
                continue
            marked = mark_text_with_entities(text, s, o)
            encoding = tokenizer(marked, truncation=True, padding='max_length',
                                 max_length=max_len, return_tensors='pt')
            sample = {k: v.squeeze(0) for k, v in encoding.items()}
            sample['labels'] = torch.tensor(0, dtype=torch.long)
            samples.append(sample)
    return samples

class RelationDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        return self.samples[idx]

def finetune():
    tokenizer = BertTokenizer.from_pretrained(BERT_PATH)

    # 加载预训练的关系映射和权重
    with open(DUIE_REL2ID_PATH, 'r', encoding='utf-8') as f:
        duie_rel2id = json.load(f)
    duie_num_labels = len(duie_rel2id)

    # 构建目标关系映射
    rel2id = {rel: idx+1 for idx, rel in enumerate(TARGET_RELATIONS)}
    rel2id['no_relation'] = 0
    id2rel = {v: k for k, v in rel2id.items()}
    num_labels = len(rel2id)

    # 加载预训练模型
    model = BertForSequenceClassification.from_pretrained(BERT_PATH, num_labels=duie_num_labels)
    model.load_state_dict(torch.load(PRETRAINED_MODEL_PATH, map_location=DEVICE))
    # 提取旧的分类器
    old_classifier = model.classifier

    # 重新初始化模型为新的输出大小
    model = BertForSequenceClassification.from_pretrained(BERT_PATH, num_labels=num_labels)
    # 迁移权重
    with torch.no_grad():
        for new_label, rel_name in id2rel.items():
            if new_label == 0:
                model.classifier.weight[0] = old_classifier.weight[0]
                model.classifier.bias[0] = old_classifier.bias[0]
            else:
                old_label = duie_rel2id.get(rel_name)
                if old_label is not None and old_label != 0:
                    model.classifier.weight[new_label] = old_classifier.weight[old_label]
                    model.classifier.bias[new_label] = old_classifier.bias[old_label]
                # 否则保持随机初始化
    model.to(DEVICE)

    # 加载微调数据
    train_data = load_jsonl(os.path.join(CUSTOM_DATA_DIR, 'train.json'))
    dev_data = load_jsonl(os.path.join(CUSTOM_DATA_DIR, 'dev.json'))
    train_samples = build_samples(train_data, tokenizer, rel2id, MAX_LEN)
    dev_samples = build_samples(dev_data, tokenizer, rel2id, MAX_LEN)
    print(f"微调训练样本: {len(train_samples)}, 验证样本: {len(dev_samples)}")
    train_dataset = RelationDataset(train_samples)
    dev_dataset = RelationDataset(dev_samples)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=BATCH_SIZE, shuffle=False)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    best_f1 = 0.0
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f'Finetune Epoch {epoch+1}/{EPOCHS}')
        for batch in pbar:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            token_type_ids = batch['token_type_ids'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                            token_type_ids=token_type_ids, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
        print(f"Epoch {epoch+1} 平均训练损失: {total_loss/len(train_loader):.4f}")

        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in dev_loader:
                input_ids = batch['input_ids'].to(DEVICE)
                attention_mask = batch['attention_mask'].to(DEVICE)
                token_type_ids = batch['token_type_ids'].to(DEVICE)
                labels = batch['labels'].to(DEVICE)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                                token_type_ids=token_type_ids)
                preds = torch.argmax(outputs.logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        target_names = [id2rel[i] for i in range(num_labels)]
        report = classification_report(all_labels, all_preds,
                                       labels=list(range(num_labels)),
                                       target_names=target_names,
                                       output_dict=True, zero_division=0)
        f1 = report['macro avg']['f1-score']
        print(f"验证集 Macro F1: {f1:.4f}")
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"最佳模型已保存至 {MODEL_SAVE_PATH}")

    with open(FINAL_REL2ID_PATH, 'w', encoding='utf-8') as f:
        json.dump(rel2id, f, ensure_ascii=False, indent=2)
    print("微调完成！")

if __name__ == '__main__':
    finetune()