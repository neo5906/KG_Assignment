# train_duie_base.py
import os, json, random
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from sklearn.metrics import classification_report
from tqdm import tqdm

# ---------- 配置 ----------
BERT_PATH = './bert-base-chinese'
DATA_DIR = 'data/duie2'
MODEL_SAVE_PATH = 'models/relation_bert_duie_fixed.pth'
REL2ID_PATH = 'models/rel2id_duie_fixed.json'
BATCH_SIZE = 16
EPOCHS = 5
MAX_LEN = 128
LEARNING_RATE = 2e-5
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_duie_data(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

def get_all_relations(schema_path):
    schema = []
    with open(schema_path, 'r', encoding='utf-8') as f:
        try:
            schema = json.load(f)
            if isinstance(schema, dict):
                schema = [schema]
        except json.JSONDecodeError:
            f.seek(0)
            for line in f:
                line = line.strip()
                if line:
                    schema.append(json.loads(line))
    relations = []
    for item in schema:
        pred = item['predicate']
        obj_type = item.get('object_type', {})
        if isinstance(obj_type, dict) and '@value' in obj_type:
            relations.append(pred)
    relations = list(set(relations))
    relations.sort()
    rel2id = {rel: i+1 for i, rel in enumerate(relations)}
    rel2id['no_relation'] = 0
    id2rel = {v: k for k, v in rel2id.items()}
    return rel2id, id2rel

def mark_text_with_entities(text, subj, obj):
    """稳健的实体标记，从后往前替换避免重叠"""
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
        simple_spos = []
        for spo in spo_list:
            obj = spo['object']
            if isinstance(obj, dict) and '@value' in obj:
                simple_spos.append({
                    'subject': spo['subject'],
                    'predicate': spo['predicate'],
                    'object': obj['@value']
                })
        if not simple_spos:
            continue   # 没有简单关系，不生成样本

        entities = set()
        for spo in simple_spos:
            entities.add(spo['subject'])
            entities.add(spo['object'])
        entities_list = list(entities)
        positive_pairs = set((spo['subject'], spo['object']) for spo in simple_spos)

        # 正样本
        for spo in simple_spos:
            subj, obj, pred = spo['subject'], spo['object'], spo['predicate']
            if pred not in rel2id:
                continue
            marked = mark_text_with_entities(text, subj, obj)
            encoding = tokenizer(marked, truncation=True, padding='max_length',
                                 max_length=max_len, return_tensors='pt')
            sample = {k: v.squeeze(0) for k, v in encoding.items()}
            sample['labels'] = torch.tensor(rel2id[pred], dtype=torch.long)
            samples.append(sample)

        # 负样本
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

def train():
    tokenizer = BertTokenizer.from_pretrained(BERT_PATH)
    train_data = load_duie_data(os.path.join(DATA_DIR, 'train.json'))
    dev_data = load_duie_data(os.path.join(DATA_DIR, 'dev.json'))
    rel2id, id2rel = get_all_relations(os.path.join(DATA_DIR, 'schema.json'))
    print(f"关系类别数: {len(rel2id)}")
    os.makedirs('models', exist_ok=True)
    with open(REL2ID_PATH, 'w', encoding='utf-8') as f:
        json.dump(rel2id, f, ensure_ascii=False, indent=2)

    train_samples = build_samples(train_data, tokenizer, rel2id, MAX_LEN)
    dev_samples = build_samples(dev_data, tokenizer, rel2id, MAX_LEN)
    print(f"训练样本数: {len(train_samples)}, 验证样本数: {len(dev_samples)}")
    train_dataset = RelationDataset(train_samples)
    dev_dataset = RelationDataset(dev_samples)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = BertForSequenceClassification.from_pretrained(BERT_PATH, num_labels=len(rel2id))
    model.to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

    best_f1 = 0.0
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{EPOCHS}')
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
        target_names = [id2rel[i] for i in range(len(rel2id))]
        report = classification_report(all_labels, all_preds,
                                       labels=list(range(len(rel2id))),
                                       target_names=target_names,
                                       output_dict=True, zero_division=0)
        f1 = report['macro avg']['f1-score']
        print(f"验证集 Macro F1: {f1:.4f}")
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"最佳模型已保存至 {MODEL_SAVE_PATH}")
    print("预训练完成！")

if __name__ == '__main__':
    train()