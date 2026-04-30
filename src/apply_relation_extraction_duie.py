# -*- coding: utf-8 -*-
import json, torch, re
from collections import defaultdict
from transformers import BertTokenizer, BertForSequenceClassification

# ---------- 配置 ----------
BERT_PATH = './bert-base-chinese'
MODEL_PATH = 'models/relation_bert_cs_finetuned.pth'
REL2ID_PATH = 'models/rel2id_cs.json'
SCHEMA_PATH = 'data/duie2/schema.json'
ENTITY_FILE = 'data/output/turing_entities_cluster.json'
TEXT_FILE = 'data/raw/Turing_Baidu.txt'
OUTPUT_FILE = 'data/output/turing_triples_cs_finetuned.json'
MAX_LEN = 128
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---------- 实体类型映射 ----------
# 你的粗粒度类型 → Schema 中的标准细粒度类型列表
NER_TYPE_TO_SCHEMA = {
    'PER': ['人物'],
    'LOC': ['地点', '城市', '国家'],
    'ORG': ['机构', '学校'],
    'DATE': ['Date'],
    'WORK': ['图书作品','影视作品'],
}

def load_model():
    tokenizer = BertTokenizer.from_pretrained(BERT_PATH)
    with open(REL2ID_PATH, 'r', encoding='utf-8') as f:
        rel2id = json.load(f)
    id2rel = {int(v): k for k, v in rel2id.items()}
    model = BertForSequenceClassification.from_pretrained(BERT_PATH, num_labels=len(rel2id))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return tokenizer, model, id2rel

def load_schema_valid_pairs(schema_path):
    valid_pairs = set()
    with open(schema_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line.strip())
            predicate = item['predicate']
            subject_type = item['subject_type']
            object_type = item['object_type']
            # 只处理简单客体（有 @value 的情况）
            if isinstance(object_type, dict) and '@value' in object_type:
                obj_type = object_type['@value']
                valid_pairs.add((subject_type, predicate, obj_type))
            # 对于复杂客体（如 inWork, onDate），也加入每种可能的类型
            if isinstance(object_type, dict):
                for key, value in object_type.items():
                    if key != '@value' and isinstance(value, str):
                        valid_pairs.add((subject_type, predicate, value))
    print(f"从 Schema 中加载了 {len(valid_pairs)} 个合法关系组合")
    return valid_pairs

def predict_relation(tokenizer, model, sentence, subj, obj, id2rel):
    marked = sentence.replace(subj, f'[E1]{subj}[/E1]', 1)
    marked = marked.replace(obj, f'[E2]{obj}[/E2]', 1)
    inputs = tokenizer(marked, truncation=True, padding='max_length',
                       max_length=MAX_LEN, return_tensors='pt')
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    pred = torch.argmax(outputs.logits, dim=1).item()
    return id2rel[pred]

def build_normalization_map(disambiguation_results):
    """根据消歧结果构建 文本片段 → 归一化名称 的映射"""
    name_map = {}
    for ent in disambiguation_results:
        mention = ent.get('mention', '').strip()
        linked = ent.get('linked_title', '').strip()
        if linked:
            name_map[mention] = linked
            name_map[linked] = linked   # 归一化名称也映射到自身
    return name_map

def extract_normalized_entities_from_sentence(sentence, name_map):
    # 收集所有出现的实体及其位置
    spans = []
    for text_span in name_map:
        start = sentence.find(text_span)
        while start != -1:
            spans.append((start, start + len(text_span), name_map[text_span]))
            start = sentence.find(text_span, start + 1)

    if not spans:
        return []

    # 按起始位置排序，然后过滤重叠，保留最长的
    spans.sort(key=lambda x: (x[0], -(x[1] - x[0])))  # 起始位置升序，长度降序
    filtered = []
    used_end = -1
    for start, end, canonical in spans:
        if start >= used_end:   # 不与前一个重叠
            filtered.append(canonical)
            used_end = end
    return list(set(filtered))  # 去重

def filter_triples(triples, entity_list, valid_pairs):
    """
    基于 Schema 类型约束过滤三元组，并自动修正方向。
    entity_list: 消歧后的实体列表，包含 type 和 linked_title。
    valid_pairs: 从 schema 中提取的合法 (subj_type, predicate, obj_type) 集合。
    返回符合 schema 的三元组列表。
    """
    # 构建归一化名称 → 原始类型 的映射
    name_to_type = {}
    for ent in entity_list:
        linked = ent.get('linked_title', '').strip()
        etype = ent.get('type', '').strip()
        if linked and etype in NER_TYPE_TO_SCHEMA:
            name_to_type[linked] = etype

    filtered = []
    seen = set()
    for subj, rel, obj in triples:
        # 获取主客体的原始类型
        subj_type = name_to_type.get(subj)
        obj_type = name_to_type.get(obj)
        if not subj_type or not obj_type:
            continue

        # 将原始类型映射为 Schema 中的可能类型
        subj_schema_types = NER_TYPE_TO_SCHEMA.get(subj_type, [subj_type])
        obj_schema_types = NER_TYPE_TO_SCHEMA.get(obj_type, [obj_type])

        # 检查是否存在合法的类型组合
        valid = False
        for s_type in subj_schema_types:
            for o_type in obj_schema_types:
                if (s_type, rel, o_type) in valid_pairs:
                    valid = True
                    break
            if valid:
                break

        if not valid:
            # 尝试交换主客体方向（因为模型可能预测反了）
            for s_type in obj_schema_types:      # 宾语类型作为主语
                for o_type in subj_schema_types: # 主语类型作为宾语
                    if (s_type, rel, o_type) in valid_pairs:
                        # 修正三元组方向
                        triple = (obj, rel, subj)
                        if triple not in seen:
                            seen.add(triple)
                            filtered.append(triple)
                        break
                else:
                    continue
                break
        else:
            triple = (subj, rel, obj)
            if triple not in seen:
                seen.add(triple)
                filtered.append(triple)

    return filtered

def extract_relations(text, entities, tokenizer, model, id2rel, valid_pairs):
    # 1. 构建文本到归一化名称的映射
    name_map = build_normalization_map(entities)

    # 2. 分句
    sentences = re.split(r'[。！？\n]+', text)
    triples = []

    for sent in sentences:
        if len(sent) < 5:
            continue
        normalized_entities = extract_normalized_entities_from_sentence(sent, name_map)
        if len(normalized_entities) < 2:
            continue

        # 3. 为每个归一化名词找到句子中的一个实际文本片段（用于标记）
        for i, subj_canon in enumerate(normalized_entities):
            # 查找 subj 在句中的任一对应文本
            subj_span = None
            for span, canon in name_map.items():
                if canon == subj_canon and span in sent:
                    subj_span = span
                    break
            if not subj_span:
                continue

            for j, obj_canon in enumerate(normalized_entities):
                if i == j:
                    continue
                obj_span = None
                for span, canon in name_map.items():
                    if canon == obj_canon and span in sent:
                        obj_span = span
                        break
                if not obj_span:
                    continue

                # 预测关系
                rel = predict_relation(tokenizer, model, sent, subj_span, obj_span, id2rel)
                if rel != 'no_relation':
                    triples.append((subj_canon, rel, obj_canon))

    # 4. 基于 Schema 过滤并修正方向
    filtered_triples = filter_triples(triples, entities, valid_pairs)
    return filtered_triples

def main():
    print("加载关系抽取模型...")
    tokenizer, model, id2rel = load_model()
    print("加载 Schema 中的合法关系组合...")
    valid_pairs = load_schema_valid_pairs(SCHEMA_PATH)
    print("加载实体消歧结果...")
    with open(ENTITY_FILE, 'r', encoding='utf-8') as f:
        entities = json.load(f)
    print("正在抽取关系...")
    with open(TEXT_FILE, 'r', encoding='utf-8') as f:
        text = f.read()
    triples = extract_relations(text, entities, tokenizer, model, id2rel, valid_pairs)

    unique_triples = list(set(triples))
    print(f"抽取完成，共获得 {len(unique_triples)} 个三元组：")
    triples_by_rel = defaultdict(list)
    for s, r, o in unique_triples:
        triples_by_rel[r].append((s, o))
    for rel, pairs in sorted(triples_by_rel.items()):
        print(f"\n【{rel}】({len(pairs)} 条):")
        for s, o in pairs[:5]:
            print(f"  {s} -> {o}")

    result = [{'subject': s, 'predicate': r, 'object': o} for s, r, o in unique_triples]
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n三元组已保存至 {OUTPUT_FILE}")

if __name__ == '__main__':
    main()