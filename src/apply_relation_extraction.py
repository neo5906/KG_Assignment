# -*- coding: utf-8 -*-
import json, torch, re
from collections import defaultdict
from transformers import BertTokenizer, BertForSequenceClassification

# ---------- 配置 ----------
BERT_PATH = './bert-base-chinese'
MODEL_PATH = 'models/relation_bert_cs_finetuned.pth'
REL2ID_PATH = 'models/rel2id_cs.json'
ENTITY_FILE = 'data/output/turing_entities_cluster.json'
TEXT_FILE = 'data/raw/Turing_Baidu.txt'
OUTPUT_FILE = 'data/output/turing_triples_cs_finetuned.json'
MAX_LEN = 128
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

RELATION_TYPE_MAP = {
    "出生地":  ({"人物"}, {"地点"}),
    "毕业院校": ({"人物"}, {"机构"}),
    "任职":    ({"人物"}, {"机构"}),
    "著作":    ({"人物"}, {"作品"}),
    "亲属":    ({"人物"}, {"人物"}),
}

# 粗粒度 NER 类型 → 标准化类型
NER_TYPE_MAP = {
    'PER': '人物',
    'LOC': '地点',
    'ORG': '机构',
    'DATE': '日期',
    'WORK': '作品',
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


def predict_relation(tokenizer, model, sentence, subj_mention, obj_mention, id2rel):
    pos1 = sentence.find(subj_mention)
    pos2 = sentence.find(obj_mention)
    if pos1 == -1 or pos2 == -1:
        return 'no_relation'

    replacements = [
        (pos1, pos1 + len(subj_mention), subj_mention, 'E1'),
        (pos2, pos2 + len(obj_mention), obj_mention, 'E2')
    ]
    replacements.sort(key=lambda x: x[0], reverse=True)

    chars = list(sentence)
    for start, end, original, tag in replacements:
        chars[start:end] = list(f'[{tag}]{original}[/{tag}]')
    marked = ''.join(chars)

    inputs = tokenizer(marked, truncation=True, padding='max_length',
                       max_length=MAX_LEN, return_tensors='pt')
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    pred_id = torch.argmax(outputs.logits, dim=1).item()
    return id2rel[pred_id]


def load_name_to_type(disambiguation_file):
    name2type = {}
    try:
        with open(disambiguation_file, 'r', encoding='utf-8') as f:
            entities = json.load(f)
        for ent in entities:
            linked = ent.get('linked_title', '').strip()
            etype = ent.get('type', '').strip()
            if linked and etype:
                name2type[linked] = NER_TYPE_MAP.get(etype, etype)
    except Exception as e:
        print(f"加载实体类型失败: {e}")
    return name2type


def filter_triples_by_type(triples, name2type):
    if not name2type:
        return triples

    filtered = []
    for subj, rel, obj in triples:
        subj_type = name2type.get(subj)
        obj_type = name2type.get(obj)
        if not subj_type or not obj_type:
            continue

        subj_types, obj_types = RELATION_TYPE_MAP.get(rel, (set(), set()))
        if subj_type in subj_types and obj_type in obj_types:
            filtered.append((subj, rel, obj))
    return filtered


def extract_relations(text, entities, tokenizer, model, id2rel, name2type):
    name_map = {}
    for ent in entities:
        mention = ent.get('mention', '').strip()
        linked = ent.get('linked_title', '').strip()
        if linked:
            name_map[mention] = linked
            name_map[linked] = linked

    sentences = re.split(r'[。！？\n]+', text)
    triples = set()

    for sent in sentences:
        if len(sent) < 10:
            continue

        canonical_entities = set()
        for span, canon in name_map.items():
            if span in sent:
                canonical_entities.add(canon)
        if len(canonical_entities) < 2:
            continue

        canonical_list = list(canonical_entities)
        for i in range(len(canonical_list)):
            subj_canon = canonical_list[i]
            subj_mention = next((m for m, c in name_map.items() if c == subj_canon and m in sent), None)
            if not subj_mention:
                continue
            for j in range(i+1, len(canonical_list)):
                obj_canon = canonical_list[j]
                obj_mention = next((m for m, c in name_map.items() if c == obj_canon and m in sent), None)
                if not obj_mention:
                    continue

                rel = predict_relation(tokenizer, model, sent, subj_mention, obj_mention, id2rel)
                if rel != 'no_relation':
                    triples.add((subj_canon, rel, obj_canon))

    triples = filter_triples_by_type(list(triples), name2type)
    return triples


def main():
    print("加载模型...")
    tokenizer, model, id2rel = load_model()

    print("加载实体类型映射...")
    name2type = load_name_to_type(ENTITY_FILE)
    if name2type:
        print(f"加载了 {len(name2type)} 个实体的类型信息")

    print("读取文本...")
    with open(TEXT_FILE, 'r', encoding='utf-8') as f:
        text = f.read()

    print("加载实体消歧结果...")
    with open(ENTITY_FILE, 'r', encoding='utf-8') as f:
        entities = json.load(f)

    print("抽取关系...")
    triples = extract_relations(text, entities, tokenizer, model, id2rel, name2type)

    unique_triples = list(set(triples))
    print(f"共获得 {len(unique_triples)} 个三元组")

    rel_stat = defaultdict(list)
    for s, p, o in unique_triples:
        rel_stat[p].append((s, o))
    for rel, pairs in sorted(rel_stat.items()):
        print(f"\n【{rel}】({len(pairs)} 条):")
        for s, o in pairs[:5]:
            print(f"  {s} -> {o}")

    result = [{'subject': s, 'predicate': p, 'object': o} for s, p, o in unique_triples]
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存至 {OUTPUT_FILE}")


if __name__ == '__main__':
    main()