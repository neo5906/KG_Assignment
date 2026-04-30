import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering
from collections import defaultdict, Counter
import warnings


class ClusteringEntityDisambiguator:
    def __init__(self,
                 distance_threshold=0.2,          # 同类型内聚类阈值（余弦距离）
                 context_window=80,               # 上下文窗口大小（字符数）
                 use_svd=True,
                 svd_dim=100,                     # SVD 降维后的维度
                 no_disamb_types=None):           # 不需要消歧的类型
        self.distance_threshold = distance_threshold
        self.context_window = context_window
        self.use_svd = use_svd
        self.svd_dim = svd_dim
        self.no_disamb_types = no_disamb_types if no_disamb_types else ['DATE', 'WORK']
        self.tfidf_vectorizer = None
        self.svd = None

    def extract_context(self, text, start, end):
        #根据实体的实际位置 (start, end) 从原文中提取上下文。
        if start is None or end is None or start < 0 or end > len(text):
            # 若无有效位置，返回实体本身
            return text[start:end] if (start is not None and end is not None) else ""
        ctx_start = max(0, start - self.context_window)
        ctx_end = min(len(text), end + self.context_window)
        return text[ctx_start:ctx_end]

    def _build_features(self, contexts, fit=False):
        """
        构建特征表示：
        1. 用 TF‑IDF 向量化上下文
        2. 若开启 SVD，则降维并归一化得到密集语义向量
        参数 fit: 是否在当前数据上拟合 TF‑IDF 和 SVD（每个类型首次调用时应为 True）
        """
        if fit or self.tfidf_vectorizer is None:
            self.tfidf_vectorizer = TfidfVectorizer(max_features=5000)
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(contexts)
        else:
            tfidf_matrix = self.tfidf_vectorizer.transform(contexts)

        if self.use_svd:
            if fit or self.svd is None:
                self.svd = TruncatedSVD(n_components=self.svd_dim, random_state=42)
                dense = self.svd.fit_transform(tfidf_matrix)
            else:
                dense = self.svd.transform(tfidf_matrix)
            # 归一化，便于计算余弦相似度
            norms = np.linalg.norm(dense, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            return dense / norms
        else:
            # 直接使用 TF‑IDF 稀疏矩阵
            return tfidf_matrix

    def disambiguate(self, ner_results, full_text):
        # 按类型分组聚类的实体消歧。
        # 确保每个实体都有位置信息；若缺失则尝试查找并警告
        for idx, item in enumerate(ner_results):
            if 'start' not in item or 'end' not in item:
                warnings.warn(f"实体 '{item['entity']}' 缺少 start/end 字段")
                start = full_text.find(item['entity'])
                if start == -1:
                    start = 0
                    end = len(item['entity'])
                else:
                    end = start + len(item['entity'])
                ner_results[idx]['start'] = start
                ner_results[idx]['end'] = end

        # 按类型分组索引
        type2indices = defaultdict(list)
        for idx, item in enumerate(ner_results):
            type2indices[item['type']].append(idx)

        results = [None] * len(ner_results)

        for ent_type, indices in type2indices.items():
            # 对于不需要消歧的类型，每个提及直接作为唯一实体
            if ent_type in self.no_disamb_types:
                for i in indices:
                    item = ner_results[i]
                    results[i] = {
                        'mention': item['entity'],
                        'type': ent_type,
                        'linked_title': item['entity'],  # 自身
                        'score': 1.0,
                        'context': self.extract_context(full_text, item['start'], item['end'])
                    }
                continue

            # 提取同类型所有提及的上下文
            sub_entities = [ner_results[i]['entity'] for i in indices]
            sub_types = [ner_results[i]['type'] for i in indices]
            sub_contexts = [
                self.extract_context(full_text, ner_results[i]['start'], ner_results[i]['end'])
                for i in indices
            ]

            if len(indices) == 1:
                i = indices[0]
                results[i] = {
                    'mention': sub_entities[0],
                    'type': sub_types[0],
                    'linked_title': sub_entities[0],
                    'score': 1.0,
                    'context': sub_contexts[0]
                }
                continue

            # 构建特征（每个类型单独拟合）
            features = self._build_features(sub_contexts, fit=True)

            # 计算余弦距离矩阵
            sim_matrix = cosine_similarity(features)
            distance_matrix = 1.0 - sim_matrix

            # 层次聚类
            clustering = AgglomerativeClustering(
                n_clusters=None,
                metric='precomputed',
                linkage='complete',
                distance_threshold=self.distance_threshold
            )
            labels = clustering.fit_predict(distance_matrix)

            # 按簇聚合
            cluster_dict = defaultdict(list)
            for i_local, label in enumerate(labels):
                cluster_dict[label].append(i_local)

            for label, local_indices in cluster_dict.items():
                if len(local_indices) > 1:
                    sub_sim = sim_matrix[local_indices][:, local_indices]
                    avg_sims = sub_sim.mean(axis=1)
                    best_local = local_indices[np.argmax(avg_sims)]
                else:
                    best_local = local_indices[0]

                canonical_name = sub_entities[best_local]
                canonical_type = sub_types[best_local]

                for li in local_indices:
                    global_idx = indices[li]
                    score = float(1.0 - distance_matrix[li][best_local])
                    results[global_idx] = {
                        'mention': sub_entities[li],
                        'type': sub_types[li],
                        'linked_title': canonical_name,
                        'score': score,
                        'context': sub_contexts[li]
                    }
        return results


def main():
    # 输入输出路径（根据项目实际调整）
    NER_JSON = 'data/output/turing_entities_with_rules_se.json'   # 需包含位置字段的NER结果
    TEXT_FILE = 'data/raw/Turing_Baidu.txt'
    OUTPUT_JSON = 'data/output/turing_entities_cluster.json'

    with open(NER_JSON, 'r', encoding='utf-8') as f:
        ner_data = json.load(f)
    with open(TEXT_FILE, 'r', encoding='utf-8') as f:
        text = f.read()

    # 初始化消歧器
    disambiguator = ClusteringEntityDisambiguator(
        distance_threshold=0.2,
        context_window=80,
        use_svd=True,
        svd_dim=100,
        no_disamb_types=['DATE']
    )

    results = disambiguator.disambiguate(ner_data, text)

    # 输出统计
    unique_titles = set(r['linked_title'] for r in results)
    print(f"原始实体提及数: {len(ner_data)}, 消歧后归一化实体数: {len(unique_titles)}")

    type_counts = Counter(r['type'] for r in results)
    print("各类型提及分布:", type_counts)

    # 保存结果
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # 打印部分示例
    print("\n示例结果:")
    for ent_type in ['PER', 'LOC', 'ORG', 'DATE', 'WORK']:
        samples = [r for r in results if r['type'] == ent_type][:3]
        for r in samples:
            print(f"  {r['mention']} ({r['type']}) → {r['linked_title']} (相似度: {r['score']:.2f})")


if __name__ == '__main__':
    main()