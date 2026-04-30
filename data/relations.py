import openai
import json
import re
import time
from pathlib import Path

# ===== 配置 =====
DEEPSEEK_API_KEY = "sk-6a847598570445caa8ad7afe7f505ed5"
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"  # 或官方指定的 base URL
MODEL = "deepseek-chat"  # 推荐 deepseek-chat，便宜且好用
INPUT_DIR = "raw_data"          # 存放8个人物txt文件的目录
OUTPUT_FILE = "data/computer_scientists_annotated.jsonl"

SYSTEM_PROMPT = """你是一个信息抽取专家。你的任务是从给定的中文句子中，抽取出人物（计算机科学家）的关系三元组。

**关系类型定义：**
1. 出生地：头实体是人物，尾实体是具体地区（城市、国家等）。句子中明确提到“出生于”、“出生在”等。
2. 毕业院校：头实体是人物，尾实体是机构（学校、研究所等）。句子中明确提到“毕业于”、“获得...学位”、“就读于”等。
3. 任职：头实体是人物，尾实体是机构。句子中明确提到“担任”、“任职”、“受聘”、“进入...工作”等。
4. 著作：头实体是人物，尾实体是作品名称（论文、书籍、专利等）。句子中明确提到“发表”、“出版”、“撰写”等。
5. 亲属：头实体是人物，尾实体也是人物。包含父母、配偶、子女等。句子中明确提到“父亲是”、“母亲是”、“配偶”、“结婚”等。

**规则：**
- 只抽取句子中**明确且直接**陈述的关系，不要进行任何推理。
- 实体名称必须使用**完整的原始名称**，不要缩略或改写。例如句子中是“艾伦·图灵”就不要写成“图灵”。如果句子中只出现了简称，则用简称。
- 同一个句子中如果包含多个关系，全部列出。
- 输出格式必须是一个 JSON 数组，每个元素包含 "subject"、"predicate"、"object" 三个字段。如果没有找到任何关系，输出 []。
- 不要在 JSON 外添加任何解释性文字。"""

# ===== 函数定义 =====
def split_sentences(text):
    """简单分句，保留标点"""
    # 先按常见句末标点切分
    raw_sents = re.split(r'[。！？\n]+', text)
    sentences = []
    for s in raw_sents:
        s = s.strip()
        if len(s) >= 10:   # 过滤太短的句子
            sentences.append(s)
    return sentences

def call_deepseek(sentence):
    """调用 DeepSeek API 标注一个句子"""
    client = openai.OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"句子：“{sentence}”"}
            ],
            temperature=0,
            max_tokens=300
        )
        raw_output = response.choices[0].message.content.strip()
        # 清理可能包裹的 markdown 代码块
        if raw_output.startswith("```json"):
            raw_output = raw_output[7:]
        if raw_output.endswith("```"):
            raw_output = raw_output[:-3]
        raw_output = raw_output.strip()

        # 解析 JSON
        triples = json.loads(raw_output)
        # 简单验证格式
        if not isinstance(triples, list):
            raise ValueError("输出不是列表")
        valid_triples = []
        for t in triples:
            if isinstance(t, dict) and all(k in t for k in ("subject", "predicate", "object")):
                # 只保留我们定义的关系
                if t["predicate"] in {"出生地", "毕业院校", "任职", "著作", "亲属"}:
                    valid_triples.append(t)
        return valid_triples
    except Exception as e:
        print(f"Error processing sentence: {sentence[:50]}... | {e}")
        return []

def annotate_text(text, delay=0.2):
    """标注一段完整文本"""
    sentences = split_sentences(text)
    all_results = []
    for sent in sentences:
        triples = call_deepseek(sent)
        if triples:
            all_results.append({"text": sent, "spo_list": triples})
        time.sleep(delay)   # 避免触发速率限制
    return all_results

# ===== 主流程 =====
def main():
    input_files = list(Path(INPUT_DIR).glob("*.txt"))
    print(f"找到 {len(input_files)} 个文本文件")
    all_annotations = []

    for file_path in input_files:
        print(f"正在处理: {file_path.name}")
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        results = annotate_text(text)
        all_annotations.extend(results)

    # 保存为 JSONL
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for item in all_annotations:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"标注完成，共 {len(all_annotations)} 条带关系句子，已保存至 {OUTPUT_FILE}")

if __name__ == "__main__":
    main()