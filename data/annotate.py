import os
import json
import re
import time
from openai import OpenAI

# ===================== 配置 =====================
DEEPSEEK_API_KEY = 'sk-6a847598570445caa8ad7afe7f505ed5'
if not DEEPSEEK_API_KEY:
    # 也可以直接在这里填写，但注意保密
    raise ValueError("请设置环境变量 DEEPSEEK_API_KEY")

RAW_DATA_DIR = "raw_data"
OUTPUT_FILE = "data/computer.txt"
SLEEP_INTERVAL = 1.0                # 每次 API 调用后的休眠秒数
MAX_LINE_LEN = 3000                 # 单行最大字符数（超过则截断，避免超长行）
ADD_METADATA = True                 # 是否在输出 JSON 中添加来源信息

client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com"
)

# ===================== Prompt（与之前相同） =====================
SYSTEM_PROMPT = """你是一个专业的中文命名实体标注专家，现在要标注一些计算机领域杰出人物的百科文本。请严格按照 BIO 标注体系对输入文本进行实体标注。

支持的实体类型：
- PER: 人名（真实或虚构，仅指姓名本身，不包括头衔）
- ORG: 组织（公司、高校、实验室、学会等）
- LOC: 地域（国家、城市、山川等地理实体）

BIOES 标记含义：
- B-XXX: 实体的开始字符
- I-XXX: 实体的中间与结束字符
- O: 非实体

输出格式：每行一个字符及其标签，中间用空格分隔。非实体字符标签为 O。
示例：
输入：李明出生在北京，毕业于清华大学。
输出：
李 B-PER
明 I-PER
出 O
生 O
在 O
北 B-LOC
京 I-LOC
， O
毕 O
业 O
于 O
清 B-ORG
华 I-ORG
大 I-ORG
学 I-ORG
。 O

只输出每行的标签序列，不要输出任何额外解释。"""

def build_user_prompt(text: str) -> str:
    return f"请标注以下文本：\n{text}"

def clean_text(text: str) -> str:
    """简单清洗：移除ASCII控制字符，合并多余空白（保留换行符但本脚本逐行处理，无换行）"""
    text = re.sub(r'[\x00-\x1f\x7f]', ' ', text)   # 控制字符变空格
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def annotate_line(text: str, retry: int = 3) -> str:
    """调用API，直接返回模型输出的原始字符串"""
    if not text or not text.strip():
        return ""
    cleaned = clean_text(text)
    for attempt in range(retry):
        try:
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_user_prompt(cleaned)}
                ],
                temperature=0.1
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"      API调用失败 (尝试 {attempt+1}/{retry}): {e}")
            time.sleep(2 ** attempt)
    raise RuntimeError(f"无法标注该行，已重试 {retry} 次")

def process_file(filepath: str):
    print(f"正在处理: {filepath}")

    # 自动检测编码
    encodings = ['utf-8', 'gbk', 'gb18030']
    lines = None
    used_enc = None
    for enc in encodings:
        try:
            with open(filepath, 'r', encoding=enc) as f:
                lines = f.readlines()
            used_enc = enc
            break
        except UnicodeDecodeError:
            continue
    if lines is None:
        raise ValueError(f"无法读取文件 {filepath}")

    print(f"  使用编码 {used_enc}，共 {len(lines)} 行")

    # 确保输出目录存在
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    with open(OUTPUT_FILE, "a", encoding="utf-8") as out_f:
        for idx, line in enumerate(lines, start=1):
            line = line.rstrip('\n\r')
            if not line:
                continue   # 跳过空行

            print(f"  标注第 {idx} 行 (长度 {len(line)})...")
            try:
                bioes_output = annotate_line(line)
            except Exception as e:
                print(f"    第 {idx} 行标注失败: {e}，跳过")
                continue

            # 写入分隔标记（便于区分不同人物和行）
            # out_f.write(f"# source: {os.path.basename(filepath)} line:{idx}\n")
            out_f.write(bioes_output)
            if not bioes_output.endswith('\n'):
                out_f.write('\n')
            out_f.write('\n')   # 额外空行分隔不同段落
            time.sleep(SLEEP_INTERVAL)

    print(f"  完成: {os.path.basename(filepath)}")

def main():
    if not os.path.isdir(RAW_DATA_DIR):
        print(f"错误: 目录 {RAW_DATA_DIR} 不存在")
        return

    files = [f for f in os.listdir(RAW_DATA_DIR) if f.endswith(".txt")]
    if not files:
        print(f"错误: {RAW_DATA_DIR} 中没有 .txt 文件")
        return

    files.sort()
    for filename in files:
        filepath = os.path.join(RAW_DATA_DIR, filename)
        try:
            process_file(filepath)
        except Exception as e:
            print(f"处理文件 {filename} 时发生严重错误: {e}")
            continue

    print("所有文件处理完毕。")

if __name__ == "__main__":
    main()