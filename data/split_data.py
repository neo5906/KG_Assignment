import os
import random

INPUT_FILE = "data/computer.txt"
OUTPUT_DIR = "data"
TRAIN_FILE = os.path.join(OUTPUT_DIR, "train.txt")
TEST_FILE = os.path.join(OUTPUT_DIR, "test.txt")
TEST_RATIO = 0.2
RANDOM_SEED = 42

# 允许的标签集合（BIO 标注，只有 PER, ORG, LOC 三种实体）
ALLOWED_LABELS = {
    'O',
    'B-PER', 'I-PER',
    'B-ORG', 'I-ORG',
    'B-LOC', 'I-LOC',
}


def fix_paragraph_labels(paragraph: str) -> str:
    """
    检查段落中的每一行，如果标签不在 ALLOWED_LABELS 中，则将其替换为 'O'。
    返回修正后的段落字符串（保持原格式：每行 "字符 标签"，行尾换行）。
    """
    lines = paragraph.split('\n')
    fixed_lines = []
    for line in lines:
        if not line.strip():
            fixed_lines.append(line)  # 保留空白行（但通常段落内无空行）
            continue
        parts = line.strip().split()
        if len(parts) < 2:
            # 格式异常的行，直接保留原行（但一般不会发生）
            fixed_lines.append(line)
            continue
        char = parts[0]
        label = parts[1]
        if label not in ALLOWED_LABELS:
            # 替换为 O
            label = 'O'
        fixed_lines.append(f"{char} {label}")
    # 重新组合，保持原有换行
    return '\n'.join(fixed_lines)


def read_and_fix_paragraphs(filepath):
    """
    读取 computer.txt，按空行分割段落，修正每个段落中的非法标签。
    返回修正后的段落列表（每个段落末尾保留一个换行）。
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    # 按空行分割段落（两个连续换行符）
    raw_paragraphs = [p for p in content.split('\n\n') if p.strip()]
    print(f"原始段落总数: {len(raw_paragraphs)}")

    fixed_paragraphs = []
    for idx, para in enumerate(raw_paragraphs, 1):
        fixed_para = fix_paragraph_labels(para)
        fixed_paragraphs.append(fixed_para.strip() + '\n')
    print(f"已修复所有段落，输出段落数: {len(fixed_paragraphs)}")
    return fixed_paragraphs


def write_paragraphs(paragraphs, output_path):
    """将段落列表写入文件，段落之间用空行（两个换行）分隔"""
    with open(output_path, 'w', encoding='utf-8') as f:
        for i, para in enumerate(paragraphs):
            f.write(para)
            if i != len(paragraphs) - 1:
                f.write('\n')  # 段落之间一个空行


def main():
    if not os.path.exists(INPUT_FILE):
        print(f"错误：找不到 {INPUT_FILE}")
        return

    print("正在读取并修复 computer.txt ...")
    fixed_paragraphs = read_and_fix_paragraphs(INPUT_FILE)
    if not fixed_paragraphs:
        print("没有段落，退出。")
        return

    # 随机打乱并划分
    random.seed(RANDOM_SEED)
    shuffled = fixed_paragraphs.copy()
    random.shuffle(shuffled)

    split_idx = int(len(shuffled) * (1 - TEST_RATIO))
    train_paras = shuffled[:split_idx]
    test_paras = shuffled[split_idx:]

    print(f"训练集段落数: {len(train_paras)}")
    print(f"测试集段落数: {len(test_paras)}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    write_paragraphs(train_paras, TRAIN_FILE)
    write_paragraphs(test_paras, TEST_FILE)

    print(f"训练集已保存至 {TRAIN_FILE}")
    print(f"测试集已保存至 {TEST_FILE}")


if __name__ == "__main__":
    main()