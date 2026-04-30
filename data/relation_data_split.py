import json
import random
from pathlib import Path

def split_jsonl(input_file, train_file, dev_file, train_ratio=0.8, seed=42):
    """
    将 JSONL 文件按比例随机划分为训练集和验证集。

    Args:
        input_file: 输入的 JSONL 文件路径，每行一个 JSON 对象。
        train_file: 输出的训练集文件路径。
        dev_file: 输出的验证集文件路径。
        train_ratio: 训练集所占比例（默认0.8）。
        seed: 随机种子，保证划分可复现。
    """
    # 读取所有数据
    data = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(line)  # 保留原始字符串以便直接写入

    total = len(data)
    print(f"读取到 {total} 条数据")

    # 随机打乱
    random.seed(seed)
    random.shuffle(data)

    # 划分
    split_idx = int(total * train_ratio)
    train_data = data[:split_idx]
    dev_data = data[split_idx:]

    # 写入训练集
    Path(train_file).parent.mkdir(parents=True, exist_ok=True)
    with open(train_file, 'w', encoding='utf-8') as f:
        for line in train_data:
            f.write(line + '\n')
    print(f"训练集已保存至 {train_file}，共 {len(train_data)} 条")

    # 写入验证集
    with open(dev_file, 'w', encoding='utf-8') as f:
        for line in dev_data:
            f.write(line + '\n')
    print(f"验证集已保存至 {dev_file}，共 {len(dev_data)} 条")


if __name__ == '__main__':
    # 配置路径（请根据实际情况修改）
    INPUT_FILE = 'data/computer_scientists_annotated.jsonl'   # 原始标注文件
    TRAIN_FILE = 'data/computer_scientists/train.json'        # 输出训练集
    DEV_FILE = 'data/computer_scientists/dev.json'            # 输出验证集

    split_jsonl(INPUT_FILE, TRAIN_FILE, DEV_FILE, train_ratio=0.8, seed=42)