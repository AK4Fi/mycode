import os
import re
import csv
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def extract_opcode_sequence(asm_filepath):
    """从 asm 文件提取纯净的操作码序列"""
    opcode_seq = []
    # 匹配十六进制机器码后的有效操作码（至少2个字母）
    pattern = re.compile(r'\s([a-fA-F0-9]{2}\s)+\s*([a-z]{2,})')
    # 过滤伪指令和无意义指令
    exclude_ops = {"align", "db", "dd", "dw", "byte", "word", "dword", "extrn"}

    try:
        with open(asm_filepath, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                if line.startswith(".text"):
                    match = re.search(pattern, line)
                    if match:
                        opc = match.group(2)
                        if opc not in exclude_ops:
                            opcode_seq.append(opc)
    except Exception as e:
        print(f"Error reading {asm_filepath}: {e}")
    return " ".join(opcode_seq)

def main():
    # TODO: 替换为你的 Microsoft BIG 2015 解压目录和 trainLabels.csv 路径
    asm_dir = "/root/autodl-tmp/kaggle2015-sample/subtrain" 
    labels_csv = "/root/autodl-tmp/kaggle2015-sample/subtrain/subtrainLabels.csv"
    output_dir = "../data"
    os.makedirs(output_dir, exist_ok=True)

    df_labels = pd.read_csv(labels_csv)
    dataset_records = []

    print("开始提取 Opcode 序列...")
    for _, row in tqdm(df_labels.iterrows(), total=len(df_labels)):
        file_id = row['Id']
        label = row['Class'] - 1  # PyTorch 标签需从 0 开始 (0-8)
        
        asm_path = os.path.join(asm_dir, f"{file_id}.asm")
        if not os.path.exists(asm_path):
            continue
            
        opcodes = extract_opcode_sequence(asm_path)
        if len(opcodes.strip()) > 0:
            dataset_records.append({"id": file_id, "opcodes": opcodes, "label": label})

    # 保存为全量数据，并划分为 Train/Val/Test (8:1:1)
    df_all = pd.DataFrame(dataset_records)
    train_df, temp_df = train_test_split(df_all, test_size=0.2, random_state=42, stratify=df_all['label'])
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['label'])

    train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(output_dir, "val.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)
    print("数据提取完成，已保存至 data/ 目录。")

if __name__ == "__main__":
    main()