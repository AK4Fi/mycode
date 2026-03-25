import os
import re
import csv
import pandas as pd
import concurrent.futures
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# 将正则表达式和排除集合提取为全局变量，避免在每个函数调用中重复编译，提升多进程效率
OPCODE_PATTERN = re.compile(r'\s([a-fA-F0-9]{2}\s)+\s*([a-z]{2,})')
EXCLUDE_OPS = {"align", "db", "dd", "dw", "byte", "word", "dword", "extrn"}

def extract_opcode_sequence(asm_filepath):
    """从 asm 文件提取纯净的操作码序列"""
    opcode_seq = []
    
    try:
        with open(asm_filepath, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                if line.startswith(".text"):
                    match = re.search(OPCODE_PATTERN, line)
                    if match:
                        opc = match.group(2)
                        if opc not in EXCLUDE_OPS:
                            opcode_seq.append(opc)
    except Exception as e:
        print(f"Error reading {asm_filepath}: {e}")
    return " ".join(opcode_seq)

def process_single_file(args):
    """供多进程调用的单文件处理 worker"""
    file_id, label, asm_dir = args
    asm_path = os.path.join(asm_dir, f"{file_id}.asm")
    
    if not os.path.exists(asm_path):
        return None
        
    opcodes = extract_opcode_sequence(asm_path)
    if len(opcodes.strip()) > 0:
        return {"id": file_id, "opcodes": opcodes, "label": label}
    return None

def main():
    # subtrain_dir = "/root/autodl-tmp/kaggle2015-sample"
    # asm_dir = subtrain_dir + "/subtrain" 
    # labels_csv = "/root/autodl-tmp/kaggle2015-sample/subtrain/subtrainLabels.csv"
    # output_dir = "../data"
    
    # full_data
    asm_dir = "/root/autodl-tmp/Kaggle2015/train" 
    labels_csv = "/root/autodl-tmp/Kaggle2015/trainLabels.csv"
    output_dir = "/root/autodl-tmp/Kaggle2015/full_data"
    os.makedirs(output_dir, exist_ok=True)

    df_labels = pd.read_csv(labels_csv)
    dataset_records = []

    # 准备多进程任务参数
    tasks = [(row['Id'], row['Class'] - 1, asm_dir) for _, row in df_labels.iterrows()]
    
    # 获取当前机器的 CPU 核心数，你也可以手动指定 max_workers=8 等
    cpu_cores = 32
    print(f"开始提取 Opcode 序列，启动 {cpu_cores} 个 CPU 核心...")

    # 使用多进程池
    with concurrent.futures.ProcessPoolExecutor(max_workers=cpu_cores) as executor:
        # 提交所有任务
        futures = {executor.submit(process_single_file, task): task for task in tasks}
        
        # 使用 tqdm 配合 as_completed 监控进度
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            result = future.result()
            if result is not None:
                dataset_records.append(result)

    # 保存为全量数据，并划分为 Train/Val/Test (8:1:1)
    df_all = pd.DataFrame(dataset_records)
    train_df, temp_df = train_test_split(df_all, test_size=0.2, random_state=42, stratify=df_all['label'])
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['label'])

    train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(output_dir, "val.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)
    print(f"数据提取完成，已保存至 {output_dir} 目录。")

if __name__ == "__main__":
    main()