import os
import json
import pandas as pd
import networkx as nx
import numpy as np
from collections import Counter
import concurrent.futures
from tqdm import tqdm
import re

# 将正则提取到全局，避免多进程重复编译
LABEL_PATTERN = re.compile(r'^\s*([a-zA-Z0-9_\.]+):')
OPCODE_PATTERN = re.compile(r'\s([a-fA-F0-9]{2}\s)+\s*([a-z]+)\s')
JUMP_PATTERN = re.compile(r"\s([a-fA-F0-9]{2}\s)+\s*(j[a-z]+|call)\s+([a-zA-Z0-9_\.]+)")

def compute_entropy(csv_files, output_json):
    print("🌟 [1/2] 正在计算全局操作码香农信息熵...")
    counts = Counter()
    
    for f in csv_files:
        if os.path.exists(f):
            print(f"   正在统计 {os.path.basename(f)}...")
            df = pd.read_csv(f)
            # 优化内存：直接更新 Counter，不要把几千万个词全部放进一个内存 List 里
            for seq in tqdm(df['opcodes'].dropna(), desc="   计算词频"):
                counts.update(str(seq).split())
                
    total = sum(counts.values())
    if total == 0:
        print("⚠️ 警告：没有统计到任何操作码！")
        return
        
    entropy_dict = {op: -np.log2(count / total) for op, count in counts.items()}
    
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, 'w') as f:
        json.dump(entropy_dict, f)
    print(f"✅ 信息熵字典已保存至 {output_json} (共 {len(entropy_dict)} 种)")

def process_single_cfg_guided(args):
    """多进程 Worker：处理单个文件的 CFG 构建与 PageRank 采样"""
    file_id, fallback_seq, asm_dir = args
    asm_path = os.path.join(asm_dir, f"{file_id}.asm")
    
    if not os.path.exists(asm_path):
        return file_id, fallback_seq

    G = nx.DiGraph()
    block_ops = {"entry": []}
    curr_block = "entry"
    G.add_node(curr_block)

    try:
        with open(asm_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                if not line.startswith(".text"): continue
                
                # 遇到标签 (节点)
                label_match = re.search(LABEL_PATTERN, line)
                if label_match:
                    new_block = label_match.group(1)
                    G.add_node(new_block)
                    # 补充连边逻辑，否则 PageRank 将无法发挥作用
                    G.add_edge(curr_block, new_block)
                    curr_block = new_block
                    if curr_block not in block_ops: 
                        block_ops[curr_block] = []
                
                # 提取操作码
                op_match = re.search(OPCODE_PATTERN, line)
                if op_match: 
                    block_ops[curr_block].append(op_match.group(2))
                    
                # 遇到跳转指令 (补充连边)
                jmp_match = re.search(JUMP_PATTERN, line)
                if jmp_match:
                    target_block = jmp_match.group(3)
                    G.add_node(target_block)
                    G.add_edge(curr_block, target_block)
                    
    except Exception: 
        pass

    # 计算 PageRank
    try:
        # 修正悬挂节点可能导致的报错
        scores = nx.pagerank(G, alpha=0.85, max_iter=50)
    except Exception:
        scores = {node: 1.0 for node in G.nodes()}

    # 取重要性排名前 30% 的核心代码块
    num_top = max(1, int(len(scores) * 0.3))
    top_blocks = set(sorted(scores, key=scores.get, reverse=True)[:num_top])
    
    final_ops = []
    for block in block_ops.keys():
        if block in top_blocks: 
            final_ops.extend(block_ops[block])
    
    # 返回 file_id 和新序列，以便主进程更新 DataFrame
    return file_id, " ".join(final_ops)

def extract_cfg_guided_opcodes(asm_dir, csv_files):
    print("\n🌟 [2/2] 正在执行 CFG 结构引导采样 (PageRank)...")
    num_workers = 32
    
    for csv_file in csv_files:
        if not os.path.exists(csv_file): 
            continue
            
        df = pd.read_csv(csv_file)
        print(f"\n🚀 启动 {num_workers} 进程处理: {os.path.basename(csv_file)}")
        
        # 准备任务参数
        tasks = [(row['id'], row['opcodes'], asm_dir) for _, row in df.iterrows()]
        
        # 收集结果的字典 {file_id: final_seq}
        results_dict = {}
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(process_single_cfg_guided, task): task for task in tasks}
            
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
                try:
                    file_id, final_seq = future.result()
                    results_dict[file_id] = final_seq
                except Exception as e:
                    print(f"Task failed: {e}")

        # 使用 map 根据 id 更新对应的行
        df['cfg_guided_opcodes'] = df['id'].map(results_dict)
        df.to_csv(csv_file, index=False)
        print(f"✅ {os.path.basename(csv_file)} 特征注入完毕，已保存。")

if __name__ == "__main__":
    # # subtrain
    # CSV_FILES = ["./data/train.csv", "./data/val.csv", "./data/test.csv"]
    # ASM_DIR = "/root/autodl-tmp/kaggle2015-sample/subtrain" 
    # sub_entropy_path = './data/opcode_entropy.json'
    # compute_entropy(CSV_FILES, sub_entropy_path)
    # extract_cfg_guided_opcodes(ASM_DIR, CSV_FILES)
    
    # full_data
    csv_files = [
        "/root/autodl-tmp/Kaggle2015/full_data/train.csv",
        "/root/autodl-tmp/Kaggle2015/full_data/val.csv",
        "/root/autodl-tmp/Kaggle2015/full_data/test.csv",
    ]
    ASM_DIR = "/root/autodl-tmp/Kaggle2015/train"
    sub_entropy_path = '/root/autodl-tmp/Kaggle2015/full_data/opcode_entropy.json'
    
    compute_entropy(csv_files, sub_entropy_path)
    extract_cfg_guided_opcodes(ASM_DIR, csv_files)