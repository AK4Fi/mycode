# 文件位置: data_preprocess/3_advanced_features.py
import os
import json
import pandas as pd
import networkx as nx
import numpy as np
from collections import Counter
from tqdm import tqdm
import re

def compute_entropy(csv_files, output_json):
    print("🌟 [1/2] 正在计算全局操作码香农信息熵...")
    all_opcodes = []
    for f in csv_files:
        if os.path.exists(f):
            df = pd.read_csv(f)
            for seq in df['opcodes'].dropna():
                all_opcodes.extend(str(seq).split())
                
    counts = Counter(all_opcodes)
    total = sum(counts.values())
    entropy_dict = {op: -np.log2(count / total) for op, count in counts.items()}
    
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, 'w') as f:
        json.dump(entropy_dict, f)
    print(f"✅ 信息熵字典已保存至 {output_json} (共 {len(entropy_dict)} 种)")

def extract_cfg_guided_opcodes(asm_dir, csv_files):
    print("\n🌟 [2/2] 正在执行 CFG 结构引导采样 (PageRank)...")
    label_pattern = re.compile(r'^\s*([a-zA-Z0-9_\.]+):')
    opcode_pattern = re.compile(r'\s([a-fA-F0-9]{2}\s)+\s*([a-z]+)\s')

    for csv_file in csv_files:
        if not os.path.exists(csv_file): continue
        df = pd.read_csv(csv_file)
        
        cfg_guided_seqs = []
        for file_id in tqdm(df['id'], desc=f"处理 {os.path.basename(csv_file)}"):
            asm_path = os.path.join(asm_dir, f"{file_id}.asm")
            # 如果没找到 asm 文件，用原序列兜底
            if not os.path.exists(asm_path):
                cfg_guided_seqs.append(df.loc[df['id']==file_id, 'opcodes'].values[0])
                continue

            G = nx.DiGraph()
            block_ops = {"entry": []}
            curr_block = "entry"

            try:
                with open(asm_path, 'r', encoding='utf-8', errors='ignore') as f:
                    for line in f:
                        if not line.startswith(".text"): continue
                        label_match = re.search(label_pattern, line)
                        if label_match:
                            curr_block = label_match.group(1)
                            G.add_node(curr_block)
                            if curr_block not in block_ops: block_ops[curr_block] = []
                        
                        op_match = re.search(opcode_pattern, line)
                        if op_match: block_ops[curr_block].append(op_match.group(2))
            except Exception: pass

            try:
                scores = nx.pagerank(G, alpha=0.85)
            except:
                scores = {node: 1.0 for node in G.nodes()}

            # 取重要性排名前 30% 的核心代码块
            top_blocks = set(sorted(scores, key=scores.get, reverse=True)[:max(1, int(len(scores)*0.3))])
            final_ops = []
            for block in block_ops.keys():
                if block in top_blocks: final_ops.extend(block_ops[block])
            
            cfg_guided_seqs.append(" ".join(final_ops))
            
        df['cfg_guided_opcodes'] = cfg_guided_seqs
        df.to_csv(csv_file, index=False)
        print(f"✅ {os.path.basename(csv_file)} 特征注入完毕")

if __name__ == "__main__":
    CSV_FILES = ["./data/train.csv", "./data/val.csv", "./data/test.csv"]
    ASM_DIR = "/root/autodl-tmp/kaggle2015-sample/subtrain" 
    
    compute_entropy(CSV_FILES, './data/opcode_entropy.json')
    extract_cfg_guided_opcodes(ASM_DIR, CSV_FILES)