# 文件位置: core/dataset.py
import os
import json
import torch
import pandas as pd
from torch_geometric.data import Dataset
from transformers import RobertaTokenizer

class MalwareMultimodalDataset(Dataset):
    def __init__(self, csv_file, graph_dir, max_len=512, truncate_mode='head_tail'):
        super().__init__()
        self.data_df = pd.read_csv(csv_file)
        self.graph_dir = graph_dir
        self.max_len = max_len
        self.truncate_mode = truncate_mode
        
        self.tokenizer = RobertaTokenizer.from_pretrained(
            './pretrained_models/codebert-base', 
            local_files_only=True
        )
        
        if self.truncate_mode == 'entropy':
            # subtrain
            # with open('./data/opcode_entropy.json', 'r') as f:
            # full_data
            with open('/root/autodl-tmp/Kaggle2015/full_data/opcode_entropy.json', 'r') as f:
                self.entropy_dict = json.load(f)

    def __len__(self):
        return len(self.data_df)

    def process_sequence(self, row):
        max_ops = self.max_len - 2 
        
        # 1. CFG 引导
        if self.truncate_mode == 'cfg_guided' and 'cfg_guided_opcodes' in row:
            opcodes = str(row['cfg_guided_opcodes']).split()
            return " ".join(opcodes[:max_ops])
            
        opcodes = str(row['opcodes']).split()
        if len(opcodes) <= max_ops:
            return " ".join(opcodes)
            
        # 2. 头部截断
        if self.truncate_mode == 'head_only':
            return " ".join(opcodes[:max_ops])
            
        # 3. 头尾截断
        elif self.truncate_mode == 'head_tail':
            half = max_ops // 2
            return " ".join(opcodes[:half] + opcodes[-half:])
            
        # 4. 信息熵精简
        elif self.truncate_mode == 'entropy':
            scored_ops = [(i, op, self.entropy_dict.get(op, 5.0)) for i, op in enumerate(opcodes)]
            scored_ops.sort(key=lambda x: x[2], reverse=True)
            kept_ops = sorted(scored_ops[:max_ops], key=lambda x: x[0])
            return " ".join([x[1] for x in kept_ops])

        return str(row['opcodes'])

    def __getitem__(self, idx):
        row = self.data_df.iloc[idx]
        sample_id = row['id']
        label = row['label']
        
        processed_text = self.process_sequence(row)
        tokens = self.tokenizer(
            processed_text, padding='max_length', max_length=self.max_len,
            truncation=True, return_tensors="pt"
        )
        
        graph_path = os.path.join(self.graph_dir, f"{sample_id}.pt")
        graph_data = torch.load(graph_path, weights_only=False)
        
        # 将文本数据注入 PyG 图对象
        graph_data.input_ids = tokens['input_ids']
        graph_data.attention_mask = tokens['attention_mask']
        graph_data.y = torch.tensor([label], dtype=torch.long)

        return graph_data