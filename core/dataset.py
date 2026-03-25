import os
import torch
import pandas as pd
from torch_geometric.data import Dataset
from transformers import RobertaTokenizer

class MalwareMultimodalDataset(Dataset):
    """同时加载 CFG(图) 和 Opcode(文本) 的异构数据集"""
    def __init__(self, csv_file, graph_dir, max_len=512):
        super().__init__()
        self.data_df = pd.read_csv(csv_file)
        self.graph_dir = graph_dir
        self.max_len = max_len
        self.tokenizer = RobertaTokenizer.from_pretrained('./pretrained_models/codebert-base')

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        row = self.data_df.iloc[idx]
        sample_id = row['id']
        label = row['label']
        opcode_text = str(row['opcodes'])

        # 1. 文本 Tokenize
        tokens = self.tokenizer(
            opcode_text,
            padding='max_length',
            max_length=self.max_len,
            truncation=True,
            return_tensors="pt"
        )
        
        # 2. 读取之前保存的 PyG 图数据
        graph_path = os.path.join(self.graph_dir, f"{sample_id}.pt")
        graph_data = torch.load(graph_path, weights_only=False)

        # 3. 保留张量的二维形状 [1, 512]，PyG Dataloader 拼接后会自动变成 [Batch, 512]
        graph_data.input_ids = tokens['input_ids']
        graph_data.attention_mask = tokens['attention_mask']
        graph_data.y = torch.tensor([label], dtype=torch.long)

        return graph_data