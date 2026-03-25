# 文件位置: core/model.py
import torch
import torch.nn as nn
from transformers import RobertaModel
from torch_geometric.nn import GCNConv, global_mean_pool

_original_load = torch.load
torch.load = lambda *args, **kwargs: _original_load(*args, **{**kwargs, 'weights_only': False})

class MultiModalMalwareClassifier(nn.Module):
    def __init__(self, num_node_features=3, gcn_hidden=128, bert_hidden=768, num_classes=9,
                 modality='all', fusion='cross_attn', attn_dir='cfg2seq'):
        super().__init__()
        self.modality = modality    # 'all', 'cfg_only', 'opcode_only'
        self.fusion = fusion        # 'cross_attn', 'concat', 'dense_concat'
        self.attn_dir = attn_dir    # 'cfg2seq', 'seq2cfg'
        
        # --- 层定义 (保留原有结构) ---
        self.conv1 = GCNConv(num_node_features, gcn_hidden)
        self.conv2 = GCNConv(gcn_hidden, gcn_hidden)
        self.relu = nn.ReLU()
        self.align_layer = nn.Linear(gcn_hidden, bert_hidden)
        self.bert = RobertaModel.from_pretrained('./pretrained_models/codebert-base', local_files_only=True)
        self.cross_attention = nn.MultiheadAttention(embed_dim=bert_hidden, num_heads=8, batch_first=True, dropout=0.1)
        
        # 动态计算分类器输入维度
        if self.modality == 'all':
            clf_dim = bert_hidden * 3 if self.fusion == 'dense_concat' else bert_hidden * 2
        else:
            clf_dim = bert_hidden

        self.classifier = nn.Sequential(
            nn.Linear(clf_dim, 256), nn.BatchNorm1d(256),
            nn.ReLU(), nn.Dropout(0.4), nn.Linear(256, num_classes)
        )

    def forward(self, data):
        final_feature = None
        
        # ==========================================
        # 【终极维度修复】: 防止 PyG 把 Batch 拼接成 1D
        # 使用 .view(-1, 512) 自动推断当前 Batch Size
        # ==========================================
        b_input_ids = data.input_ids.view(-1, 512)
        b_mask = data.attention_mask.view(-1, 512)

        # 1. 提取图特征
        x = self.relu(self.conv1(data.x, data.edge_index))
        x = self.relu(self.conv2(x, data.edge_index))
        cfg_embed = global_mean_pool(x, data.batch)
        cfg_embed = self.align_layer(cfg_embed) # [B, 768]
        if self.modality == 'cfg_only': final_feature = cfg_embed

        # 2. 提取序列特征
        outputs = self.bert(input_ids=b_input_ids, attention_mask=b_mask)
        seq_states = outputs.last_hidden_state # [B, 512, 768]
        cls_embed = seq_states[:, 0, :]        # [B, 768]
        if self.modality == 'opcode_only': final_feature = cls_embed

        # 3. 特征融合消融控制
        if self.modality == 'all':
            if self.fusion == 'concat':
                final_feature = torch.cat([cls_embed, cfg_embed], dim=-1)
                
            elif self.fusion in ['cross_attn', 'dense_concat']:
                if self.attn_dir == 'cfg2seq':
                    query = cfg_embed.unsqueeze(1)
                    key = value = seq_states
                    padding_mask = (b_mask == 0) # 修正后的 2D Mask: [B, 512]
                else: # 反向消融
                    query = cls_embed.unsqueeze(1)
                    key = value = cfg_embed.unsqueeze(1)
                    padding_mask = None 

                attn_output, _ = self.cross_attention(
                    query=query, key=key, value=value, key_padding_mask=padding_mask
                )
                fused_embed = attn_output.squeeze(1)

                if self.fusion == 'cross_attn':
                    final_feature = torch.cat([cls_embed, fused_embed], dim=-1)
                elif self.fusion == 'dense_concat':
                    final_feature = torch.cat([cls_embed, cfg_embed, fused_embed], dim=-1)

        logits = self.classifier(final_feature)
        return logits, None