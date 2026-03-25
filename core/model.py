import torch
import torch.nn as nn
from transformers import RobertaModel
from torch_geometric.nn import GCNConv, global_mean_pool

class MultiModalMalwareClassifier(nn.Module):
    def __init__(self, num_node_features, gcn_hidden=128, bert_hidden=768, num_classes=9):
        super(MultiModalMalwareClassifier, self).__init__()
        
        # --- 模态 1：CFG 图卷积网络 (GCN) ---
        self.conv1 = GCNConv(num_node_features, gcn_hidden)
        self.conv2 = GCNConv(gcn_hidden, gcn_hidden)
        self.relu = nn.ReLU()
        self.align_layer = nn.Linear(gcn_hidden, bert_hidden)
        
        # --- 模态 2：Opcode 序列编码器 (CodeBERT) ---
        self.bert = RobertaModel.from_pretrained('./pretrained_models/codebert-base')
        
        # --- 核心创新：交叉注意力层 ---
        # 使得图宏观结构 (Query) 引导去关注关键的指令语义 (Key, Value)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=bert_hidden, 
            num_heads=8, 
            batch_first=True,
            dropout=0.1
        )
        
        # --- 改进型 MLP 分类器 ---
        self.classifier = nn.Sequential(
            nn.Linear(bert_hidden * 2, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, data):
        # 1. CFG 特征提取
        x = self.relu(self.conv1(data.x, data.edge_index))
        x = self.relu(self.conv2(x, data.edge_index))
        cfg_embed = global_mean_pool(x, data.batch) # [Batch, GCN_dim]
        cfg_query = self.align_layer(cfg_embed).unsqueeze(1) # [Batch, 1, BERT_dim]
        
        # 2. CodeBERT 序列特征提取
        outputs = self.bert(input_ids=data.input_ids, attention_mask=data.attention_mask)
        seq_states = outputs.last_hidden_state # [Batch, Seq_Len, BERT_dim]
        cls_embed = seq_states[:, 0, :]        # [Batch, BERT_dim]
        
        # 3. 交叉注意力融合
        # 过滤掉 Padding 部分，不参与注意力计算
        padding_mask = (data.attention_mask == 0)
        attn_output, attn_weights = self.cross_attention(
            query=cfg_query,
            key=seq_states,
            value=seq_states,
            key_padding_mask=padding_mask
        )
        fused_embed = attn_output.squeeze(1) # [Batch, BERT_dim]
        
        # 4. 特征拼接与分类 (残差思想)
        final_feature = torch.cat([cls_embed, fused_embed], dim=-1) # [Batch, BERT_dim * 2]
        logits = self.classifier(final_feature)
        
        return logits, attn_weights