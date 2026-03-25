# 文件位置: ./train_ablation.py
import time
import torch
import torch.nn as nn
import argparse
import os
from tqdm import tqdm
from torch_geometric.loader import DataLoader
from core.dataset import MalwareMultimodalDataset
from core.model import MultiModalMalwareClassifier
from core.utils import FocalLoss, calculate_metrics

def run_experiment(args):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"消融实验名: {args.exp_name}")
    print(f"模态: {args.modality} | 融合: {args.fusion} | 方向: {args.attn_dir}")
    print(f"策略: {args.truncate} | 损失: {args.loss_fn}")
    print(f"{'='*60}\n")

    train_dataset = MalwareMultimodalDataset('./data/train.csv', './data/graphs/', truncate_mode=args.truncate)
    val_dataset = MalwareMultimodalDataset('./data/val.csv', './data/graphs/', truncate_mode=args.truncate)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    model = MultiModalMalwareClassifier(
        modality=args.modality, fusion=args.fusion, attn_dir=args.attn_dir
    ).to(DEVICE)

    # 统计模型大小
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📦 模型参数量: {total_params/1e6:.2f} M")

    criterion = FocalLoss(gamma=2.0) if args.loss_fn == 'focal' else nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    
    best_f1 = 0.0
    for epoch in range(args.epochs):
        model.train()
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch_data in loop:
            optimizer.zero_grad()
            logits, _ = model(batch_data.to(DEVICE))
            loss = criterion(logits, batch_data.y)
            loss.backward()
            optimizer.step()
            loop.set_postfix(loss=loss.item())

        model.eval()
        all_val_preds, all_val_labels = [], []
        if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
        start_time = time.time()
        
        with torch.no_grad():
            for batch_data in val_loader:
                batch_data = batch_data.to(DEVICE)
                logits, _ = model(batch_data)
                preds = logits.argmax(dim=1)
                all_val_preds.extend(preds.cpu().numpy())
                all_val_labels.extend(batch_data.y.cpu().numpy())
                
        end_time = time.time()
        
        # 统计运行效率
        ms_per_sample = ((end_time - start_time) / len(all_val_labels)) * 1000
        max_vram = torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0
        val_acc, val_f1, val_recall = calculate_metrics(all_val_labels, all_val_preds)
        
        print(f"\n👉 [Epoch {epoch+1}] Acc: {val_acc:.4f} | F1: {val_f1:.4f} | Recall: {val_recall:.4f}")
        print(f"   ⏱️ 推理: {ms_per_sample:.2f} ms/样本 | 💾 显存峰值: {max_vram:.2f} MB")

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), f"weights/exp_{args.exp_name}.pth")
            
    print(f"✅ 实验 {args.exp_name} 结束，历史最佳 F1: {best_f1:.4f}\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, required=True)
    parser.add_argument('--modality', type=str, default='all', choices=['all', 'cfg_only', 'opcode_only'])
    parser.add_argument('--fusion', type=str, default='cross_attn', choices=['cross_attn', 'concat', 'dense_concat'])
    parser.add_argument('--attn_dir', type=str, default='cfg2seq', choices=['cfg2seq', 'seq2cfg'])
    parser.add_argument('--truncate', type=str, default='entropy', choices=['head_only', 'head_tail', 'entropy', 'cfg_guided'])
    parser.add_argument('--loss_fn', type=str, default='focal', choices=['focal', 'ce'])
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=5)
    args = parser.parse_args()
    
    os.makedirs('weights', exist_ok=True)
    run_experiment(args)