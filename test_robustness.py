import os
import torch
import argparse
from sklearn.metrics import accuracy_score, f1_score, recall_score

# 导入真实的类名 MalwareMultimodalDataset
from core.dataset import MalwareMultimodalDataset
from core.model import MultiModalMalwareClassifier
from torch_geometric.loader import DataLoader

def parse_args():
    parser = argparse.ArgumentParser(description="恶意代码抗混淆鲁棒性测试")
    parser.add_argument('--checkpoint', type=str, required=True, help='训练好的模型权重路径 (.pth)')
    parser.add_argument('--test_csv', type=str, required=True, help='脏测试集路径 (如 test_nop_30.csv)')
    
    # 增加 graph_dir 参数（指向图数据存放的目录）
    parser.add_argument('--graph_dir', type=str, default='/root/autodl-tmp/Kaggle2015/full_data/graphs', help='图特征文件夹')
    
    # 模型架构参数
    parser.add_argument('--truncate', type=str, default='cfg_guided', choices=['head_only', 'head_tail', 'entropy', 'cfg_guided'])
    parser.add_argument('--modality', type=str, default='all', choices=['all', 'cfg_only', 'opcode_only'])
    parser.add_argument('--fusion', type=str, default='cross_attn', choices=['concat', 'dense_concat', 'cross_attn'])
    parser.add_argument('--attn_dir', type=str, default='cfg2seq', choices=['cfg2seq', 'seq2cfg'])
    parser.add_argument('--batch_size', type=int, default=16)
    
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"\n{'='*50}")
    print(f"[*] 正在加载对抗测试集: {args.test_csv}")
    print(f"[*] 截断策略: {args.truncate} | 模态: {args.modality}")
    print(f"{'='*50}")

    # 1. 初始化对抗测试集 Dataset 与 DataLoader
    test_dataset = MalwareMultimodalDataset(
        csv_file=args.test_csv, 
        graph_dir=args.graph_dir,
        max_len=512,
        truncate_mode=args.truncate
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # 2. 初始化模型架构
    model = MultiModalMalwareClassifier(
        modality=args.modality,
        fusion=args.fusion,
        attn_dir=args.attn_dir
    ).to(device)
    
    # 3. 加载训练好的权重
    if not os.path.exists(args.checkpoint):
        print(f"\n[严重错误] 找不到模型权重文件: {args.checkpoint}")
        return
        
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    print(f"[+] 成功加载模型权重: {args.checkpoint}\n")

    # 4. 开始评估
    model.eval()
    all_preds = []
    all_labels = []
    
    print("正在进行推理预测，请稍候...")
    with torch.no_grad():
        for batch in test_loader:
            if hasattr(batch, 'y'):  # 适配 PyG Data 对象
                batch = batch.to(device)
                labels = batch.y
                # 【修复核心点】：正确解包模型返回的元组
                logits, _ = model(batch)
            else: 
                inputs, labels = batch[0].to(device), batch[1].to(device)
                logits, _ = model(inputs)

            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 5. 计算并打印鲁棒性指标
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')
    
    print(f"\n{'='*20} 对抗攻击测试结果 {'='*20}")
    print(f"测试集: {os.path.basename(args.test_csv)}")
    print(f"使用模型: {os.path.basename(args.checkpoint)}")
    print(f"Accuracy    : {acc:.4f}")
    print(f"Macro-F1    : {f1:.4f}  <--- (把这个数字填进论文)")
    print(f"Macro-Recall: {recall:.4f}")
    print(f"{'='*58}\n")

if __name__ == '__main__':
    main()
