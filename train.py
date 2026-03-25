import torch
from tqdm import tqdm
from torch_geometric.loader import DataLoader
from core.dataset import MalwareMultimodalDataset
from core.model import MultiModalMalwareClassifier
from core.utils import FocalLoss, calculate_metrics

def train():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    # 超参数配置
    EPOCHS = 10
    BATCH_SIZE = 8
    LEARNING_RATE = 1e-5
    NUM_NODE_FEATURES = 3  # 在 extract_cfg.py 中我们提取了 3 维节点特征
    
    # 建立 DataLoader (注意这里必须用 PyG 的 DataLoader，不能用标准的 DataLoader)
    train_dataset = MalwareMultimodalDataset('./data/train.csv', './data/graphs/')
    val_dataset = MalwareMultimodalDataset('./data/val.csv', './data/graphs/')
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = MultiModalMalwareClassifier(num_node_features=NUM_NODE_FEATURES).to(DEVICE)
    criterion = FocalLoss(gamma=2.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    best_f1 = 0.0

    for epoch in range(EPOCHS):
        # ---------- Train ----------
        model.train()
        train_loss = 0
        all_preds, all_labels = [], []
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
        for batch_data in loop:
            batch_data = batch_data.to(DEVICE)
            optimizer.zero_grad()
            
            logits, _ = model(batch_data)
            loss = criterion(logits, batch_data.y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_data.y.cpu().numpy())
            loop.set_postfix(loss=loss.item())

        train_acc, train_f1, train_recall = calculate_metrics(all_labels, all_preds)

        # ---------- Validation ----------
        model.eval()
        val_loss = 0
        all_val_preds, all_val_labels = [], []
        
        with torch.no_grad():
            for batch_data in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]"):
                batch_data = batch_data.to(DEVICE)
                logits, _ = model(batch_data)
                loss = criterion(logits, batch_data.y)
                
                val_loss += loss.item()
                preds = logits.argmax(dim=1)
                all_val_preds.extend(preds.cpu().numpy())
                all_val_labels.extend(batch_data.y.cpu().numpy())

        val_acc, val_f1, val_recall = calculate_metrics(all_val_labels, all_val_preds)
        
        print(f"\n[Result] Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f} | Val Recall: {val_recall:.4f}")

        # 保存最优模型
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), 'best_malware_model.pth')
            print(">>> Saved New Best Model!")

if __name__ == '__main__':
    train()