import torch
from torch_geometric.loader import DataLoader
from core.dataset import MalwareMultimodalDataset
from core.model import MultiModalMalwareClassifier
from core.utils import calculate_metrics

def test():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_NODE_FEATURES = 3
    
    test_dataset = MalwareMultimodalDataset('./data/test.csv', './data/graphs/')
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    model = MultiModalMalwareClassifier(num_node_features=NUM_NODE_FEATURES).to(DEVICE)
    model.load_state_dict(torch.load('best_malware_model.pth'))
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch_data in test_loader:
            batch_data = batch_data.to(DEVICE)
            logits, _ = model(batch_data)
            preds = logits.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_data.y.cpu().numpy())

    acc, f1, recall = calculate_metrics(all_labels, all_preds)
    print("\n" + "="*40)
    print("最终测试集表现 (Final Test Metrics):")
    print(f"Accuracy : {acc:.4f}")
    print(f"F1 Macro : {f1:.4f}")
    print(f"Recall   : {recall:.4f}")
    print("="*40 + "\n")

if __name__ == '__main__':
    test()