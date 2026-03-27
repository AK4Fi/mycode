import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
from torch_geometric.loader import DataLoader
from core.dataset import MalwareMultimodalDataset
from core.model import MultiModalMalwareClassifier
import os

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FAMILY_NAMES = [
    "Ramnit",
    "Lollipop",
    "Kelihos_v3",
    "Vundo",
    "Simda",
    "Tracur",
    "Kelihos_v1",
    "Obfuscator",
    "Gamarue",
]


def load_model(exp_name, modality="all", fusion="cross_attn"):
    model = MultiModalMalwareClassifier(
        modality=modality, fusion=fusion, attn_dir="cfg2seq"
    ).to(DEVICE)
    weight_path = f"weights/exp_{exp_name}.pth"
    if os.path.exists(weight_path):
        model.load_state_dict(torch.load(weight_path, map_location=DEVICE))
    return model


def plot_confusion_matrix(model, dataloader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            logits, _ = model(batch.to(DEVICE))
            all_preds.extend(logits.argmax(dim=1).cpu().numpy())
            all_labels.extend(batch.y.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=FAMILY_NAMES,
        yticklabels=FAMILY_NAMES,
    )
    plt.title("Confusion Matrix (Multi-modal CFG-Guided)")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=300)


def plot_tsne(model_single, model_multi, dataloader):
    def get_features(m):
        m.eval()
        feats, labels = [], []
        hook = m.classifier[2].register_forward_hook(
            lambda module, inp, out: feats.append(out.cpu().numpy())
        )
        with torch.no_grad():
            for batch in dataloader:
                _ = m(batch.to(DEVICE))
                labels.extend(batch.y.cpu().numpy())
        hook.remove()
        return np.concatenate(feats, axis=0), np.array(labels)

    feat_s, labels = get_features(model_single)
    feat_m, _ = get_features(model_multi)

    tsne = TSNE(n_components=2, random_state=42)
    tsne_s = tsne.fit_transform(feat_s)
    tsne_m = tsne.fit_transform(feat_m)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    palette = sns.color_palette("tab10", 9)

    sns.scatterplot(
        x=tsne_s[:, 0],
        y=tsne_s[:, 1],
        hue=labels,
        palette=palette,
        ax=ax1,
        legend=False,
        s=40,
    )
    ax1.set_title("Fig A: Single-Modal (Opcode Only) Feature Space")

    sns.scatterplot(
        x=tsne_m[:, 0], y=tsne_m[:, 1], hue=labels, palette=palette, ax=ax2, s=40
    )
    ax2.set_title("Fig B: Multi-Modal (Ours) Feature Space")
    ax2.legend(
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        title="Families",
        labels=FAMILY_NAMES,
    )

    plt.tight_layout()
    plt.savefig("tsne_comparison.png", dpi=300)


def plot_attention_heatmap(model, dataset):
    model.eval()
    sample = dataset[0].to(DEVICE)

    b_input_ids = sample.input_ids.view(1, -1)
    b_mask = sample.attention_mask.view(1, -1)

    x = model.relu(model.conv1(sample.x, sample.edge_index))
    x = model.relu(model.conv2(x, sample.edge_index))
    from torch_geometric.nn import global_mean_pool

    cfg_embed = global_mean_pool(
        x, torch.zeros(sample.x.size(0), dtype=torch.long).to(DEVICE)
    )
    cfg_embed = model.align_layer(cfg_embed)

    outputs = model.bert(input_ids=b_input_ids, attention_mask=b_mask)
    seq_states = outputs.last_hidden_state

    attn_output, attn_weights = model.cross_attention(
        query=cfg_embed.unsqueeze(1),
        key=seq_states,
        value=seq_states,
        key_padding_mask=(b_mask == 0),
    )

    tokens = dataset.tokenizer.convert_ids_to_tokens(b_input_ids[0].cpu().numpy())
    weights = attn_weights[0, 0, :].detach().cpu().numpy()

    weights = (weights - weights.min()) / (weights.max() - weights.min() + 1e-9)

    valid_len = b_mask[0].sum().item()
    plot_tokens = tokens[1 : min(35, valid_len)]  # 跳过开头 <s>
    plot_weights = weights[1 : min(35, valid_len)]

    plt.figure(figsize=(15, 2))
    sns.heatmap(
        [plot_weights], cmap="Reds", xticklabels=plot_tokens, yticklabels=["Attention"]
    )
    plt.title("Cross-Attention Heatmap: CFG-guided Opcode Focus")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("attention_heatmap.png", dpi=300)


if __name__ == "__main__":
    test_dataset = MalwareMultimodalDataset(
        "/root/autodl-tmp/Kaggle2015/full_data/test.csv",
        "/root/autodl-tmp/Kaggle2015/full_data/graphs/",
        truncate_mode="cfg_guided",
    )
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    best_multi = load_model("trunc_cfg_guided", modality="all")
    best_single = load_model("mod_opc_only", modality="opcode_only")

    plot_confusion_matrix(best_multi, test_loader)
    plot_tsne(best_single, best_multi, test_loader)
    plot_attention_heatmap(best_multi, test_dataset)
    print("done")
