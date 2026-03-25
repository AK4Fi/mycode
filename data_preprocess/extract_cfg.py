import os
import re
import torch
import pandas as pd
import networkx as nx
import concurrent.futures
from torch_geometric.data import Data
from tqdm import tqdm

# 将正则表达式提取为全局变量，避免在每个函数调用中重复编译，提升多进程效率
JUMP_PATTERN = re.compile(r"\s([a-fA-F0-9]{2}\s)+\s*(j[a-z]+|call)\s+([a-zA-Z0-9_\.]+)")
LABEL_PATTERN = re.compile(r"^\s*([a-zA-Z0-9_\.]+):")

def extract_heuristic_cfg(asm_filepath):
    """启发式构建基本块与控制流图，生成 PyG Data 对象"""
    G = nx.DiGraph()
    current_block = "entry"
    G.add_node(current_block, weight=1)

    try:
        with open(asm_filepath, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if not line.startswith(".text"):
                    continue

                # 遇到跳转目标(Label)
                label_match = re.search(LABEL_PATTERN, line)
                if label_match:
                    new_block = label_match.group(1)
                    # 【修复点 1】：在建边之前，确保目标节点存在且初始化了 weight
                    if not G.has_node(new_block):
                        G.add_node(new_block, weight=0)

                    G.add_edge(current_block, new_block)
                    current_block = new_block

                # 【修复点 2】：安全地增加当前块的指令计数，防止隐式节点没有 weight 键
                G.nodes[current_block]["weight"] = (
                    G.nodes[current_block].get("weight", 0) + 1
                )

                # 遇到跳转指令
                jmp_match = re.search(JUMP_PATTERN, line)
                if jmp_match:
                    target_block = jmp_match.group(3)
                    # 【修复点 3】：同样确保 target_block 被安全初始化
                    if not G.has_node(target_block):
                        G.add_node(target_block, weight=0)

                    G.add_edge(current_block, target_block)
                    current_block = f"block_{G.number_of_nodes()}"
                    G.add_node(current_block, weight=0)

    except Exception as e:
        # 忽略无法解析的脏数据
        pass

    # 将 NetworkX 转换为 PyTorch Geometric 的 Data 对象
    node_mapping = {node: i for i, node in enumerate(G.nodes())}
    edge_index = (
        torch.tensor(
            [[node_mapping[u], node_mapping[v]] for u, v in G.edges()], dtype=torch.long
        )
        .t()
        .contiguous()
    )

    # 节点特征：目前使用 [入度, 出度, 块内指令数] 作为初始特征 (3维度)
    x = []
    for node in G.nodes():
        in_deg = G.in_degree(node)
        out_deg = G.out_degree(node)
        # 【修复点 4】：使用 .get 安全提取特征，兜底为 0
        weight = G.nodes[node].get("weight", 0)
        x.append([in_deg, out_deg, weight])

    x = torch.tensor(x, dtype=torch.float)

    # 【修复点 5】：防止孤立节点导致的 edge_index 维度不匹配问题
    if x.size(0) == 0:
        x = torch.tensor([[0.0, 0.0, 1.0]])
        edge_index = torch.empty((2, 0), dtype=torch.long)
    elif edge_index.numel() == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    return Data(x=x, edge_index=edge_index)

def process_single_graph(args):
    """供多进程调用的单文件图构建 worker"""
    file_id, asm_dir, graph_out_dir = args
    asm_path = os.path.join(asm_dir, f"{file_id}.asm")
    out_path = os.path.join(graph_out_dir, f"{file_id}.pt")
    
    # 如果该文件对应的图已经存在，可以选择跳过（断点续传），这里根据你的原逻辑选择覆盖
    # if os.path.exists(out_path):
    #     return True
        
    if os.path.exists(asm_path):
        graph_data = extract_heuristic_cfg(asm_path)
        torch.save(graph_data, out_path)
        return True
    return False

def main():
    # 请确保这里是你电脑上 .asm 文件的原始路径
    # asm_dir = "/root/autodl-tmp/kaggle2015-sample/subtrain"
    # csv_files = ["../data/train.csv", "../data/val.csv", "../data/test.csv"]
    # graph_out_dir = "../data/graphs"

    # full_data
    asm_dir = "/root/autodl-tmp/Kaggle2015/train"
    csv_files = [
        "/root/autodl-tmp/Kaggle2015/full_data/train.csv",
        "/root/autodl-tmp/Kaggle2015/full_data/val.csv",
        "/root/autodl-tmp/Kaggle2015/full_data/test.csv",
    ]
    graph_out_dir = "/root/autodl-tmp/Kaggle2015/full_data/graphs"
    os.makedirs(graph_out_dir, exist_ok=True)

    # 固定 32 进程
    num_workers = 32
    print(f"开始构建 CFG 图数据，启用 {num_workers} 个工作进程...")
    
    for csv_file in csv_files:
        if not os.path.exists(csv_file):
            print(f"未找到 {csv_file}，请先运行 1_extract_opcode.py")
            continue

        df = pd.read_csv(csv_file)
        # 准备任务参数
        tasks = [(file_id, asm_dir, graph_out_dir) for file_id in df["id"]]
        
        # 使用多进程池处理单个 CSV 文件内的数据
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(process_single_graph, task): task for task in tasks}
            
            # 使用 tqdm 监控当前 CSV 的处理进度
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc=f"Processing {os.path.basename(csv_file)}"):
                # 可以通过 future.result() 获取子进程返回值或捕获异常
                try:
                    future.result()
                except Exception as e:
                    print(f"处理任务时发生异常: {e}")

    print("CFG 图数据全部提取完成！")

if __name__ == "__main__":
    main()