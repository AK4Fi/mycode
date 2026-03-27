import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def main():
    # 路径配置
    script_dir = os.path.dirname(os.path.abspath(__file__))
    train_csv_path = os.path.join('/root/autodl-tmp/Kaggle2015/full_data', 'train.csv')
    test_csv_path = os.path.join('/root/autodl-tmp/Kaggle2015/full_data', 'test.csv')
    
    results_dir = os.path.join(script_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    print("正在加载数据并统计长度...")
    try:
        train_df = pd.read_csv(train_csv_path)
        test_df = pd.read_csv(test_csv_path)
        
        # 合并所有序列进行整体统计
        all_opcodes = pd.concat([train_df['opcodes'], test_df['opcodes']]).dropna().astype(str)
        
        # 计算每个样本的 opcode 数量（按空格分割算长度）
        lengths = all_opcodes.apply(lambda x: len(x.split()))
        
        print(f"\n==== Opcode 序列长度统计 ====")
        print(f"总样本数: {len(lengths)}")
        print(f"平均长度: {lengths.mean():.2f}")
        print(f"中位数: {lengths.median():.2f}")
        print(f"最大长度: {lengths.max()}")
        print(f"最小长度: {lengths.min()}")
        
        # 统计长度 <= 512 的比例
        threshold = 512
        covered = (lengths <= threshold).sum()
        ratio = covered / len(lengths) * 100
        print(f"\n[核心数据] 长度小于等于 {threshold} 的样本占比: {ratio:.2f}%")
        
        # ------ 画 CDF 图 ------
        plt.figure(figsize=(8, 6))
        # 排序并计算累积概率
        sorted_lengths = np.sort(lengths)
        yvals = np.arange(1, len(sorted_lengths) + 1) / len(sorted_lengths)
        
        plt.plot(sorted_lengths, yvals, color='#1f77b4', linewidth=2)
        
        # 标出 512 的参考线
        plt.axvline(x=threshold, color='red', linestyle='--', alpha=0.7, label=f'L_max = {threshold}')
        plt.axhline(y=ratio/100, color='red', linestyle='--', alpha=0.7)
        plt.plot(threshold, ratio/100, 'ro') # 红点标注
        
        # 限制X轴范围，让图表更好看（截取到99%分位数）
        p99 = np.percentile(lengths, 99)
        plt.xlim(0, max(1000, p99)) 
        plt.ylim(0, 1.05)
        
        plt.title('CDF of Opcode Sequence Lengths')
        plt.xlabel('Sequence Length (Number of Opcodes)')
        plt.ylabel('Cumulative Distribution')
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.legend()
        
        save_path = os.path.join(results_dir, 'length_cdf_plot.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n[√] CDF分布图已成功保存至: {save_path}")
        
    except Exception as e:
        print(f"运行出错: {e}")

if __name__ == "__main__":
    main()