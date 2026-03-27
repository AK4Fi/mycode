import pandas as pd
import random
import os
import time

def inject_nops(sequence, ratio):
    """
    在操作码序列中随机插入指定比例的 nop 指令
    """
    if not isinstance(sequence, str) or not sequence.strip():
        return sequence
    
    opcodes = sequence.split()
    num_to_insert = int(len(opcodes) * ratio)
    
    if num_to_insert == 0:
        return sequence
        
    # 随机选择插入位置并插入 'nop'
    for _ in range(num_to_insert):
        insert_pos = random.randint(0, len(opcodes))
        opcodes.insert(insert_pos, "nop")
        
    return " ".join(opcodes)

def main():
    print("==== 恶意代码黑盒对抗攻击：NOP 垃圾指令注入 ====")
    
    # 原始数据路径 (保持与你之前基线代码一致)
    data_dir = '/root/autodl-tmp/Kaggle2015/full_data'
    original_test_path = os.path.join(data_dir, 'test.csv')
    
    if not os.path.exists(original_test_path):
        print(f"[错误] 找不到原始测试集文件: {original_test_path}")
        return

    print(f"正在加载原始测试集: {original_test_path}")
    df_test = pd.read_csv(original_test_path)
    
    # 设置攻击强度 (注入 10%, 20%, 30% 的 NOP)
    attack_ratios = [0.10, 0.20, 0.30]
    
    for ratio in attack_ratios:
        print(f"\n[+] 正在生成 {int(ratio*100)}% 注入强度的对抗测试集...")
        start_time = time.time()
        
        # 复制一份 dataframe
        df_attacked = df_test.copy()
        
        # 对 opcodes 列应用 NOP 注入
        df_attacked['opcodes'] = df_attacked['opcodes'].apply(lambda x: inject_nops(x, ratio))
        
        # 生成新文件的保存路径
        save_name = f'test_nop_{int(ratio*100)}.csv'
        save_path = os.path.join(data_dir, save_name)
        
        # 保存到 CSV
        df_attacked.to_csv(save_path, index=False)
        print(f"    - 完成！耗时: {time.time() - start_time:.2f} 秒")
        print(f"    - 文件已保存至: {save_path}")

    print("\n[√] 所有强度的对抗测试集生成完毕！")
    print("接下来，你可以修改你的测试脚本，读取这些新文件进行抗混淆能力测试。")

if __name__ == "__main__":
    # 固定随机种子保证可复现
    random.seed(42)
    main()
