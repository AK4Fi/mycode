import os
import time
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score

def main():
    # ---------------------------------------------------------
    # 0. 路径与目录初始化
    # ---------------------------------------------------------
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    results_dir = os.path.join(project_root, 'results')
    os.makedirs(results_dir, exist_ok=True)

    output_log_path = os.path.join(results_dir, 'ml_baseline_results.txt')

    print(f"==== 传统机器学习基线测试 (TF-IDF + ML) ====")
    print(f"结果将自动保存至: {output_log_path}\n")

    # ---------------------------------------------------------
    # 1. 加载数据 (路径保持你的原样)
    # ---------------------------------------------------------
    train_csv_path = os.path.join('/root/autodl-tmp/Kaggle2015/full_data', 'train.csv')
    test_csv_path = os.path.join('/root/autodl-tmp/Kaggle2015/full_data', 'test.csv')

    try:
        train_df = pd.read_csv(train_csv_path)
        test_df = pd.read_csv(test_csv_path)

        text_col = 'opcodes'
        label_col = 'label'

        X_train = train_df[text_col].fillna('').astype(str)
        y_train = train_df[label_col]
        X_test = test_df[text_col].fillna('').astype(str)
        y_test = test_df[label_col]
        print(f"[+] 数据加载成功！训练集: {len(X_train)} 条, 测试集: {len(X_test)} 条")

    except FileNotFoundError:
        print(f"\n[错误] 找不到数据文件！")
        return

    with open(output_log_path, 'w', encoding='utf-8') as f:
        f.write("==== 传统机器学习基线测试 (TF-IDF + ML) 结果 ====\n")
        f.write(f"训练集大小: {len(X_train)} | 测试集大小: {len(X_test)}\n\n")

        # ---------------------------------------------------------
        # 2. 特征工程 (TF-IDF)
        # ---------------------------------------------------------
        print("\n[+] 正在提取 TF-IDF 特征，这可能需要一两分钟...")
        start_tfidf = time.time()
        vectorizer = TfidfVectorizer(max_features=2000, ngram_range=(1, 2))
        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_test_tfidf = vectorizer.transform(X_test)

        tfidf_msg = f"TF-IDF 提取完成，耗时: {time.time() - start_tfidf:.2f} 秒。特征维度: {X_train_tfidf.shape[1]}\n"
        print(tfidf_msg)
        f.write(tfidf_msg + "\n")

        # ---------------------------------------------------------
        # 3. 定义并训练模型 (去掉了容易过拟合的随机森林)
        # ---------------------------------------------------------
        models = {
            "Logistic Regression (逻辑回归)": LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1),
            "Naive Bayes (朴素贝叶斯)": MultinomialNB(),
            "Decision Tree (决策树)": DecisionTreeClassifier(random_state=42)
        }

        for name, model in models.items():
            split_line = "=" * 50
            print(f"\n{split_line}")
            print(f"正在训练: {name} ...")
            start_train = time.time()

            # 训练模型
            model.fit(X_train_tfidf, y_train)
            train_time = time.time() - start_train

            # 预测与评估
            y_pred = model.predict(X_test_tfidf)
            acc = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, digits=4)

            # 控制台打印
            print(f"[{name}] 训练耗时: {train_time:.2f} 秒")
            print(f"[{name}] 准确率 (Accuracy): {acc:.4f}")
            print("详细分类报告:\n", report)

            # 写入日志文件
            f.write(f"{split_line}\n")
            f.write(f"模型名称: {name}\n")
            f.write(f"训练耗时: {train_time:.2f} 秒\n")
            f.write(f"准确率 (Accuracy): {acc:.4f}\n")
            f.write("分类报告:\n")
            f.write(report + "\n")

    print(f"\n[√] 所有测试已完成，完整结果已成功保存至: {output_log_path}")

if __name__ == "__main__":
    main()
