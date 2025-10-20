"""
简化测试脚本：测试单个数据集的单个fold
用于验证代码逻辑是否正确
"""
import torch
from FS import FS
import pandas as pd
import numpy as np
from pathlib import Path
from evox.algorithms import PSO
from evox.workflows import StdWorkflow, EvalMonitor
from GPUKNN import TensorKNNClassifier
from sklearn.model_selection import train_test_split

# 设置设备
torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"使用设备: {device}")

# 加载单个数据集进行测试
csv_path = "csvdata/9_Tumors.csv"
data = pd.read_csv(csv_path).values
X = data[:, :-1].astype(float)
y = data[:, -1].astype(int)

print(f"数据集: {Path(csv_path).stem}")
print(f"样本数: {X.shape[0]}, 特征数: {X.shape[1]}, 类别数: {len(np.unique(y))}")

# 简单划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"训练集: {X_train.shape[0]} 样本")
print(f"测试集: {X_test.shape[0]} 样本")

# 创建FS问题实例
problem = FS(device=device)
problem.set_data(X_train, y_train)

dim = problem.n_features

# PSO配置
pop_size = 50  # 减小种群以加快测试
n_iterations = 20  # 减少迭代次数

print(f"\nPSO配置: 种群={pop_size}, 迭代={n_iterations}")

# 初始化PSO
algorithm = PSO(
    pop_size=pop_size,
    lb=0 * torch.ones(dim),
    ub=1 * torch.ones(dim)
)

# 创建工作流
monitor = EvalMonitor()
workflow = StdWorkflow(algorithm, problem, monitor)

# 运行PSO
print("\n开始PSO优化...")
workflow.init_step()

for i in range(n_iterations):
    workflow.step()
    if (i + 1) % 5 == 0:
        current_best = monitor.get_best_fitness()
        print(f"迭代 {i+1}/{n_iterations}, 当前最优: {float(current_best):.4f}")

# 获取最优解
best_fitness = monitor.get_best_fitness()
best_solution = monitor.get_best_solution()

print(f"\nPSO完成!")
print(f"最优适应度: {float(best_fitness):.4f}")

# 转换为特征掩码
threshold = 0.6
feature_mask = (best_solution.cpu().numpy() > threshold).astype(int)
n_selected = np.sum(feature_mask)

print(f"选择特征数: {n_selected}/{dim}")
print(f"特征选择率: {n_selected/dim*100:.1f}%")

# 在测试集上评估
if n_selected == 0:
    print("警告: 未选择任何特征，使用全部特征")
    feature_mask = np.ones_like(feature_mask)
    n_selected = len(feature_mask)

# 应用特征掩码
X_train_selected = X_train[:, feature_mask.astype(bool)]
X_test_selected = X_test[:, feature_mask.astype(bool)]

# 转换为Tensor
X_train_tensor = torch.tensor(X_train_selected, dtype=torch.float32, device=device)
X_test_tensor = torch.tensor(X_test_selected, dtype=torch.float32, device=device)
y_train_tensor = torch.tensor(y_train, dtype=torch.long, device=device)
y_test_tensor = torch.tensor(y_test, dtype=torch.long, device=device)

# 使用KNN评估
print("\n在测试集上评估...")
knn = TensorKNNClassifier(k=1, device=device)

X_train_batch = X_train_tensor.unsqueeze(0)
X_test_batch = X_test_tensor.unsqueeze(0)

preds = knn.predict_batched_subsets(X_train_batch, y_train_tensor, X_test_batch)
test_acc = knn.accuracy(preds, y_test_tensor)

print(f"\n最终结果:")
print(f"测试集准确率: {float(test_acc[0]):.4f}")
print(f"选择的特征: {n_selected}/{dim} ({n_selected/dim*100:.1f}%)")
