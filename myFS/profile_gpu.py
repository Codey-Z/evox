"""
GPU性能分析脚本 - 检查GPU使用情况和性能瓶颈
"""
import torch
import time
import pandas as pd
import numpy as np
from FS import FS
from GPUKNN import TensorKNNClassifier

print("=" * 60)
print("GPU性能分析")
print("=" * 60)

# 检查CUDA可用性
print(f"\nCUDA可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA设备: {torch.cuda.get_device_name(0)}")
    print(f"当前设备: {torch.cuda.current_device()}")
    print(f"默认设备: {torch.get_default_device()}")

# 加载数据
csv_path = "csvdata/9_Tumors.csv"
data = pd.read_csv(csv_path).values
X = data[:, :-1].astype(float)
y = data[:, -1].astype(int)

print(f"\n数据集信息:")
print(f"样本数: {X.shape[0]}, 特征数: {X.shape[1]}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 创建问题实例
problem = FS(device=device)
problem.set_data(X, y)

print(f"\n数据在GPU上: {problem.X.device}")
print(f"标签在GPU上: {problem.y.device}")

# 测试性能
pop_size = 50
dim = problem.n_features
pop = torch.rand(pop_size, dim, device=device)

print(f"\n种群在GPU上: {pop.device}")

# 测试1: 不使用交叉验证
print("\n" + "=" * 60)
print("测试1: 不使用交叉验证 (use_cv=False)")
print("=" * 60)

if torch.cuda.is_available():
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

start = time.time()
fitness = problem.evaluate(pop, batch_size=256, use_cv=False, n_splits=5)
if torch.cuda.is_available():
    torch.cuda.synchronize()
elapsed = time.time() - start

print(f"耗时: {elapsed:.2f}秒")
print(f"Fitness在GPU上: {fitness.device}")
if torch.cuda.is_available():
    peak_memory = torch.cuda.max_memory_allocated() / 1024**2
    print(f"峰值GPU内存: {peak_memory:.2f} MB")

# 测试2: 使用5折交叉验证
print("\n" + "=" * 60)
print("测试2: 使用5折交叉验证 (use_cv=True)")
print("=" * 60)

if torch.cuda.is_available():
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

start = time.time()
fitness = problem.evaluate(pop, batch_size=256, use_cv=True, n_splits=5)
if torch.cuda.is_available():
    torch.cuda.synchronize()
elapsed = time.time() - start

print(f"耗时: {elapsed:.2f}秒")
print(f"Fitness在GPU上: {fitness.device}")
if torch.cuda.is_available():
    peak_memory = torch.cuda.max_memory_allocated() / 1024**2
    print(f"峰值GPU内存: {peak_memory:.2f} MB")

# 测试3: 直接测试KNN性能
print("\n" + "=" * 60)
print("测试3: KNN批量预测性能")
print("=" * 60)

knn = TensorKNNClassifier(k=1, device=device)

# 创建测试数据
batch_size = 50
X_tensor = torch.tensor(X, dtype=torch.float32, device=device)
y_tensor = torch.tensor(y, dtype=torch.long, device=device)

# 生成随机特征掩码
feature_masks = torch.rand(batch_size, dim, device=device) > 0.6
all_feature_subsets = X_tensor.unsqueeze(0) * feature_masks.unsqueeze(1).float()

print(f"特征子集形状: {all_feature_subsets.shape}")
print(f"特征子集在GPU上: {all_feature_subsets.device}")

if torch.cuda.is_available():
    torch.cuda.synchronize()

start = time.time()
acc = knn.cross_validate(all_feature_subsets, y_tensor, n_splits=5)
if torch.cuda.is_available():
    torch.cuda.synchronize()
elapsed = time.time() - start

print(f"耗时: {elapsed:.2f}秒")
print(f"准确率形状: {acc.shape}")
print(f"准确率在GPU上: {acc.device}")

# 测试4: 检查是否有CPU-GPU数据传输
print("\n" + "=" * 60)
print("测试4: 检查数据传输瓶颈")
print("=" * 60)

# 监控GPU利用率
if torch.cuda.is_available():
    print("\n提示: 请在另一个终端运行 'nvidia-smi -l 1' 监控GPU使用率")
    print("如果GPU利用率很低，说明存在CPU-GPU传输瓶颈或计算未充分并行")
    
    input("\n按回车开始长时间测试...")
    
    torch.cuda.synchronize()
    start = time.time()
    
    for i in range(5):
        print(f"迭代 {i+1}/5...")
        fitness = problem.evaluate(pop, batch_size=256, use_cv=True, n_splits=5)
    
    torch.cuda.synchronize()
    elapsed = time.time() - start
    
    print(f"\n总耗时: {elapsed:.2f}秒")
    print(f"平均每次: {elapsed/5:.2f}秒")

print("\n" + "=" * 60)
print("性能分析完成")
print("=" * 60)
