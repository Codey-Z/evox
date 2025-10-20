"""
性能对比测试：原版 vs 优化版
"""
import torch
import time
import pandas as pd
import numpy as np
from pathlib import Path

# 导入原版
from FS import FS
from GPUKNN import TensorKNNClassifier

# 导入优化版
from FS_optimized import FS_Optimized
from GPUKNN_optimized import TensorKNNClassifier_Optimized

print("=" * 80)
print("性能对比测试：原版 vs 优化版")
print("=" * 80)

# 设备检查
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n使用设备: {device}")

if torch.cuda.is_available():
    print(f"GPU型号: {torch.cuda.get_device_name(0)}")
    print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

# 加载数据
csv_path = "csvdata/9_Tumors.csv"
data = pd.read_csv(csv_path).values
X = data[:, :-1].astype(float)
y = data[:, -1].astype(int)

print(f"\n数据集信息:")
print(f"样本数: {X.shape[0]}, 特征数: {X.shape[1]}, 类别数: {len(np.unique(y))}")

# 测试参数
pop_size = 100
n_iterations = 3
batch_size = 256

print(f"\n测试参数:")
print(f"种群大小: {pop_size}")
print(f"测试迭代: {n_iterations}")
print(f"批次大小: {batch_size}")

# ============================================================================
# 测试1: KNN性能对比
# ============================================================================
print("\n" + "=" * 80)
print("测试1: KNN分类器性能对比")
print("=" * 80)

# 准备测试数据
X_tensor = torch.tensor(X, dtype=torch.float32, device=device)
y_tensor = torch.tensor(y, dtype=torch.long, device=device)

# 生成随机特征掩码
feature_masks = torch.rand(pop_size, X.shape[1], device=device) > 0.6
all_feature_subsets = X_tensor.unsqueeze(0) * feature_masks.unsqueeze(1).float()

print(f"\n测试数据形状: {all_feature_subsets.shape}")

# 原版KNN
print("\n原版KNN:")
knn_original = TensorKNNClassifier(k=1, device=device)

if torch.cuda.is_available():
    torch.cuda.synchronize()
start = time.time()

for i in range(n_iterations):
    acc = knn_original.cross_validate(all_feature_subsets, y_tensor, n_splits=5)

if torch.cuda.is_available():
    torch.cuda.synchronize()
elapsed_original = time.time() - start

print(f"总耗时: {elapsed_original:.3f}秒")
print(f"平均每次: {elapsed_original/n_iterations:.3f}秒")
print(f"平均准确率: {acc.mean().item():.4f}")

# 优化版KNN
print("\n优化版KNN:")
knn_optimized = TensorKNNClassifier_Optimized(k=1, device=device)

if torch.cuda.is_available():
    torch.cuda.synchronize()
start = time.time()

for i in range(n_iterations):
    acc = knn_optimized.cross_validate(all_feature_subsets, y_tensor, n_splits=5)

if torch.cuda.is_available():
    torch.cuda.synchronize()
elapsed_optimized = time.time() - start

print(f"总耗时: {elapsed_optimized:.3f}秒")
print(f"平均每次: {elapsed_optimized/n_iterations:.3f}秒")
print(f"平均准确率: {acc.mean().item():.4f}")

speedup_knn = elapsed_original / elapsed_optimized
print(f"\n加速比: {speedup_knn:.2f}x")

# ============================================================================
# 测试2: FS Problem评估性能对比
# ============================================================================
print("\n" + "=" * 80)
print("测试2: FS Problem评估性能对比")
print("=" * 80)

# 生成测试种群
pop = torch.rand(pop_size, X.shape[1], device=device)

# 原版FS
print("\n原版FS (use_cv=True):")
problem_original = FS(device=device)
problem_original.set_data(X, y)

if torch.cuda.is_available():
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

start = time.time()
for i in range(n_iterations):
    fitness = problem_original.evaluate(pop, batch_size=batch_size, use_cv=True, n_splits=5)

if torch.cuda.is_available():
    torch.cuda.synchronize()
elapsed_original_fs = time.time() - start

print(f"总耗时: {elapsed_original_fs:.3f}秒")
print(f"平均每次: {elapsed_original_fs/n_iterations:.3f}秒")
print(f"平均fitness: {fitness.mean().item():.4f}")

if torch.cuda.is_available():
    peak_mem = torch.cuda.max_memory_allocated() / 1024**2
    print(f"峰值GPU内存: {peak_mem:.2f} MB")

# 优化版FS
print("\n优化版FS (use_cv=True):")
problem_optimized = FS_Optimized(device=device)
problem_optimized.set_data(X, y)

if torch.cuda.is_available():
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

start = time.time()
for i in range(n_iterations):
    fitness = problem_optimized.evaluate(pop, batch_size=batch_size, use_cv=True, n_splits=5)

if torch.cuda.is_available():
    torch.cuda.synchronize()
elapsed_optimized_fs = time.time() - start

print(f"总耗时: {elapsed_optimized_fs:.3f}秒")
print(f"平均每次: {elapsed_optimized_fs/n_iterations:.3f}秒")
print(f"平均fitness: {fitness.mean().item():.4f}")

if torch.cuda.is_available():
    peak_mem = torch.cuda.max_memory_allocated() / 1024**2
    print(f"峰值GPU内存: {peak_mem:.2f} MB")

speedup_fs = elapsed_original_fs / elapsed_optimized_fs
print(f"\n加速比: {speedup_fs:.2f}x")

# ============================================================================
# 测试3: 快速评估模式（不使用交叉验证）
# ============================================================================
print("\n" + "=" * 80)
print("测试3: 快速评估模式 (use_cv=False)")
print("=" * 80)

# 原版FS - 快速模式
print("\n原版FS (use_cv=False):")
if torch.cuda.is_available():
    torch.cuda.synchronize()

start = time.time()
for i in range(n_iterations):
    fitness = problem_original.evaluate(pop, batch_size=batch_size, use_cv=False)

if torch.cuda.is_available():
    torch.cuda.synchronize()
elapsed_original_fast = time.time() - start

print(f"总耗时: {elapsed_original_fast:.3f}秒")
print(f"平均每次: {elapsed_original_fast/n_iterations:.3f}秒")

# 优化版FS - 快速模式
print("\n优化版FS (use_cv=False):")
if torch.cuda.is_available():
    torch.cuda.synchronize()

start = time.time()
for i in range(n_iterations):
    fitness = problem_optimized.evaluate_fast(pop, batch_size=batch_size)

if torch.cuda.is_available():
    torch.cuda.synchronize()
elapsed_optimized_fast = time.time() - start

print(f"总耗时: {elapsed_optimized_fast:.3f}秒")
print(f"平均每次: {elapsed_optimized_fast/n_iterations:.3f}秒")

speedup_fast = elapsed_original_fast / elapsed_optimized_fast
print(f"\n加速比: {speedup_fast:.2f}x")

# ============================================================================
# 总结
# ============================================================================
print("\n" + "=" * 80)
print("性能对比总结")
print("=" * 80)

print(f"\nKNN交叉验证:")
print(f"  原版: {elapsed_original/n_iterations:.3f}秒/次")
print(f"  优化: {elapsed_optimized/n_iterations:.3f}秒/次")
print(f"  加速比: {speedup_knn:.2f}x")

print(f"\nFS评估 (with CV):")
print(f"  原版: {elapsed_original_fs/n_iterations:.3f}秒/次")
print(f"  优化: {elapsed_optimized_fs/n_iterations:.3f}秒/次")
print(f"  加速比: {speedup_fs:.2f}x")

print(f"\nFS快速评估 (no CV):")
print(f"  原版: {elapsed_original_fast/n_iterations:.3f}秒/次")
print(f"  优化: {elapsed_optimized_fast/n_iterations:.3f}秒/次")
print(f"  加速比: {speedup_fast:.2f}x")

print(f"\n综合加速比: {((speedup_knn + speedup_fs + speedup_fast) / 3):.2f}x")

# ============================================================================
# 建议
# ============================================================================
print("\n" + "=" * 80)
print("优化建议")
print("=" * 80)

print("\n1. 如果速度仍然较慢，考虑:")
print("   - 减少交叉验证折数 (n_splits: 5 -> 3)")
print("   - 增大批次大小 (batch_size: 256 -> 512)")
print("   - 使用快速评估模式 (use_cv=False)")
print("   - 减少PSO迭代次数")

print("\n2. 如果GPU利用率低:")
print("   - 检查数据是否都在GPU上")
print("   - 增大种群大小以增加并行度")
print("   - 使用 'nvidia-smi -l 1' 监控GPU使用")

print("\n3. 如果内存不足:")
print("   - 减小批次大小")
print("   - 减小种群大小")
print("   - 使用梯度检查点")

print("\n" + "=" * 80)
