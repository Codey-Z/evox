"""
快速性能测试 - 对比优化前后的速度
"""
import torch
import time
import pandas as pd
from FS import FS

print("=" * 80)
print("快速性能测试")
print("=" * 80)

# 检查CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n设备: {device}")

if not torch.cuda.is_available():
    print("⚠️ 警告: CUDA不可用，将使用CPU（会很慢）")

# 加载数据
print("\n加载数据...")
data = pd.read_csv("csvdata/9_Tumors.csv").values
X = data[:, :-1].astype(float)
y = data[:, -1].astype(int)
print(f"数据集: {X.shape[0]}样本 x {X.shape[1]}特征")

# 创建问题
problem = FS(device=device)
problem.set_data(X, y)

# 生成测试种群
pop_size = 100
pop = torch.rand(pop_size, problem.n_features, device=device)

print(f"\n测试配置:")
print(f"  种群大小: {pop_size}")
print(f"  批次大小: 256")
print(f"  测试次数: 3次")

# 测试不同配置
configs = [
    ("快速模式 (无CV)", {"batch_size": 512, "use_cv": False}),
    ("3折CV", {"batch_size": 512, "use_cv": True, "n_splits": 3}),
    ("5折CV (标准)", {"batch_size": 256, "use_cv": True, "n_splits": 5}),
]

print("\n" + "=" * 80)
print("性能测试结果")
print("=" * 80)

for name, config in configs:
    print(f"\n{name}:")
    print(f"  配置: {config}")
    
    # 预热
    _ = problem.evaluate(pop[:10], **config)
    
    # 正式测试
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    start = time.time()
    for i in range(3):
        fitness = problem.evaluate(pop, **config)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    elapsed = time.time() - start
    avg_time = elapsed / 3
    
    print(f"  总耗时: {elapsed:.2f}秒")
    print(f"  平均: {avg_time:.2f}秒/次")
    print(f"  Fitness均值: {fitness.mean().item():.4f}")
    
    if torch.cuda.is_available():
        memory = torch.cuda.max_memory_allocated() / 1024**2
        print(f"  峰值GPU内存: {memory:.0f}MB")

print("\n" + "=" * 80)
print("测试完成")
print("=" * 80)

print("\n💡 优化建议:")
print("1. 如果'快速模式'足够快，PSO中可以使用它")
print("2. 如果'3折CV'结果可接受，建议使用它代替5折")
print("3. 增大batch_size可能进一步提速（如果内存足够）")
print("4. 在test.py中修改CONFIG参数以应用这些优化")

print("\n📝 修改test.py的CONFIG:")
print("CONFIG = {")
print("    'n_folds': 10,      # 或改为5加速")
print("    'pop_size': 100,     # 或改为50加速")  
print("    'n_iterations': 50,  # 或改为30加速")
print("    'threshold': 0.6,")
print("    'knn_k': 1,")
print("}")

print("\n📝 修改FS.py的evaluate默认参数:")
print("def evaluate(self, pop, batch_size=512, use_cv=True, n_splits=3):")
print("              #                  ^^^           ^^^           ^^^")
print("              #                  增大批次      或False       减少折数")
