"""
GPU性能诊断脚本
检查是否存在CPU-GPU数据传输瓶颈
"""
import torch
import time
import pandas as pd
import numpy as np
from FS import FS

print("=" * 80)
print("GPU性能诊断")
print("=" * 80)

# 检查CUDA
if not torch.cuda.is_available():
    print("\n警告: CUDA不可用，将使用CPU运行")
    print("这会非常慢！请检查:")
    print("1. 是否安装了正确的CUDA版本")
    print("2. PyTorch是否支持CUDA")
    print("3. 运行: python -c 'import torch; print(torch.cuda.is_available())'")
    exit(1)

device = torch.device("cuda")
print(f"\n✓ CUDA可用")
print(f"GPU型号: {torch.cuda.get_device_name(0)}")
print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
print(f"默认设备: {torch.get_default_device()}")

# 加载数据
print("\n加载数据...")
csv_path = "csvdata/9_Tumors.csv"
data = pd.read_csv(csv_path).values
X = data[:, :-1].astype(float)
y = data[:, -1].astype(int)

print(f"数据集: 样本={X.shape[0]}, 特征={X.shape[1]}")

# 创建问题实例
print("\n创建FS问题实例...")
problem = FS(device=device)
problem.set_data(X, y)

# 检查数据位置
print(f"\n数据位置检查:")
print(f"  X在GPU: {problem.X.is_cuda} (device: {problem.X.device})")
print(f"  y在GPU: {problem.y.is_cuda} (device: {problem.y.device})")

if not problem.X.is_cuda or not problem.y.is_cuda:
    print("  ⚠️ 警告: 数据不在GPU上！")
else:
    print("  ✓ 数据已在GPU上")

# 创建测试种群
pop_size = 100
pop = torch.rand(pop_size, problem.n_features, device=device)
print(f"\n种群在GPU: {pop.is_cuda} (device: {pop.device})")

# 测试1: 检查数据传输
print("\n" + "=" * 80)
print("测试1: 检查是否有CPU-GPU数据传输")
print("=" * 80)

# 使用profiler检测数据传输
print("\n运行评估（监控数据传输）...")

with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    record_shapes=True,
    with_stack=True,
) as prof:
    fitness = problem.evaluate(pop, batch_size=50, use_cv=False, n_splits=3)

# 分析profiler结果
print("\n性能分析结果:")
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

# 检查数据传输操作
memcpy_events = [evt for evt in prof.key_averages() if 'memcpy' in evt.key.lower() or 'to' in evt.key.lower()]
if memcpy_events:
    print("\n⚠️ 检测到CPU-GPU数据传输:")
    for evt in memcpy_events:
        print(f"  {evt.key}: {evt.device_time_total/1000:.3f}ms")
else:
    print("\n✓ 未检测到明显的CPU-GPU数据传输")

# 测试2: GPU利用率测试
print("\n" + "=" * 80)
print("测试2: GPU计算时间分析")
print("=" * 80)

# 不使用交叉验证
print("\n不使用交叉验证 (use_cv=False):")
torch.cuda.synchronize()
start = time.time()
fitness = problem.evaluate(pop, batch_size=256, use_cv=False)
torch.cuda.synchronize()
elapsed = time.time() - start
print(f"  耗时: {elapsed:.3f}秒")

# 使用3折交叉验证
print("\n使用3折交叉验证 (use_cv=True, n_splits=3):")
torch.cuda.synchronize()
start = time.time()
fitness = problem.evaluate(pop, batch_size=256, use_cv=True, n_splits=3)
torch.cuda.synchronize()
elapsed_3fold = time.time() - start
print(f"  耗时: {elapsed_3fold:.3f}秒")

# 使用5折交叉验证
print("\n使用5折交叉验证 (use_cv=True, n_splits=5):")
torch.cuda.synchronize()
start = time.time()
fitness = problem.evaluate(pop, batch_size=256, use_cv=True, n_splits=5)
torch.cuda.synchronize()
elapsed_5fold = time.time() - start
print(f"  耗时: {elapsed_5fold:.3f}秒")

print(f"\n交叉验证开销: {elapsed_5fold/elapsed:.1f}x")

# 测试3: 批次大小影响
print("\n" + "=" * 80)
print("测试3: 批次大小对性能的影响")
print("=" * 80)

batch_sizes = [32, 64, 128, 256, 512]
times = []

for bs in batch_sizes:
    torch.cuda.synchronize()
    start = time.time()
    try:
        fitness = problem.evaluate(pop, batch_size=bs, use_cv=False)
        torch.cuda.synchronize()
        elapsed = time.time() - start
        times.append(elapsed)
        print(f"batch_size={bs:3d}: {elapsed:.3f}秒")
    except RuntimeError as e:
        if "out of memory" in str(e):
            print(f"batch_size={bs:3d}: OOM (内存不足)")
            break
        else:
            raise

if times:
    best_bs = batch_sizes[np.argmin(times)]
    print(f"\n最优批次大小: {best_bs}")

# 测试4: GPU内存使用
print("\n" + "=" * 80)
print("测试4: GPU内存使用分析")
print("=" * 80)

torch.cuda.reset_peak_memory_stats()
torch.cuda.synchronize()

fitness = problem.evaluate(pop, batch_size=256, use_cv=True, n_splits=5)
torch.cuda.synchronize()

allocated = torch.cuda.memory_allocated() / 1024**2
reserved = torch.cuda.memory_reserved() / 1024**2
peak = torch.cuda.max_memory_allocated() / 1024**2

print(f"\n当前分配: {allocated:.2f} MB")
print(f"当前保留: {reserved:.2f} MB")
print(f"峰值使用: {peak:.2f} MB")
print(f"GPU总内存: {torch.cuda.get_device_properties(0).total_memory / 1024**2:.2f} MB")
print(f"内存利用率: {peak / (torch.cuda.get_device_properties(0).total_memory / 1024**2) * 100:.1f}%")

# 总结和建议
print("\n" + "=" * 80)
print("诊断总结和优化建议")
print("=" * 80)

print("\n如果性能仍然不理想，可以尝试:")
print("\n1. 减少交叉验证折数:")
print("   - 当前使用5折，可以减少到3折")
print("   - 在FS.evaluate()中设置 n_splits=3")

print("\n2. 增大批次大小:")
print(f"   - 当前批次大小可以从256增加到{best_bs if times else 512}")
print("   - 更大的批次=更好的GPU利用率")

print("\n3. 使用快速评估模式:")
print("   - 设置 use_cv=False 跳过交叉验证")
print("   - 速度提升但可能过拟合")

print("\n4. 减少PSO参数:")
print("   - 减少种群大小: pop_size=200 -> 100")
print("   - 减少迭代次数: n_iterations=100 -> 50")

print("\n5. 监控GPU使用:")
print("   - 在另一个终端运行: nvidia-smi -l 1")
print("   - GPU利用率应该接近100%")

print("\n" + "=" * 80)
