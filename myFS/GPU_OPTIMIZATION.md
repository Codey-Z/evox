# GPU性能优化指南

## 🔍 已发现的性能问题

### 1. **潜在的CPU-GPU数据传输**
- **问题**: 数据在CPU和GPU之间频繁传输会严重降低性能
- **原因**: 
  - pop未明确转移到GPU
  - 中间结果可能回传到CPU
  - 使用`.to(device)`而不是检查设备

### 2. **低效的tensor操作**
- **问题**: 使用`unsqueeze`和广播操作效率低
- **优化**: 改用Einstein求和 (`torch.einsum`)

### 3. **交叉验证开销大**
- **问题**: 5折交叉验证使计算量增加5倍
- **优化**: 可以减少折数或使用快速评估模式

### 4. **批次大小不优化**
- **问题**: 批次太小导致GPU未充分利用
- **优化**: 增大batch_size以提高并行度

## ✅ 已实施的优化

### FS.py 优化

```python
# 优化1: 使用Einstein求和代替广播
# 原来: self.X.unsqueeze(0) * batch_pop.unsqueeze(1)
all_feature_subsets = torch.einsum('bf,sf->bsf', batch_pop, self.X)

# 优化2: 检查设备避免隐式传输
if pop.device != self.device:
    pop = pop.to(self.device, non_blocking=True)

# 优化3: 向量化特征数量计算
n_features_ratio = pop_mask.sum(dim=1) / self.n_features

# 优化4: 所有操作在GPU上完成
penalized_fitness = -fitness + alpha * n_features_ratio
```

### GPUKNN.py 优化

```python
# 优化1: 预分配结果tensor
accs = torch.zeros(B, n_splits, device=device, dtype=torch.float32)

# 优化2: 避免列表拼接
if i == 0:
    train_idx = idxs[end:]
elif i == n_splits - 1:
    train_idx = idxs[:start]
else:
    train_idx = torch.cat([idxs[:start], idxs[end:]])

# 优化3: 使用non_blocking传输
y = y.to(device, non_blocking=True)
```

## 📊 性能测试工具

### 1. `diagnose_gpu.py` - GPU诊断
检查GPU使用情况和性能瓶颈

```bash
python diagnose_gpu.py
```

**输出内容**:
- CUDA可用性检查
- 数据位置验证
- CPU-GPU数据传输检测
- GPU内存使用分析
- 批次大小优化建议

### 2. `profile_gpu.py` - 性能分析
详细分析GPU性能

```bash
python profile_gpu.py
```

**测试项目**:
- 不使用交叉验证的性能
- 使用交叉验证的性能
- KNN批量预测性能
- 长时间运行测试

### 3. `benchmark_performance.py` - 性能对比
对比原版和优化版的性能

```bash
python benchmark_performance.py
```

**对比内容**:
- KNN交叉验证速度
- FS Problem评估速度
- 快速评估模式速度
- 内存使用情况

## 🚀 优化配置建议

### 配置1: 平衡模式（推荐）
```python
CONFIG = {
    'n_folds': 10,          # 外层10折交叉验证
    'pop_size': 100,        # 减小种群大小
    'n_iterations': 50,     # 减少迭代次数
    'threshold': 0.6,       
    'knn_k': 1,            
}

# 在FS.evaluate()中
fitness = problem.evaluate(
    pop, 
    batch_size=512,         # 增大批次大小
    use_cv=True, 
    n_splits=3              # 减少内层交叉验证折数
)
```

### 配置2: 快速模式
```python
CONFIG = {
    'n_folds': 5,           # 减少到5折
    'pop_size': 50,         # 小种群
    'n_iterations': 30,     # 少迭代
    'threshold': 0.6,       
    'knn_k': 1,            
}

# 不使用内层交叉验证
fitness = problem.evaluate(
    pop, 
    batch_size=512,        
    use_cv=False            # 快速模式
)
```

### 配置3: 精确模式
```python
CONFIG = {
    'n_folds': 10,          
    'pop_size': 200,        
    'n_iterations': 100,    
    'threshold': 0.6,       
    'knn_k': 1,            
}

# 使用标准配置
fitness = problem.evaluate(
    pop, 
    batch_size=256,         
    use_cv=True, 
    n_splits=5
)
```

## 📝 修改后的test.py使用方法

### 使用优化后的代码

**test.py已自动应用优化**，无需额外修改。但你可以调整参数:

```python
# 在test.py中修改CONFIG
CONFIG = {
    'n_folds': 10,          # 改为5或3可加速
    'pop_size': 200,        # 改为100可加速
    'n_iterations': 100,    # 改为50可加速
    'threshold': 0.6,       
    'knn_k': 1,            
}

# 如果想使用快速模式，修改FS.py中的默认参数
# 或者在run_pso_for_fold()函数中修改evaluate()调用
```

### 快速修改方法

在`test.py`的`run_pso_for_fold()`函数中找到:
```python
problem = FS(device=device)
```

暂时没有直接调用evaluate的地方，因为它在workflow内部调用。

要修改评估参数，需要在`FS.py`中修改`evaluate()`的默认参数:

```python
def evaluate(self, pop: torch.Tensor, 
             batch_size: int = 512,        # 增大批次
             use_cv: bool = True,          # 或改为False加速
             n_splits: int = 3):           # 减少折数
```

## 🔧 故障排查

### 问题1: 速度仍然很慢

**检查清单**:
1. 运行`diagnose_gpu.py`检查GPU使用
2. 在另一个终端运行`nvidia-smi -l 1`监控GPU
3. 检查GPU利用率是否接近100%

**解决方案**:
- GPU利用率低 → 增大batch_size和pop_size
- GPU内存不足 → 减小batch_size
- 交叉验证太慢 → 减少n_splits或use_cv=False

### 问题2: CUDA Out of Memory

**解决方案**:
```python
# 减小批次大小
batch_size=128  # 或更小

# 减小种群大小
pop_size=50

# 清理GPU缓存
torch.cuda.empty_cache()
```

### 问题3: 结果不准确

**原因**: 使用了快速模式(use_cv=False)

**解决方案**:
```python
# 使用交叉验证
use_cv=True
n_splits=5  # 或更多
```

## 📈 预期性能提升

基于优化后的代码，预期性能提升:

| 优化项 | 预期加速比 |
|--------|-----------|
| Einstein求和 | 1.2-1.5x |
| 减少数据传输 | 1.3-2.0x |
| 批次大小优化 | 1.2-1.8x |
| 减少交叉验证折数 (5→3) | 1.6x |
| 快速模式 (use_cv=False) | 4-5x |

**综合加速**: 2-10x（取决于配置）

## 💡 最佳实践

### 1. 开发阶段
- 使用快速模式快速迭代
- 小数据集测试
- 减少迭代次数

```python
CONFIG = {
    'n_folds': 3,
    'pop_size': 50,
    'n_iterations': 20,
}
# use_cv=False
```

### 2. 正式实验
- 使用完整配置
- 多次运行求平均
- 保存详细日志

```python
CONFIG = {
    'n_folds': 10,
    'pop_size': 200,
    'n_iterations': 100,
}
# use_cv=True, n_splits=5
```

### 3. 调试性能
- 运行`diagnose_gpu.py`
- 监控`nvidia-smi`
- 使用PyTorch Profiler

## 📞 进一步帮助

如果问题仍未解决:

1. 运行`diagnose_gpu.py`并查看输出
2. 检查GPU型号和CUDA版本
3. 尝试不同的batch_size
4. 考虑使用`FS_optimized.py`和`GPUKNN_optimized.py`

## 🎯 关键要点

✅ **已优化**:
- Einstein求和代替广播
- 减少CPU-GPU传输
- 预分配tensor
- 向量化操作

⚠️ **主要瓶颈**:
- 交叉验证开销（5x计算量）
- 批次大小影响GPU利用率
- 种群大小和迭代次数

🚀 **快速改进**:
1. 减少n_splits: 5 → 3
2. 增大batch_size: 256 → 512
3. 减少迭代: 100 → 50
