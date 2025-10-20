# 基于PSO的特征选择实验

## 文件说明

- `FS.py`: 特征选择问题定义，继承自EvoX的Problem类
- `GPUKNN.py`: GPU加速的KNN分类器，支持批量并行处理
- `test_single.py`: 简化测试脚本，用于快速验证单个数据集
- `test.py`: 完整实验脚本，遍历所有数据集并进行10折交叉验证

## 实验流程

### 完整实验 (test.py)
1. 遍历 `csvdata/` 目录下的所有CSV数据集
2. 对每个数据集进行10折交叉验证
3. 在每个fold中：
   - 使用训练集运行PSO进行特征选择
   - PSO内部使用5折交叉验证评估特征子集
   - 获得最优特征子集后，在测试集上评估准确率
4. 输出每个数据集的平均准确率和标准差

### 快速测试 (test_single.py)
- 仅测试单个数据集 (9_Tumors.csv)
- 简单的训练/测试集划分 (80%/20%)
- 减少PSO种群大小和迭代次数
- 用于快速验证代码逻辑

## 配置参数

在 `test.py` 中可以修改 `CONFIG` 字典：

```python
CONFIG = {
    'n_folds': 10,          # 外层交叉验证的折数
    'pop_size': 200,        # PSO种群大小
    'n_iterations': 100,    # PSO迭代次数
    'threshold': 0.6,       # 特征选择阈值 (>0.6则选择该特征)
    'knn_k': 1,            # KNN的K值
}
```

在 `FS.py` 的 `evaluate()` 方法中：
- `use_cv=True`: 使用内层交叉验证
- `n_splits=5`: 内层交叉验证的折数
- `alpha=0.01`: 特征数量惩罚权重

## 运行示例

### 快速测试
```bash
python test_single.py
```

### 完整实验
```bash
python test.py
```

## 输出说明

完整实验会输出：
1. 每个数据集的处理进度
2. 每个fold的详细结果：
   - 训练集适应度（负准确率）
   - 选择的特征数量
   - 测试集准确率
3. 每个数据集的汇总统计：
   - 平均测试准确率 ± 标准差
   - 平均特征选择数量
4. 所有数据集的最终汇总表格

## 适应度函数设计

```python
fitness = -accuracy + alpha * (n_selected_features / total_features)
```

- 主要目标：最大化分类准确率（最小化负准确率）
- 次要目标：减少特征数量（通过惩罚项）
- `alpha=0.01` 表示特征数量的权重较小

## GPU加速

- 所有计算都在GPU上进行（如果可用）
- KNN距离计算使用 `torch.cdist`
- 支持批量并行评估多个特征子集
- 批次大小可在 `FS.evaluate()` 中通过 `batch_size` 参数调整

## 依赖项

- torch >= 2.6.0
- pandas
- numpy
- scikit-learn (用于StratifiedKFold)
- evox >= 1.0.0

## 注意事项

1. PSO寻找最小值，因此fitness = -accuracy
2. 特征选择使用阈值0.6（可调整）
3. 如果某个解未选择任何特征，会自动回退到使用全部特征
4. 内层和外层都使用交叉验证，可能导致计算时间较长
5. 可以通过减少 `n_iterations` 或 `pop_size` 来加快速度（会影响结果质量）
