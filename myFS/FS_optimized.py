# FS_optimized.py - GPU优化版本
import torch
from evox.core import Problem
from GPUKNN import TensorKNNClassifier

class FS_Optimized(Problem):
    """
    GPU优化版本的特征选择问题
    
    主要优化:
    1. 减少GPU-CPU数据传输
    2. 向量化批处理
    3. 预计算和缓存
    4. 减少不必要的内存分配
    """
    
    def __init__(self, device=None):
        super().__init__()
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.X = None
        self.y = None
        self.n_features = 0
        self.knn = None  # 重用KNN实例
        self.threshold = 0.6

    def set_data(self, X, y):
        """设置数据并移动到GPU"""
        if not torch.is_tensor(X):
            X = torch.tensor(X, dtype=torch.float32, device=self.device)
        else:
            X = X.to(dtype=torch.float32, device=self.device)

        if not torch.is_tensor(y):
            y = torch.tensor(y, dtype=torch.long, device=self.device)
        else:
            y = y.to(dtype=torch.long, device=self.device)

        self.X = X
        self.y = y
        self.n_features = self.X.shape[1]
        
        # 预创建KNN实例
        self.knn = TensorKNNClassifier(k=1, device=self.device)
        
        return self

    def evaluate(self, pop: torch.Tensor, batch_size: int = 256, use_cv: bool = True, n_splits: int = 5):
        """
        GPU优化的并行评估函数
        
        关键优化:
        1. 所有操作在GPU上完成，避免CPU-GPU传输
        2. 批处理减少内核启动开销
        3. 原地操作减少内存分配
        """
        # 确保pop在正确设备上
        if pop.device != self.device:
            pop = pop.to(self.device)
        
        # 应用阈值获取特征掩码（原地操作）
        pop_mask = (pop > self.threshold).float()
        pop_size = pop.shape[0]
        
        fitness_list = []
        
        # 分批处理以控制内存使用
        for b in range(0, pop_size, batch_size):
            batch_end = min(b + batch_size, pop_size)
            batch_pop = pop_mask[b:batch_end]
            current_batch_size = batch_end - b
            
            # 应用特征掩码: (batch_size, n_samples, n_features)
            # 使用Einstein求和加速
            all_feature_subsets = torch.einsum('bf,sf->bsf', batch_pop, self.X)
            
            if use_cv:
                # 使用交叉验证
                acc = self.knn.cross_validate(all_feature_subsets, self.y, n_splits=n_splits)
            else:
                # 不使用交叉验证（更快但可能过拟合）
                preds = self.knn.predict_batched_subsets(
                    all_feature_subsets, self.y, all_feature_subsets
                )
                acc = self.knn.accuracy(preds, self.y)
            
            fitness_list.append(acc)
        
        # 合并结果（在GPU上）
        fitness = torch.cat(fitness_list, dim=0)
        
        # 计算特征数量惩罚（向量化）
        n_features_ratio = pop_mask.sum(dim=1) / self.n_features
        alpha = 0.01  # 特征数量惩罚权重
        
        # 返回负准确率（PSO最小化）+ 特征惩罚
        penalized_fitness = -fitness + alpha * n_features_ratio
        
        return penalized_fitness

    @torch.no_grad()
    def evaluate_fast(self, pop: torch.Tensor, batch_size: int = 256):
        """
        快速评估版本：不使用交叉验证，直接在训练集上评估
        速度更快但可能导致过拟合，仅用于初步筛选
        """
        return self.evaluate(pop, batch_size=batch_size, use_cv=False)

    @torch.no_grad()
    def evaluate_accurate(self, pop: torch.Tensor, batch_size: int = 64, n_splits: int = 10):
        """
        精确评估版本：使用更多折数的交叉验证
        速度较慢但结果更可靠，用于最终评估
        """
        return self.evaluate(pop, batch_size=batch_size, use_cv=True, n_splits=n_splits)
