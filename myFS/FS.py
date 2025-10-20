# FS.py
import torch
from evox.core import Problem
from GPUKNN import TensorKNNClassifier
import time

class FS(Problem):
    def __init__(self, device=None):
        super().__init__()
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.X = None
        self.y = None
        self.n_features = 0

    def set_data(self, X, y):
        """
        设置数据并移动到GPU。
        """
        if not torch.is_tensor(X):
            X = torch.tensor(X, dtype=torch.float32)
        else:
            X = X.to(torch.float32)

        if not torch.is_tensor(y):
            y = torch.tensor(y, dtype=torch.long)
        else:
            y = y.to(torch.long)

        self.X = X.to(self.device)
        self.y = y.to(self.device)
        self.n_features = self.X.shape[1]
        return self

    def evaluate(self, pop: torch.Tensor, batch_size: int = 256, use_cv: bool = True, n_splits: int = 5):
        """
        GPU优化的并行批次化评估函数。
        每批次在GPU上同时计算多个个体的KNN准确率。
        
        参数:
            pop: 种群，形状为 (pop_size, n_features)
            batch_size: 批次大小
            use_cv: 是否使用交叉验证（默认True）。如果False，则使用全部训练数据
            n_splits: 交叉验证的折数（仅当use_cv=True时有效）
        
        返回:
            fitness: 负准确率（因为PSO是最小化问题），形状为 (pop_size,)
            
        GPU优化:
        1. 确保所有操作在GPU上完成，避免CPU-GPU数据传输
        2. 使用Einstein求和加速特征掩码应用
        3. 预计算和重用KNN实例
        4. 向量化特征数量计算
        """
        eval_start_time = time.time()
        print(f"    [FS.evaluate] 开始评估，种群大小={pop.shape[0]}, 批次大小={batch_size}, use_cv={use_cv}, n_splits={n_splits}")
        
        # 确保pop在正确的设备上（避免隐式传输）
        transfer_start = time.time()
        if pop.device != self.device:
            pop = pop.to(self.device, non_blocking=True)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        transfer_time = time.time() - transfer_start
        if transfer_time > 0.001:
            print(f"    [FS.evaluate] 数据传输耗时: {transfer_time:.3f}秒")
        
        # 应用阈值获取特征掩码（在GPU上）
        mask_start = time.time()
        pop_mask = (pop > 0.6).float()
        pop_size = pop.shape[0]
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        mask_time = time.time() - mask_start
        print(f"    [FS.evaluate] 特征掩码计算耗时: {mask_time:.3f}秒")

        # 创建KNN实例（可以考虑预创建以减少开销）
        knn = TensorKNNClassifier(k=1, device=self.device)
        fitness_list = []

        # 分批处理
        total_batches = (pop_size + batch_size - 1) // batch_size
        print(f"    [FS.evaluate] 开始批处理，共{total_batches}个批次")
        
        for batch_idx, b in enumerate(range(0, pop_size, batch_size)):
            batch_start = time.time()
            batch_end = min(b + batch_size, pop_size)
            batch_pop = pop_mask[b:batch_end]
            current_batch_size = batch_end - b
            
            # GPU优化：使用Einstein求和代替unsqueeze和广播
            einsum_start = time.time()
            all_feature_subsets = torch.einsum('bf,sf->bsf', batch_pop, self.X)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            einsum_time = time.time() - einsum_start
            
            if use_cv:
                # 使用交叉验证评估（所有操作在GPU上）
                cv_start = time.time()
                acc = knn.cross_validate(all_feature_subsets, self.y, n_splits=n_splits, batch_idx=batch_idx)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                cv_time = time.time() - cv_start
                print(f"      [Batch {batch_idx+1}/{total_batches}] einsum={einsum_time:.3f}s, CV={cv_time:.3f}s, 总计={time.time()-batch_start:.3f}s")
            else:
                # 不使用交叉验证（更快）
                pred_start = time.time()
                preds = knn.predict_batched_subsets(
                    all_feature_subsets, self.y, all_feature_subsets
                )
                acc = knn.accuracy(preds, self.y)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                pred_time = time.time() - pred_start
                print(f"      [Batch {batch_idx+1}/{total_batches}] einsum={einsum_time:.3f}s, 快速评估={pred_time:.3f}s, 总计={time.time()-batch_start:.3f}s")
            
            fitness_list.append(acc)

        # 在GPU上合并结果
        merge_start = time.time()
        fitness = torch.cat(fitness_list, dim=0)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        merge_time = time.time() - merge_start
        print(f"    [FS.evaluate] 结果合并耗时: {merge_time:.3f}秒")
        
        # 计算特征数量惩罚（向量化，在GPU上）
        penalty_start = time.time()
        n_features_ratio = pop_mask.sum(dim=1) / self.n_features
        alpha = 0.1  # 特征数量惩罚权重
        
        # 返回负准确率（PSO最小化目标）+ 特征惩罚
        # 注意：这里返回负值，因为PSO是最小化算法
        penalized_fitness = fitness + alpha * n_features_ratio
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        penalty_time = time.time() - penalty_start
        print(f"    [FS.evaluate] 惩罚计算耗时: {penalty_time:.3f}秒")
        
        total_time = time.time() - eval_start_time
        print(f"    [FS.evaluate] 评估总耗时: {total_time:.3f}秒\n")
        
        return penalized_fitness
