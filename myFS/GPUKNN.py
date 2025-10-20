# GPUKNN.py
import torch
import time

class TensorKNNClassifier:
    """
    完全GPU加速的KNN分类器，用于特征选择（FS）并行评估。
    使用torch.cdist计算距离，支持batched并行处理多个特征子集。
    """

    def __init__(self, k: int = 1, device=None):
        self.k = k
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

    def predict_batched_subsets(self, train_X_batch, train_y, test_X_batch):
        """
        对多个特征子集同时计算KNN预测。
        输入:
          train_X_batch: (B, n_train, d)
          test_X_batch:  (B, n_test, d)
          train_y: (n_train,)
        输出:
          preds: (B, n_test)
        """
        B, n_train, _ = train_X_batch.shape
        _, n_test, _ = test_X_batch.shape

        # 计算距离矩阵 (B, n_test, n_train)
        dists = torch.cdist(test_X_batch, train_X_batch)  # (B, n_test, n_train)

        if self.k == 1:
            idx = dists.argmin(dim=2)  # (B, n_test)
            train_y_expand = train_y.unsqueeze(0).expand(B, -1)
            preds = torch.gather(train_y_expand, 1, idx)
            return preds
        else:
            knn_vals, knn_idx = torch.topk(dists, k=self.k, largest=False, dim=2)  # (B, n_test, k)
            train_y_expand = train_y.unsqueeze(0).unsqueeze(0).expand(B, n_test, n_train)
            knn_labels = torch.gather(train_y_expand, 2, knn_idx)
            preds, _ = knn_labels.mode(dim=2)
            return preds

    def accuracy(self, preds, target):
        """
        preds: (B, n_test)
        target: (n_test,)
        返回每个子集的准确率 (B,)
        """
        target_expand = target.unsqueeze(0).expand(preds.shape[0], -1)
        acc = (preds == target_expand).float().mean(dim=1)
        return acc

    def cross_validate(self, all_feature_subsets, y, n_splits=5, batch_idx=0):
        """
        对多个特征子集同时进行交叉验证。
        
        GPU优化:
        1. 预分配结果tensor避免动态内存分配
        2. 使用更高效的索引操作
        3. 减少不必要的数据复制
        
        输入:
          all_feature_subsets: (B, n_samples, d)
          y: (n_samples,)
          batch_idx: 当前批次索引（用于打印）
        输出:
          acc: (B,) - 每个特征子集的平均准确率
        """
        cv_start_time = time.time()
        device = self.device
        B, n_samples, d = all_feature_subsets.shape
        
        # 确保数据在正确设备上（避免隐式传输）
        transfer_start = time.time()
        if y.device != device:
            y = y.to(device, non_blocking=True)
        if all_feature_subsets.device != device:
            all_feature_subsets = all_feature_subsets.to(device, non_blocking=True)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        transfer_time = time.time() - transfer_start
        
        # 生成随机排列索引（在GPU上）
        prep_start = time.time()
        idxs = torch.randperm(n_samples, device=device)
        
        # 预计算fold大小
        fold_size = n_samples // n_splits
        remainder = n_samples % n_splits
        
        # 预分配准确率tensor（避免动态append）
        accs = torch.zeros(B, n_splits, device=device, dtype=torch.float32)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        prep_time = time.time() - prep_start

        # 对每个fold进行评估
        fold_times = []
        for i in range(n_splits):
            fold_start = time.time()
            
            # 计算当前fold的索引范围
            idx_start = time.time()
            start = i * fold_size + min(i, remainder)
            if i < remainder:
                end = start + fold_size + 1
            else:
                end = start + fold_size
            
            # 获取测试集索引
            test_idx = idxs[start:end]
            
            # 获取训练集索引（GPU优化：避免列表拼接）
            if i == 0:
                train_idx = idxs[end:]
            elif i == n_splits - 1:
                train_idx = idxs[:start]
            else:
                train_idx = torch.cat([idxs[:start], idxs[end:]])
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            idx_time = time.time() - idx_start
            
            # 使用高级索引分割数据（一次性完成，在GPU上）
            split_start = time.time()
            train_y = y[train_idx]
            test_y = y[test_idx]
            train_X_batch = all_feature_subsets[:, train_idx, :]
            test_X_batch = all_feature_subsets[:, test_idx, :]
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            split_time = time.time() - split_start

            # 预测并计算准确率（所有操作在GPU上）
            pred_start = time.time()
            preds = self.predict_batched_subsets(train_X_batch, train_y, test_X_batch)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            pred_time = time.time() - pred_start
            
            acc_start = time.time()
            accs[:, i] = self.accuracy(preds, test_y)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            acc_time = time.time() - acc_start
            
            fold_total = time.time() - fold_start
            fold_times.append(fold_total)
            
            print(f"        [CV Fold {i+1}/{n_splits}] 索引={idx_time:.3f}s, 分割={split_time:.3f}s, 预测={pred_time:.3f}s, 准确率={acc_time:.3f}s, 总计={fold_total:.3f}s")

        # 返回平均准确率
        mean_start = time.time()
        result = accs.mean(dim=1)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        mean_time = time.time() - mean_start
        
        total_cv_time = time.time() - cv_start_time
        avg_fold_time = sum(fold_times) / len(fold_times) if fold_times else 0
        
        print(f"        [CV 总结] 传输={transfer_time:.3f}s, 准备={prep_time:.3f}s, 平均每fold={avg_fold_time:.3f}s, 求均值={mean_time:.3f}s, CV总计={total_cv_time:.3f}s")
        
        return result
