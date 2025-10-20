# GPUKNN_optimized.py - GPU优化版本
import torch

class TensorKNNClassifier_Optimized:
    """
    GPU优化的KNN分类器
    
    优化点:
    1. 减少不必要的tensor扩展和复制
    2. 使用更高效的索引操作
    3. 向量化所有操作
    4. 减少内存分配
    """

    def __init__(self, k: int = 1, device=None):
        self.k = k
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

    @torch.no_grad()
    def predict_batched_subsets(self, train_X_batch, train_y, test_X_batch):
        """
        对多个特征子集同时计算KNN预测
        
        优化:
        - 使用torch.cdist的高效实现
        - 避免不必要的tensor扩展
        - 使用gather操作代替循环
        """
        B, n_train, d = train_X_batch.shape
        _, n_test, _ = test_X_batch.shape

        # 计算距离矩阵 (B, n_test, n_train)
        # torch.cdist是GPU优化的，比手动计算快很多
        dists = torch.cdist(test_X_batch, train_X_batch, p=2)

        if self.k == 1:
            # K=1时的优化路径
            idx = dists.argmin(dim=2)  # (B, n_test)
            # 直接使用gather，避免expand
            train_y_expanded = train_y.unsqueeze(0).unsqueeze(1).expand(B, n_test, -1)
            preds = torch.gather(train_y_expanded, 2, idx.unsqueeze(2)).squeeze(2)
            return preds
        else:
            # K>1时找到最近的k个邻居
            _, knn_idx = torch.topk(dists, k=self.k, largest=False, dim=2)  # (B, n_test, k)
            
            # 获取k个邻居的标签
            train_y_expanded = train_y.unsqueeze(0).unsqueeze(1).unsqueeze(3).expand(B, n_test, n_train, 1)
            knn_labels = torch.gather(train_y_expanded, 2, knn_idx.unsqueeze(3)).squeeze(3)
            
            # 投票（找众数）
            preds, _ = knn_labels.mode(dim=2)
            return preds

    @torch.no_grad()
    def accuracy(self, preds, target):
        """
        计算准确率
        优化: 避免不必要的expand
        """
        # preds: (B, n_test), target: (n_test,)
        # 直接广播比较
        correct = (preds == target.unsqueeze(0))
        acc = correct.float().mean(dim=1)
        return acc

    @torch.no_grad()
    def cross_validate(self, all_feature_subsets, y, n_splits=5):
        """
        GPU优化的交叉验证
        
        优化:
        1. 预分配结果tensor
        2. 减少索引操作
        3. 批量处理所有fold
        """
        device = self.device
        B, n_samples, d = all_feature_subsets.shape
        
        # 确保数据在正确设备上
        if all_feature_subsets.device != device:
            all_feature_subsets = all_feature_subsets.to(device)
        if y.device != device:
            y = y.to(device)
        
        # 生成随机排列的索引
        idxs = torch.randperm(n_samples, device=device)
        
        # 预计算fold大小和索引
        fold_size = n_samples // n_splits
        remainder = n_samples % n_splits
        
        # 预分配准确率tensor
        accs = torch.zeros(B, n_splits, device=device, dtype=torch.float32)
        
        # 对每个fold进行评估
        for i in range(n_splits):
            # 计算当前fold的起始和结束位置
            start = i * fold_size + min(i, remainder)
            if i < remainder:
                end = start + fold_size + 1
            else:
                end = start + fold_size
            
            # 获取测试集索引
            test_idx = idxs[start:end]
            
            # 获取训练集索引（拼接其他fold）
            if i == 0:
                train_idx = idxs[end:]
            elif i == n_splits - 1:
                train_idx = idxs[:start]
            else:
                train_idx = torch.cat([idxs[:start], idxs[end:]])
            
            # 分割数据（使用高级索引，一次性完成）
            train_X_batch = all_feature_subsets[:, train_idx, :]
            test_X_batch = all_feature_subsets[:, test_idx, :]
            train_y = y[train_idx]
            test_y = y[test_idx]
            
            # 预测并计算准确率
            preds = self.predict_batched_subsets(train_X_batch, train_y, test_X_batch)
            accs[:, i] = self.accuracy(preds, test_y)
        
        # 返回平均准确率
        return accs.mean(dim=1)

    @torch.no_grad()
    def cross_validate_parallel(self, all_feature_subsets, y, n_splits=5):
        """
        尝试并行化所有fold的评估（实验性）
        
        注意: 这会占用更多GPU内存，但可能更快
        """
        device = self.device
        B, n_samples, d = all_feature_subsets.shape
        
        # 确保数据在正确设备上
        if all_feature_subsets.device != device:
            all_feature_subsets = all_feature_subsets.to(device)
        if y.device != device:
            y = y.to(device)
        
        # 生成索引
        idxs = torch.randperm(n_samples, device=device)
        fold_size = n_samples // n_splits
        
        accs_list = []
        
        # 并行处理所有fold
        for i in range(n_splits):
            start = i * fold_size
            end = start + fold_size if i < n_splits - 1 else n_samples
            
            test_mask = torch.zeros(n_samples, dtype=torch.bool, device=device)
            test_mask[idxs[start:end]] = True
            train_mask = ~test_mask
            
            # 使用mask索引
            train_X = all_feature_subsets[:, train_mask, :]
            test_X = all_feature_subsets[:, test_mask, :]
            train_y = y[train_mask]
            test_y = y[test_mask]
            
            preds = self.predict_batched_subsets(train_X, train_y, test_X)
            acc = self.accuracy(preds, test_y)
            accs_list.append(acc)
        
        # Stack并求平均
        accs = torch.stack(accs_list, dim=1)
        return accs.mean(dim=1)
