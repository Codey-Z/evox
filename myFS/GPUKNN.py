# GPUKNN.py
import torch

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

    def cross_validate(self, all_feature_subsets, y, n_splits=5):
        """
        对多个特征子集同时进行交叉验证。
        输入:
          all_feature_subsets: (B, n_samples, d)
          y: (n_samples,)
        输出:
          acc: (B,)
        """
        device = self.device
        B, n_samples, d = all_feature_subsets.shape
        y = y.to(device)
        all_feature_subsets = all_feature_subsets.to(device)

        idxs = torch.randperm(n_samples, device=device)
        fold_sizes = [(n_samples // n_splits) + (1 if i < (n_samples % n_splits) else 0) for i in range(n_splits)]
        folds, start = [], 0
        for fs in fold_sizes:
            folds.append(idxs[start:start+fs])
            start += fs

        accs = torch.zeros((B, n_splits), device=device)

        for i in range(n_splits):
            test_idx = folds[i]
            train_idx = torch.cat([folds[j] for j in range(n_splits) if j != i])
            train_y, test_y = y[train_idx], y[test_idx]
            train_X_batch = all_feature_subsets[:, train_idx, :]
            test_X_batch = all_feature_subsets[:, test_idx, :]

            preds = self.predict_batched_subsets(train_X_batch, train_y, test_X_batch)
            accs[:, i] = self.accuracy(preds, test_y)

        return accs.mean(dim=1)
