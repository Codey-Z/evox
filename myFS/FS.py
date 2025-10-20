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

    def evaluate(self, pop: torch.Tensor, batch_size: int = 256):
        """
        并行批次化评估函数。
        每批次在GPU上同时计算多个个体的KNN准确率。
        """
        start_time = time.time()
        pop = pop.to(self.device)
        pop_mask = (pop > 0.6).float()
        popSize = pop.shape[0]

        knn = TensorKNNClassifier(k=1, device=self.device)
        fitness_list = []

        for b in range(0, popSize, batch_size):
            batch_pop = pop_mask[b:b+batch_size]
            print("batch_pop shape:", batch_pop.shape)
            print("self.X shape:", self.X.shape)
            all_feature_subsets = self.X.unsqueeze(0) * batch_pop.unsqueeze(1)
            acc = knn.cross_validate(all_feature_subsets, self.y, n_splits=5)
            fitness_list.append(acc)

        fitness = torch.cat(fitness_list, dim=0)
        print(fitness)
        return fitness
