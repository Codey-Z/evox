import torch
from FS import FS
import pandas as pd
import numpy as np
from pathlib import Path
from evox.algorithms import PSO
from evox.workflows import StdWorkflow, EvalMonitor
from GPUKNN import TensorKNNClassifier
from sklearn.model_selection import StratifiedKFold

# 设置设备
torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 配置参数
CONFIG = {
    'n_folds': 10,          # 10折交叉验证
    'pop_size': 200,        # PSO种群大小
    'n_iterations': 100,    # PSO迭代次数
    'threshold': 0.6,       # 特征选择阈值
    'knn_k': 1,            # KNN的K值
}

def load_dataset(csv_path):
    """加载数据集并分离特征和标签"""
    data = pd.read_csv(csv_path).values
    X = data[:, :-1].astype(float)
    y = data[:, -1].astype(int)
    return X, y

def run_pso_for_fold(X_train, y_train, config, device):
    """
    在单个fold的训练集上运行PSO进行特征选择
    
    返回:
        best_features: 最优特征子集的二值掩码 (numpy array)
        best_fitness: 最优适应度值
    """
    # 创建特征选择问题实例
    problem = FS(device=device)
    problem.set_data(X_train, y_train)
    
    dim = problem.n_features
    
    # 初始化PSO算法
    algorithm = PSO(
        pop_size=config['pop_size'],
        lb=0 * torch.ones(dim),
        ub=1 * torch.ones(dim)
    )
    
    # 创建监控器和工作流
    monitor = EvalMonitor()
    workflow = StdWorkflow(algorithm, problem, monitor)
    
    # 初始化并运行PSO
    workflow.init_step()
    
    print(f"    PSO迭代中...", end='', flush=True)
    for i in range(config['n_iterations']):
        workflow.step()
        if (i + 1) % 20 == 0:
            print(f"\r    PSO迭代: {i+1}/{config['n_iterations']}", end='', flush=True)
    print(f"\r    PSO迭代完成: {config['n_iterations']}/{config['n_iterations']}")
    
    # 获取最优解
    best_fitness = monitor.get_best_fitness()
    best_solution = monitor.get_best_solution()
    
    # 转换为二值特征掩码
    best_features = (best_solution.cpu().numpy() > config['threshold']).astype(int)
    
    return best_features, float(best_fitness)

def evaluate_on_test(X_train, y_train, X_test, y_test, feature_mask, config, device):
    """
    使用选定的特征在测试集上评估准确率
    
    参数:
        X_train, y_train: 训练集
        X_test, y_test: 测试集
        feature_mask: 特征选择掩码 (1D numpy array)
        
    返回:
        test_accuracy: 测试集准确率
        n_selected_features: 选择的特征数量
    """
    # 统计选择的特征数量
    n_selected = np.sum(feature_mask)
    
    if n_selected == 0:
        print("    警告: 未选择任何特征，使用全部特征")
        feature_mask = np.ones_like(feature_mask)
        n_selected = len(feature_mask)
    
    # 应用特征掩码
    X_train_selected = X_train[:, feature_mask.astype(bool)]
    X_test_selected = X_test[:, feature_mask.astype(bool)]
    
    # 转换为Tensor
    X_train_tensor = torch.tensor(X_train_selected, dtype=torch.float32, device=device)
    X_test_tensor = torch.tensor(X_test_selected, dtype=torch.float32, device=device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long, device=device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long, device=device)
    
    # 使用KNN评估
    knn = TensorKNNClassifier(k=config['knn_k'], device=device)
    
    # 预测
    X_train_batch = X_train_tensor.unsqueeze(0)  # (1, n_train, d)
    X_test_batch = X_test_tensor.unsqueeze(0)    # (1, n_test, d)
    
    preds = knn.predict_batched_subsets(X_train_batch, y_train_tensor, X_test_batch)
    test_acc = knn.accuracy(preds, y_test_tensor)
    
    return float(test_acc[0]), n_selected

def process_dataset(csv_path, config, device):
    """
    处理单个数据集，进行10折交叉验证
    
    返回:
        results: 包含每个fold结果的字典列表
    """
    dataset_name = Path(csv_path).stem
    print(f"\n{'='*60}")
    print(f"处理数据集: {dataset_name}")
    print(f"{'='*60}")
    
    # 加载数据
    X, y = load_dataset(csv_path)
    print(f"数据集大小: {X.shape[0]} 样本, {X.shape[1]} 特征, {len(np.unique(y))} 类别")
    
    # 初始化10折交叉验证
    skf = StratifiedKFold(n_splits=config['n_folds'], shuffle=True, random_state=42)
    
    results = []
    
    # 遍历每个fold
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        print(f"\n  Fold {fold_idx}/{config['n_folds']}:")
        
        # 分割数据
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        print(f"    训练集: {X_train.shape[0]} 样本, 测试集: {X_test.shape[0]} 样本")
        
        # 在训练集上运行PSO
        best_features, train_fitness = run_pso_for_fold(
            X_train, y_train, config, device
        )
        
        # 在测试集上评估
        test_acc, n_features = evaluate_on_test(
            X_train, y_train, X_test, y_test, 
            best_features, config, device
        )
        
        print(f"    训练集适应度: {train_fitness:.4f}")
        print(f"    选择特征数: {n_features}/{X.shape[1]}")
        print(f"    测试集准确率: {test_acc:.4f}")
        
        # 保存结果
        results.append({
            'fold': fold_idx,
            'train_fitness': train_fitness,
            'test_accuracy': test_acc,
            'n_features_selected': n_features,
            'total_features': X.shape[1],
            'feature_mask': best_features
        })
    
    # 汇总统计
    avg_test_acc = np.mean([r['test_accuracy'] for r in results])
    std_test_acc = np.std([r['test_accuracy'] for r in results])
    avg_features = np.mean([r['n_features_selected'] for r in results])
    
    print(f"\n  {dataset_name} 汇总结果:")
    print(f"  平均测试准确率: {avg_test_acc:.4f} ± {std_test_acc:.4f}")
    print(f"  平均特征数: {avg_features:.1f}/{X.shape[1]}")
    
    return {
        'dataset_name': dataset_name,
        'fold_results': results,
        'avg_test_accuracy': avg_test_acc,
        'std_test_accuracy': std_test_acc,
        'avg_features_selected': avg_features
    }

def main():
    """主函数：遍历所有数据集"""
    # 获取所有CSV文件
    csv_dir = Path("csvdata")
    csv_files = sorted(csv_dir.glob("*.csv"))
    
    print(f"找到 {len(csv_files)} 个数据集")
    print(f"配置: {CONFIG}")
    
    all_results = []
    
    # 遍历每个数据集
    for csv_file in csv_files:
        try:
            result = process_dataset(csv_file, CONFIG, device)
            all_results.append(result)
            break  # 仅处理第一个数据集以节省时间
        except Exception as e:
            print(f"\n错误: 处理 {csv_file.name} 时出错: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 打印最终汇总
    print(f"\n{'='*60}")
    print("所有数据集最终汇总")
    print(f"{'='*60}")
    print(f"{'数据集':<25} {'平均准确率':<15} {'平均特征数':<15}")
    print(f"{'-'*60}")
    
    for result in all_results:
        print(f"{result['dataset_name']:<25} "
              f"{result['avg_test_accuracy']:.4f}±{result['std_test_accuracy']:.4f}  "
              f"{result['avg_features_selected']:.1f}")
    
    print(f"{'='*60}")

if __name__ == "__main__":
    main()