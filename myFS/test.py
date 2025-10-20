import torch
from FS import FS
import pandas as pd
from evox.algorithms import PSO # 导入PSO算法
from evox.problems.numerical import Ackley # 导入Ackley优化问题
from evox.workflows import StdWorkflow, EvalMonitor # 导入标准工作流和监控器
# 1. 定义优化算法和问题
torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")
data = pd.read_csv("csvdata/9_Tumors.csv").values
X = data[:, :-1].astype(float)
Y = data[:, -1].astype(int)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
problem = FS(device=device)
problem.set_data(X, Y)
dim = problem.n_features
algorithm = PSO(
pop_size=200, # 种群规模为200
lb=-0 * torch.ones(dim), # 决策变量下界：二维向量，每维-32
ub= 1 * torch.ones(dim) # 决策变量上界：二维向量，每维32
)
# 2. 组合工作流（Workflow），并添加监控器用于跟踪结果
monitor = EvalMonitor()
workflow = StdWorkflow(algorithm, problem, monitor)
# 3. 初始化工作流
workflow.init_step() # 初始化算法和问题内部状态
# 4. 执行优化迭代
for i in range(100):
    workflow.step() # 推进优化一步
# 5. 获取结果（例如打印最优值）
best_fitness = monitor.get_best_fitness() # 从监控器获取当前迭代的最佳适应度值
print("迭代完成，当前找到的最优适应度值:", float(best_fitness))