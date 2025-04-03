import numpy as np  # 导入numpy库，用于数值计算
from mesa import Model  # 导入mesa的Model类，作为模型的基础
from mesa.space import SingleGrid  # 导入SingleGrid类，用于创建单个网格
from mesa.time import BaseScheduler  # 导入BaseScheduler类，作为基础的调度器
from mesa.datacollection import DataCollector  # 导入DataCollector类，用于收集模型数据
import pandas as pd  # 导入pandas库，用于数据分析

# 导入自定义模块
from core.agent import CulturalAgent  # 导入Agent类，定义了模型中的agent
from utils.reporters import get_cooperation_rate  # 导入合作率计算函数，用于计算模型中的合作率

# --- 定义数据收集函数 ---
def get_average_culture(model):
    """计算平均文化值"""
    if not model.schedule.agents:  # 如果模型中没有agent
        return 0  # 返回0
    return np.mean([agent.C for agent in model.schedule.agents])  # 计算所有agent的文化值的平均值

def get_std_culture(model):
    """计算文化值标准差"""
    if len(model.schedule.agents) < 2:  # 如果模型中agent数量小于2
        return 0  # 返回0
    return np.std([agent.C for agent in model.schedule.agents])  # 计算所有agent的文化值的标准差
# --- 结束数据收集函数 ---


class CulturalGame(Model):
    """
    文化博弈模型
    """
    def __init__(self, L=50, initial_coop_ratio=0.5, b=1.5, K=0.5,
                 C_dist="uniform", mu=0.5, sigma=0.1, seed=None,
                 # --- 文化演化参数 ---
                 K_C=0.1,           # 文化模仿噪声
                 p_update_C=0.1,    # 文化更新概率
                 p_mut=0.001):      # 文化突变概率
        
        super().__init__(seed=seed)  # 调用父类Model的初始化方法
        self.grid = SingleGrid(L, L, torus=True)  # 创建一个L*L的网格，torus=True表示网格是环形的
        self.schedule = BaseScheduler(self)  # 创建一个基础调度器，用于调度模型中的agent
        self.L = L  # 存储网格大小
        self.b = b  # 存储收益参数，表示背叛的诱惑
        self.K = K  # 存储策略更新噪声参数
        self.C_dist = C_dist  # 存储文化分布类型
        self.mu = mu  # 存储文化均值
        self.sigma = sigma  # 存储文化标准差
        self.running = True  # 设置模型运行状态为True

        # --- 存储文化演化参数 ---
        self.K_C = K_C  # 存储文化模仿噪声
        self.p_update_C = p_update_C  # 存储文化更新概率
        self.p_mut = p_mut  # 存储文化突变概率

        self.payoff_matrix = {  # 定义收益矩阵，表示agent之间的博弈收益
            1: {1: (1, 1), 0: (0, self.b)},  # 如果agent选择合作，对方也选择合作，则双方收益为(1, 1)；如果agent选择合作，对方选择背叛，则双方收益为(0, self.b)
            0: {1: (self.b, 0), 0: (0, 0)}   # 如果agent选择背叛，对方选择合作，则双方收益为(self.b, 0)；如果agent选择背叛，对方也选择背叛，则双方收益为(0, 0)
        }

        # 初始化Agent
        for _, pos in self.grid.coord_iter():  # 遍历网格中的每个位置
            strategy = 1 if self.random.random() < initial_coop_ratio else 0  # 随机初始化agent的策略，initial_coop_ratio表示初始合作率
            C_value = self._generate_culture()  # 生成agent的文化值
            agent = CulturalAgent(self.next_id(), self, strategy, C_value)  # 创建agent对象
            self.grid.place_agent(agent, pos)  # 将agent放置在网格中
            self.schedule.add(agent)  # 将agent添加到调度器中

        # 设置数据收集
        self.datacollector = DataCollector(  # 创建数据收集器
            model_reporters={  # 定义模型层面的数据收集
                "CooperationRate": get_cooperation_rate,  # 收集合作率
                "AverageCulture": get_average_culture,  # 收集平均文化值
                "StdCulture": get_std_culture  # 收集文化值标准差
            },
        )
        self.datacollector.collect(self)  # 收集初始数据

    def _generate_culture(self):
        """生成文化值"""
        if self.C_dist == "uniform":  # 如果文化分布是均匀分布
            return self.random.uniform(0, 1)  # 生成一个0到1之间的随机数
        elif self.C_dist == "bimodal":  # 如果文化分布是双峰分布
            if self.mu == 0.5:  # 如果mu=0.5
                return self.random.choice([0.0, 1.0])  # 随机选择0或1
            else:  # 如果mu!=0.5
                prob_1 = self.mu  # 设置选择1的概率为mu
                return self.random.choice([0.0, 1.0], p=[1 - prob_1, prob_1])  # 按照概率选择0或1
        elif self.C_dist == "normal":  # 如果文化分布是正态分布
            val = self.random.normalvariate(self.mu, self.sigma)  # 生成一个符合正态分布的随机数
            return np.clip(val, 0, 1)  # 将随机数截断到0到1之间
        elif self.C_dist == "fixed":  # 如果文化分布是固定值
            return self.mu  # 返回mu
        else:  # 如果文化分布类型不支持
            raise ValueError(f"Unsupported C distribution type: {self.C_dist}")  # 抛出异常

    def step(self):
        """执行一步"""
        # 1. 计算效用
        for agent in self.schedule.agents:  # 遍历所有agent
            agent.calculate_utility()  # 计算agent的效用
            
        # 2. 策略更新
        for agent in self.schedule.agents:  # 遍历所有agent
            agent.decide_strategy_update()  # agent决定是否更新策略

        # 3. 文化更新
        for agent in self.schedule.agents:  # 遍历所有agent
            agent.decide_culture_update()  # agent决定是否更新文化

        # 4. 应用更新
        self.schedule.step()  # 调度器执行一步，所有agent执行advance()方法

        # 5. 文化突变
        for agent in self.schedule.agents:  # 遍历所有agent
            agent.mutate_culture()  # agent发生文化突变

        # 5. 应用更新 (将agent的策略和文化值更新到实际状态)
        for agent in self.schedule.agents:  # 遍历所有agent
            agent.advance()  # agent执行advance()方法，更新策略和文化值

        # 6. 收集数据
        self.datacollector.collect(self)  # 收集数据

    def run_model(self, n_steps):
        """运行模型"""
        for i in range(n_steps):  # 循环n_steps次
            self.step()  # 执行一步
            # 可选: 打印进度
            # if (i+1) % 10 == 0:
            #     print(f"Step {i+1}/{n_steps}")
