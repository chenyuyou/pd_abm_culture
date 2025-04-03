import numpy as np
from mesa import Model
from mesa.space import SingleGrid
from mesa.time import BaseScheduler
from mesa.datacollection import DataCollector
import pandas as pd

# 导入自定义模块
from core.agent import CulturalAgent  # 导入Agent类
from utils.reporters import get_cooperation_rate  # 导入合作率计算函数

# --- 定义数据收集函数 ---
def get_average_culture(model):
    """计算平均文化值"""
    if not model.schedule.agents:
        return 0
    return np.mean([agent.C for agent in model.schedule.agents])

def get_std_culture(model):
    """计算文化值标准差"""
    if len(model.schedule.agents) < 2:
        return 0
    return np.std([agent.C for agent in model.schedule.agents])
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
        
        super().__init__(seed=seed)
        self.grid = SingleGrid(L, L, torus=True)  # 网格
        self.schedule = BaseScheduler(self)  # 调度器
        self.L = L  # 网格大小
        self.b = b  # 收益参数
        self.K = K  # 策略更新噪声
        self.C_dist = C_dist  # 文化分布
        self.mu = mu  # 文化均值
        self.sigma = sigma  # 文化标准差
        self.running = True  # 运行状态

        # --- 存储文化演化参数 ---
        self.K_C = K_C
        self.p_update_C = p_update_C
        self.p_mut = p_mut

        self.payoff_matrix = {
            1: {1: (1, 1), 0: (0, self.b)},
            0: {1: (self.b, 0), 0: (0, 0)}
        }

        # 初始化Agent
        for _, pos in self.grid.coord_iter():
            strategy = 1 if self.random.random() < initial_coop_ratio else 0
            C_value = self._generate_culture()
            agent = CulturalAgent(self.next_id(), self, strategy, C_value)
            self.grid.place_agent(agent, pos)
            self.schedule.add(agent)

        # 设置数据收集
        self.datacollector = DataCollector(
            model_reporters={
                "CooperationRate": get_cooperation_rate,
                "AverageCulture": get_average_culture,
                "StdCulture": get_std_culture
            },
        )
        self.datacollector.collect(self)

    def _generate_culture(self):
        """生成文化值"""
        if self.C_dist == "uniform":
            return self.random.uniform(0, 1)
        elif self.C_dist == "bimodal":
            if self.mu == 0.5:
                return self.random.choice([0.0, 1.0])
            else:
                prob_1 = self.mu
                return self.random.choice([0.0, 1.0], p=[1 - prob_1, prob_1])
        elif self.C_dist == "normal":
            val = self.random.normalvariate(self.mu, self.sigma)
            return np.clip(val, 0, 1)
        elif self.C_dist == "fixed":
            return self.mu
        else:
            raise ValueError(f"Unsupported C distribution type: {self.C_dist}")

    def step(self):
        """执行一步"""
        # 1. 计算效用
        for agent in self.schedule.agents:
            agent.calculate_utility()
            
        # 2. 策略更新
        for agent in self.schedule.agents:
            agent.decide_strategy_update()

        # 3. 文化更新
        for agent in self.schedule.agents:
            agent.decide_culture_update()

        # 4. 应用更新
        self.schedule.step()

        # 5. 文化突变
        for agent in self.schedule.agents:
            agent.mutate_culture()

        # 5. 应用更新 (将agent的策略和文化值更新到实际状态)
        for agent in self.schedule.agents:
            agent.advance()

        # 6. 收集数据
        self.datacollector.collect(self)

    def run_model(self, n_steps):
        """运行模型"""
        for i in range(n_steps):
            self.step()
            # 可选: 打印进度
            # if (i+1) % 10 == 0:
            #     print(f"Step {i+1}/{n_steps}")
