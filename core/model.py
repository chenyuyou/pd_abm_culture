# model.py
import numpy as np
from mesa import Model
from mesa.space import SingleGrid
# Import StagedActivation
from mesa.time import StagedActivation # Correct import
from mesa.datacollection import DataCollector
import pandas as pd

from core.agent import CulturalAgent
from utils.reporters import get_cooperation_rate # Ensure this is available

# --- Data collection functions (keep as is) ---
def get_average_culture(model):
    # ... (no change)
    if not model.schedule.agents: return 0
    return np.mean([agent.C for agent in model.schedule.agents])

def get_std_culture(model):
    # ... (no change)
    if len(model.schedule.agents) < 2: return 0
    return np.std([agent.C for agent in model.schedule.agents])
# --- End data collection functions ---

class CulturalGame(Model):
    """
    文化博弈模型 (Using Multi-Stage StagedActivation)
    """
    def __init__(self, L=50, initial_coop_ratio=0.5, b=1.5, K=0.5,
                 C_dist="uniform", mu=0.5, sigma=0.1, seed=None,
                 K_C=0.1, p_update_C=0.1, p_mut=0.001):

        super().__init__(seed=seed)
        self.grid = SingleGrid(L, L, torus=True)

        # --- Use StagedActivation with Multiple Stages ---
        # Define the stages in the exact order they should run for ALL agents
        stage_list = [
            "calculate_utility",        # Stage 1: All agents calculate utility
            "decide_strategy_update",   # Stage 2: All agents decide next strategy
            "decide_culture_update",    # Stage 3: All agents decide next culture
            "mutate_culture",           # Stage 4: All agents potentially mutate next culture
            "advance"                   # Stage 5: All agents apply the updates
        ]
        # The method names in the agent match the stage names
        self.schedule = StagedActivation(self, stage_list=stage_list, shuffle=False, shuffle_between_stages=False)
        # shuffle=False: Agents activate in the order they were added within each stage.
        # shuffle_between_stages=False: The order is maintained across stages.
        # If order doesn't matter, you can set shuffle=True.
        # --- End Scheduler Change ---

        self.L = L
        self.b = b
        self.K = K
        # ... (rest of parameters are the same) ...
        self.C_dist = C_dist
        self.mu = mu
        self.sigma = sigma
        self.running = True
        self.K_C = K_C
        self.p_update_C = p_update_C
        self.p_mut = p_mut

        self.payoff_matrix = { # ... (no change) ...
            1: {1: (1, 1), 0: (0, self.b)},
            0: {1: (self.b, 0), 0: (0, 0)}
        }

        # --- Agent Initialization (No change needed) ---
        for _, pos in self.grid.coord_iter():
            # ... (no change) ...
            strategy = 1 if self.random.random() < initial_coop_ratio else 0
            C_value = self._generate_culture()
            agent = CulturalAgent(self.next_id(), self, strategy, C_value)
            self.grid.place_agent(agent, pos)
            self.schedule.add(agent)

        # --- Data Collection (No change needed) ---
        model_reporters={ # ... (no change) ...
            "CooperationRate": get_cooperation_rate,
            "AverageCulture": get_average_culture,
            "StdCulture": get_std_culture
        }
        try: # ... (no change) ...
            from utils.reporters import get_segregation_index, get_cooperation_rate_A, get_cooperation_rate_B
            model_reporters["SegregationIndex"] = get_segregation_index
            model_reporters["CoopRate_A"] = get_cooperation_rate_A
            model_reporters["CoopRate_B"] = get_cooperation_rate_B
        except ImportError:
            print("Warning: Segregation/Group Cooperation reporters not found.")
        self.datacollector = DataCollector(model_reporters=model_reporters)
        # --- End Data Collection Setup ---

    def _generate_culture(self):
        # --- (No change needed) ---
        # ... (same logic) ...
        if self.C_dist == "uniform": return self.random.uniform(0, 1)
        elif self.C_dist == "bimodal":
            if self.mu == 0.5: return self.random.choice([0.0, 1.0])
            else: return self.random.choice([0.0, 1.0], p=[1 - self.mu, self.mu])
        elif self.C_dist == "normal": return np.clip(self.random.normalvariate(self.mu, self.sigma), 0, 1)
        elif self.C_dist == "fixed": return self.mu
        else: raise ValueError(f"Unsupported C distribution type: {self.C_dist}")

    def step(self):
        """
        执行模型的一步。
        StagedActivation 调度器现在将按 stage_list 中定义的顺序，
        为所有智能体执行每个阶段对应的方法。
        """
        # The scheduler handles the multi-stage execution flow defined in __init__
        self.schedule.step()

        # Collect data after the final 'advance' stage is complete for all agents
        self.datacollector.collect(self)

    def run_model(self, n_steps):
        """运行模型 (No change needed)"""
        # ... (same logic) ...
        for i in range(n_steps):
            self.step()
