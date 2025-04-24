# model.py
import numpy as np
from mesa import Model
from mesa.space import SingleGrid
from mesa.time import StagedActivation
from mesa.datacollection import DataCollector
import pandas as pd
from tqdm import tqdm
# --- Import Agents and Reporters ---
try:
    from core.agent import CulturalAgent
    from utils.reporters import (
        get_cooperation_rate,
        get_average_culture,
        get_std_culture,
        get_segregation_index,
        get_cooperation_rate_A,
        get_cooperation_rate_B,
        # --- Ensure NEW reporters are imported correctly ---
        get_cluster_size_distribution,
        get_boundary_fraction,
        get_boundary_coop_rate,
        get_bulk_coop_rate
    )
    REPORTERS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Error importing modules: {e}")
    print("Reporters might not be available. Using placeholders.")
    REPORTERS_AVAILABLE = False
    # Define dummy functions if imports fail
    def get_cooperation_rate(model): return np.nan
    def get_average_culture(model): return np.nan
    def get_std_culture(model): return np.nan
    def get_segregation_index(model): return np.nan
    def get_cooperation_rate_A(model): return np.nan
    def get_cooperation_rate_B(model): return np.nan
    def get_cluster_size_distribution(model): return {
        'A': [], 'B': []}  # Return empty dict

    def get_boundary_fraction(model): return np.nan
    def get_boundary_coop_rate(model): return np.nan
    def get_bulk_coop_rate(model): return np.nan


class CulturalGame(Model):
    """
    Cultural Game Model with Staged Activation and Enhanced Data Collection.
    """

    def __init__(self, L=50, initial_coop_ratio=0.5, b=1.5, K=0.1,
                 C_dist="uniform", mu=0.5, sigma=0.1, seed=None,
                 K_C=0.1, p_update_C=0.1, p_mut_culture=0.01, p_mut_strategy=0.001):

        super().__init__(seed=seed)
        self.random = np.random.default_rng(self._seed)
        self.grid = SingleGrid(L, L, torus=True)

        stage_list = [
            "calculate_utility",
            "decide_strategy_update",
            "decide_culture_update",
            "mutate",  # Ensure mutation happens before advance if it modifies next_C
            "advance"
        ]
        # NOTE: If mutate_culture modifies self.C directly, it should happen AFTER advance.
        # If it modifies self.next_C, it should happen BEFORE advance.
        # Let's assume it modifies self.next_C based on the agent code.
        self.schedule = StagedActivation(
            self, stage_list=stage_list, shuffle=True, shuffle_between_stages=False)

        self.L = L
        self.b = b
        self.K = K
        self.C_dist = C_dist
        self.mu = mu
        self.sigma = sigma
        self.running = True
        self.K_C = K_C
        self.p_update_C = p_update_C
        self.p_mut_culture = p_mut_culture   # Cultural mutation probability
        self.p_mut_strategy = p_mut_strategy  # Strategy mutation probability

        self.payoff_matrix = {
            # Note: Payoffs should be (my_payoff, neighbor_payoff) structure if matrix lookup does that. Usually it's just my_payoff. Let's assume it gives MY payoff.
            1: {1: (1, 1), 0: (0, b)},
            # Payoff when I am Row player, neighbor is Col player
            0: {1: (b, 0), 0: (0, 0)}
        }
        # Revisit agent utility calculation if payoff matrix interpretation is different.
        # The agent code calculates Utility = sum[(1-C)*my_payoff + C*neighbor_payoff].
        # Let's assume the payoff matrix gives (my_payoff, neighbor_payoff) for (my_strat, neighbor_strat).
        # If payoff matrix only gives my payoff, the agent code needs neighbor's payoff by reversing roles.
        # Example: my_payoff = self.model.payoff_matrix[self.strategy][neighbor.strategy]
        # neighbor_payoff = self.model.payoff_matrix[neighbor.strategy][self.strategy]
        # utility += (1-self.C)*my_payoff + self.C*neighbor_payoff
        # --> Let's assume the current agent code is correct and the payoff matrix provides tuples.

        for _, pos in self.grid.coord_iter():
            strategy = 1 if self.random.random() < initial_coop_ratio else 0
            C_value = self._generate_culture()
            agent = CulturalAgent(self.next_id(), self, strategy, C_value)
            self.grid.place_agent(agent, pos)
            self.schedule.add(agent)

        # --- Data Collection Setup ---
        model_reporters = {}
        if REPORTERS_AVAILABLE:
            model_reporters = {
                "CooperationRate": get_cooperation_rate,
                "AverageCulture": get_average_culture,
                "StdCulture": get_std_culture,
                "SegregationIndex": get_segregation_index,
                "CoopRate_A": get_cooperation_rate_A,
                "CoopRate_B": get_cooperation_rate_B,
                # --- Add NEW reporters ---
                # Returns dict {'A':[], 'B':[]}
                "ClusterSizeDistribution": get_cluster_size_distribution,
                "BoundaryFraction": get_boundary_fraction,
                "BoundaryCoopRate": get_boundary_coop_rate,
                "BulkCoopRate": get_bulk_coop_rate
            }
            print(
                f"DataCollector activated with reporters: {list(model_reporters.keys())}")
        else:
            print("DataCollector running with placeholder reporters.")
            model_reporters = {  # Use dummies if import failed
                "CooperationRate": get_cooperation_rate,
                "AverageCulture": get_average_culture,
                "StdCulture": get_std_culture,
                "SegregationIndex": get_segregation_index,
                "ClusterSizeDistribution": get_cluster_size_distribution,
                "BoundaryFraction": get_boundary_fraction,
                "BoundaryCoopRate": get_boundary_coop_rate,
                "BulkCoopRate": get_bulk_coop_rate
            }

        self.datacollector = DataCollector(model_reporters=model_reporters)
        # --- End Data Collection Setup ---

    def _generate_culture(self):
        # (No change needed)
        if self.C_dist == "uniform":
            return self.random.uniform(0, 1)
        elif self.C_dist == "bimodal":
            # Split based on mu. If mu=0.5, 50% 0.0, 50% 1.0. If mu=0.1, 90% 0.0, 10% 1.0
            return self.random.choice([0.0, 1.0], p=[1 - self.mu, self.mu])
        elif self.C_dist == "normal":
            # Use np.random.normal
            return np.clip(self.random.normal(self.mu, self.sigma), 0, 1)
        elif self.C_dist == "fixed":
            return self.mu
        else:
            raise ValueError(f"Unsupported C distribution type: {self.C_dist}")

    def step(self):
        """Executes one step of the model."""
        self.schedule.step()
        try:
            self.datacollector.collect(self)
        except Exception as e:
            print(
                f"Error during data collection at step {self.schedule.steps}: {e}")

    def run_model(self, n_steps):
        """Runs the model for n_steps."""
        print(
            f"Starting model run (L={self.L}, b={self.b}, K={self.K}, K_C={self.K_C}, steps={n_steps})...")
        for i in tqdm(range(n_steps), desc="Model Run", leave=False):
            self.step()
        print("Model run finished.")


# Example usage (remains the same)
if __name__ == '__main__':
    print("Testing CulturalGame model initialization...")
    try:
        model_instance = CulturalGame(L=10, steps=5)
        print("Model initialized successfully.")
        model_instance.run_model(2)
        print("Model ran 2 steps successfully.")
        if hasattr(model_instance, 'datacollector') and model_instance.datacollector.model_reporters:
            model_df = model_instance.datacollector.get_model_vars_dataframe()
            print("Collected data columns:", model_df.columns.tolist())
            print("Sample data:\n", model_df.tail())
        else:
            print("No valid reporters found for data collection.")

    except Exception as e:
        print(f"Error during model test: {e}")
        import traceback
        traceback.print_exc()
