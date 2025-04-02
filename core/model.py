import numpy as np
from mesa import Model
from mesa.space import SingleGrid
from mesa.time import BaseScheduler
from mesa.datacollection import DataCollector
import pandas as pd

# Assuming your project structure allows these imports
# If not, adjust paths or place reporters directly in this file
from core.agent import CulturalAgent
from utils.reporters import get_cooperation_rate # Make sure this file exists or define the function here

# --- Define Reporter Functions (if not in utils.reporters) ---
def get_average_culture(model):
    """Calculates the average cultural value C across all agents."""
    if not model.schedule.agents:
        return 0
    return np.mean([agent.C for agent in model.schedule.agents])

def get_std_culture(model):
    """Calculates the standard deviation of cultural value C."""
    if len(model.schedule.agents) < 2:
        return 0
    return np.std([agent.C for agent in model.schedule.agents])
# --- End Reporter Functions ---


class CulturalGame(Model):
    """
    The main model for the spatial cultural game simulation
    with endogenous cultural evolution.
    """
    def __init__(self, L=50, initial_coop_ratio=0.5, b=1.5, K=0.5,
                 C_dist="uniform", mu=0.5, sigma=0.1, seed=None,
                 # --- New parameters for cultural evolution ---
                 K_C=0.1,           # Noise for cultural imitation (can differ from K)
                 p_update_C=0.1,    # Probability per agent to attempt cultural update each step
                 p_mut=0.001):      # Probability of cultural mutation per agent after potential update
        # --- End new parameters ---
        super().__init__(seed=seed)
        self.grid = SingleGrid(L, L, torus=True)
        self.schedule = BaseScheduler(self)
        self.L = L
        self.b = b
        self.K = K
        self.C_dist = C_dist
        self.mu = mu
        self.sigma = sigma
        self.running = True

        # --- Store new cultural evolution parameters ---
        self.K_C = K_C
        self.p_update_C = p_update_C
        self.p_mut = p_mut
        # --- End storing new parameters ---

        self.payoff_matrix = {
            1: {1: (1, 1), 0: (0, self.b)},
            0: {1: (self.b, 0), 0: (0, 0)}
        }

        # Initialize agents
        for _, pos in self.grid.coord_iter():
            strategy = 1 if self.random.random() < initial_coop_ratio else 0
            C_value = self._generate_culture()
            agent = CulturalAgent(self.next_id(), self, strategy, C_value)
            self.grid.place_agent(agent, pos)
            self.schedule.add(agent)
#            print(f"Initial C values: {[agent.C for agent in self.schedule.agents][:5]}")
        # Setup data collection - ADDED CULTURAL REPORTERS
        self.datacollector = DataCollector(
            model_reporters={
                "CooperationRate": get_cooperation_rate,
                "AverageCulture": get_average_culture, # New reporter
                "StdCulture": get_std_culture          # New reporter
            },
            # agent_reporters={"Strategy": "strategy", "Culture": "C"} # Keep commented unless needed
        )
        self.datacollector.collect(self)

    def _generate_culture(self):
        """Generates initial cultural parameter C."""
        # (Code remains the same as provided)
        if self.C_dist == "uniform":
            return self.random.uniform(0, 1)
        elif self.C_dist == "bimodal":
             # If mu intended as mean for bimodal [0,1]: 0.5 implies 50/50
             # If mu intended as prob of C=1: Use as is. Let's assume mean.
            if self.mu == 0.5:
                 return self.random.choice([0.0, 1.0]) # Corrected to float
            else: # Heuristic: bias towards closer value
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
        """
        Execute one simulation step with distinct phases for strategy and culture.
        1. Calculate Utilities: All agents calculate utility based on current state.
        2. Decide Strategy: All agents decide their next strategy.
        3. Decide Culture: All agents *potentially* decide their next culture.
        4. Advance: All agents update their strategy and culture simultaneously.
        5. Mutate Culture: Apply mutation to culture.
        6. Collect Data.
        """
        # Phase 1: Calculate utilities based on current strategies and cultures
        for agent in self.schedule.agents:
            agent.calculate_utility()
            
        # Phase 2: Decide next strategy based on calculated utilities
        for agent in self.schedule.agents:
            agent.decide_strategy_update() # Renamed method

        # Phase 3: Decide next culture based on calculated utilities and update probability
        for agent in self.schedule.agents:
            agent.decide_culture_update() # New method

        # Phase 4: Apply the updates for strategy and culture
        # The agent.advance() method will now update both
        self.schedule.step() # Calls agent.advance() for all agents

        # Phase 5: Apply cultural mutation AFTER advance
        for agent in self.schedule.agents:
            agent.mutate_culture() # New method

        for agent in self.schedule.agents:
            agent.advance() # New method

        # Phase 6: Collect data for this step
        self.datacollector.collect(self)




    def run_model(self, n_steps):
        """Run the model for n steps."""
        for i in range(n_steps):
            self.step()
            # Optional: Add progress print
            # if (i+1) % 10 == 0:
            #     print(f"Step {i+1}/{n_steps} complete.")
