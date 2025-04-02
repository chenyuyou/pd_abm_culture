# <ATTACHMENT_FILE>
# <FILE_INDEX>File 4</FILE_INDEX>
# <FILE_NAME>config.py</FILE_NAME>
# <FILE_CONTENT>
from dataclasses import dataclass, field
import numpy as np
import itertools
from typing import List, Dict, Any

@dataclass
class SimConfig:
    # Simulation Parameters
    L: int = 50
    initial_coop_ratio: float = 0.5
    b: float = 1.5
    K: float = 0.1

    # Cultural Parameters (Initial Distribution)
    C_dist: str = "uniform"
    mu: float = 0.5      # Relevant for normal, bimodal, fixed initial distributions
    sigma: float = 0.1   # Relevant for normal initial distribution

    # --- New Cultural Evolution Parameters ---
    K_C: float = 0.1     # Noise for cultural imitation (default same as K)
    p_update_C: float = 0.1 # Probability per agent to attempt cultural update each step
    p_mut: float = 0.001    # Probability of cultural mutation per agent after potential update
    # --- End New Parameters ---

    steps: int = 500     # Number of steps per simulation run
    seed: int = None     # Random seed for reproducibility

    # --- Methods for Batch Runs ---
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary, useful for logging."""
        # Use vars() for dataclasses to get all fields as a dict
        return vars(self)

    @staticmethod
    def generate_param_sweep(base_config: Dict[str, Any],
                             sweep_params: Dict[str, List]) -> List['SimConfig']:
        """
        Generates a list of SimConfig objects for a parameter sweep.

        Args:
            base_config: A dictionary with default/fixed parameter values.
            sweep_params: A dictionary where keys are parameter names to sweep,
                          and values are lists of values for that parameter.

        Returns:
            A list of SimConfig objects, one for each parameter combination.
        """
        configs = []
        # Get keys and value lists for sweeping
        param_names = list(sweep_params.keys())
        value_lists = list(sweep_params.values())

        # Generate all combinations of sweep parameters
        for values_combination in itertools.product(*value_lists):
            current_params = base_config.copy()
            # Update the current config with the specific values for this run
            for name, value in zip(param_names, values_combination):
                current_params[name] = value

            # Create SimConfig object using all current parameters
            # dataclasses handle field matching automatically
            try:
                configs.append(SimConfig(**current_params))
            except TypeError as e:
                print(f"Error creating SimConfig with params: {current_params}")
                print(f"Missing or unexpected parameter? Error: {e}")
                # Handle error appropriately, maybe skip or raise

        return configs

# Example usage (can be moved to main_batch.py)
if __name__ == '__main__':
    base = {
        "L": 50, "initial_coop_ratio": 0.5, "K": 0.1, "steps": 200,
        "sigma": 0.1, "K_C": 0.1, "p_update_C": 0.1, "p_mut": 0.001 # Added new params
    }
    sweep = {
        "b": np.linspace(1.1, 1.9, 3), # Reduced size for faster example
        "C_dist": ["uniform", "normal"],
        "mu": [0.3, 0.7], # Relevant for initial condition
        "p_update_C": [0.05, 0.2] # Example sweep for new param
    }

    config_list = SimConfig.generate_param_sweep(base, sweep)
    print(f"Generated {len(config_list)} configurations.")
    if config_list:
        print("First config:", config_list[0])
        print("Last config:", config_list[-1])
        print("Example dict:", config_list[0].to_dict())

# </FILE_CONTENT>
# </ATTACHMENT_FILE>
