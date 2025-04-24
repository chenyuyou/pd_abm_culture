# config.py
from dataclasses import dataclass, field
import numpy as np
import itertools
from typing import List, Dict, Any
from dataclasses import dataclass, field, fields  # Import fields


@dataclass
class SimConfig:
    # Simulation Parameters
    L: int = 50
    initial_coop_ratio: float = 0.5
    b: float = 1.5
    K: float = 0.1

    # Cultural Parameters (Initial Distribution)
    C_dist: str = "uniform"  # Options: "uniform", "normal", "bimodal", "fixed"
    # Meaning depends on C_dist (e.g., mean for normal, fixed value, p(C=1) for bimodal)
    mu: float = 0.5
    sigma: float = 0.1   # Std dev for normal distribution

    # Cultural Evolution Parameters
    K_C: float = 0.1         # Noise in cultural update rule
    p_update_C: float = 0.1  # Probability to attempt cultural update per step
    p_mut_culture: float = 0.01     # Probability of random cultural mutation per step
    p_mut_strategy: float = 0.001    # Probability of random strategy mutation per step

    steps: int = 500
    seed: int = None     # Random seed for reproducibility

    # Unique identifier for the parameter combination (optional but useful)
    param_set_id: str = ""

    # --- ADDED FIELD ---
    run_id: int = -1     # Index of the specific run for a parameter set

    # --- ADD THIS LINE ---
    # Descriptive label for the simulation set (e.g., 'C=0.1', 'Heterogeneous')
    label: str = ""
    # ---------------------

    # --- Methods for Batch Runs ---
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary, useful for logging."""
        # Use dataclasses.asdict for a more robust conversion if needed,
        # but vars() is often sufficient for simple cases.
        return vars(self)

    @classmethod
    def generate_param_sweep(cls, base_params: Dict[str, Any], sweep_params: Dict[str, List[Any]]) -> List['SimConfig']:
        """
        Generates a list of SimConfig objects for a parameter sweep.

        Args:
            base_params: Dictionary of fixed parameters.
            sweep_params: Dictionary where keys are parameter names to sweep,
                          and values are lists of values for that parameter.

        Returns:
            A list of SimConfig objects covering all combinations.
        """
        configs = []
        # Get the names and value lists for swept parameters
        sweep_keys = list(sweep_params.keys())
        sweep_values = list(sweep_params.values())

        # Generate all combinations of swept parameter values
        value_combinations = list(itertools.product(*sweep_values))

        # Create SimConfig for each combination
        for combo in value_combinations:
            current_params = base_params.copy()
            param_set_id_parts = []
            for i, key in enumerate(sweep_keys):
                value = combo[i]
                current_params[key] = value
                # Create a meaningful ID part (handle floats carefully)
                id_part = f"{key}{value:.3f}" if isinstance(
                    value, float) else f"{key}{value}"
                param_set_id_parts.append(id_part)

            # Assign a unique ID based on swept parameters
            current_params['param_set_id'] = "_".join(param_set_id_parts)

            # Create SimConfig, ensuring all fields are present
            config_fields = {f.name for f in fields(cls)}
            valid_params = {k: v for k,
                            v in current_params.items() if k in config_fields}
            # Add default values for any missing base parameters if necessary
            # for f in fields(cls):
            #     if f.name not in valid_params and f.default != field.MISSING:
            #         valid_params[f.name] = f.default
            #     elif f.name not in valid_params and f.default_factory != field.MISSING:
            #         valid_params[f.name] = f.default_factory()

            try:
                configs.append(cls(**valid_params))
            except TypeError as e:
                print(f"Error creating SimConfig with params: {valid_params}")
                print(f"Missing or unexpected arguments: {e}")
                # Handle error appropriately, e.g., skip this combo or raise

        return configs


# Example usage (remains the same)
if __name__ == '__main__':
    base = {
        "L": 50, "initial_coop_ratio": 0.5, "K": 0.1, "steps": 200,
        "sigma": 0.1, "K_C": 0.1, "p_update_C": 0.1, "p_mut": 0.001,
        # "param_set_id": "base_example" # ID is now generated automatically
    }
    sweep = {
        "b": np.linspace(1.1, 1.9, 3),
        "C_dist": ["uniform", "normal"],
        "mu": [0.3, 0.7],
        # "p_update_C": [0.05, 0.2] # Example sweep
    }

    config_list = SimConfig.generate_param_sweep(base, sweep)
    print(f"Generated {len(config_list)} configurations.")
    if config_list:
        print("First config:", config_list[0])
        # run_id will be -1
        print("First config dict:", config_list[0].to_dict())
