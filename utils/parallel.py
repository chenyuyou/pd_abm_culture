# <ATTACHMENT_FILE>
# <FILE_INDEX>File 3</FILE_INDEX>
# <FILE_NAME>parallel.py</FILE_NAME>
# <FILE_CONTENT>
import multiprocessing as mp
from typing import Any, Dict, List
from tqdm import tqdm
import pandas as pd
import time # Optional: for timing runs
from functools import partial
import numpy as np # Import numpy for mean/std calculation

# Adjust imports if your directory structure differs
try:
    from core.model import CulturalGame
    from utils.config import SimConfig
except ImportError:
    # Attempt relative import if run as script within utils or similar context
    import sys
    import os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from core.model import CulturalGame
    from utils.config import SimConfig


def run_single_simulation(config: SimConfig, steady_state_window=100) -> Dict[str, Any]:
    """
    Runs a single simulation instance based on the provided config,
    now including cultural evolution parameters and metrics.

    Args:
        config: A SimConfig object containing all parameters for the run.
        steady_state_window: Number of final steps to average for results.

    Returns:
        A dictionary containing the input parameters and averaged results.
    """
    start_time = time.time()

    # Handle seeding if needed (current implementation uses config.seed directly)
    run_seed = config.seed

    # --- Pass New Parameters to Model ---
    model = CulturalGame(
        L=config.L,
        initial_coop_ratio=config.initial_coop_ratio,
        b=config.b,
        K=config.K,
        C_dist=config.C_dist,
        mu=config.mu,
        sigma=config.sigma,
        seed=run_seed, # Pass the specific seed for this run
        # Pass cultural evolution parameters
        K_C=config.K_C,
        p_update_C=config.p_update_C,
        p_mut=config.p_mut
    )
    # --- End Passing New Parameters ---

    model.run_model(config.steps)

    # Get results dataframe from datacollector
    model_df = model.datacollector.get_model_vars_dataframe()

    # Calculate average results over the last 'steady_state_window' steps
    avg_results = {}
    n_rows = len(model_df)

    if n_rows >= steady_state_window:
        window_df = model_df.iloc[-steady_state_window:]
    elif n_rows > 0:
        window_df = model_df.iloc[-n_rows:] # Avg over available steps
    else:
        window_df = pd.DataFrame() # Empty dataframe if no steps run

    # --- Calculate Averages for All Reporters ---
    # Reporters included: "CooperationRate", "AverageCulture", "StdCulture"
    for col in model.datacollector.model_reporters.keys():
        if not window_df.empty and col in window_df.columns:
            avg_results[f"avg_{col}"] = window_df[col].mean()
            # Optionally calculate std dev within the window too
            avg_results[f"std_{col}"] = window_df[col].std()
        else:
            avg_results[f"avg_{col}"] = None
            avg_results[f"std_{col}"] = None
    # --- End Calculating Averages ---

    end_time = time.time()

    # Return a dictionary combining config and results
    result_dict = config.to_dict() # Get all input parameters
    result_dict.update(avg_results) # Add calculated averages
    result_dict["runtime_seconds"] = end_time - start_time

    return result_dict


def batch_run_parallel(config_list: List[SimConfig],
                         num_workers: int = None,
                         steady_state_window: int = 100) -> pd.DataFrame:
    """
    Runs simulations in parallel for a list of configurations.

    Args:
        config_list: A list of SimConfig objects.
        num_workers: Number of parallel processes. Defaults to cpu_count() - 1.
        steady_state_window: Window for calculating average results.

    Returns:
        A pandas DataFrame containing parameters and results for all runs.
    """
    if num_workers is None:
        cpu_cores = mp.cpu_count()
        num_workers = max(1, cpu_cores - 1 if cpu_cores > 1 else 1) # Use available cores safely

    print(f"Starting batch run with {len(config_list)} configurations using {num_workers} workers...")

    # Use functools.partial to pass the fixed steady_state_window arg
    run_func = partial(run_single_simulation, steady_state_window=steady_state_window)

    # Use try-finally to ensure pool closure
    pool = mp.Pool(processes=num_workers)
    results = []
    try:
        # Use imap_unordered for potentially faster processing and tqdm for progress
        results = list(tqdm(pool.imap_unordered(run_func, config_list), total=len(config_list), desc="Simulations"))
    finally:
        pool.close() # Close the pool pool
        pool.join()  # Wait for worker processes to exit

    print("Batch run completed.")
    if not results:
        print("Warning: No results were generated.")
        return pd.DataFrame() # Return empty DataFrame
    else:
        return pd.DataFrame(results)

# </FILE_CONTENT>
# </ATTACHMENT_FILE>
