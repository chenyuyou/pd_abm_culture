# parallel.py
import multiprocessing as mp
from typing import Any, Dict, List
from tqdm import tqdm
import pandas as pd
import time
from functools import partial
import numpy as np

# Adjust imports if your directory structure differs
try:
    from core.model import CulturalGame
    from utils.config import SimConfig
except ImportError:
    import sys
    import os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from core.model import CulturalGame
    from utils.config import SimConfig


def run_single_simulation(config: SimConfig, steady_state_window=100) -> Dict[str, Any]:
    """
    Runs a single simulation, calculating steady-state averages for most reporters,
    but taking the *last value* for specified non-averageable reporters like
    ClusterSizeDistribution.

    Args:
        config: SimConfig object.
        steady_state_window: Window for averaging standard reporters.

    Returns:
        Dictionary with parameters and results.
    """
    start_time = time.time()
    run_seed = config.seed

    # Define reporters that should NOT be averaged (take last value)
    # Ensure the names match the keys in model.datacollector.model_reporters
    NON_AVERAGE_REPORTERS = ["ClusterSizeDistribution"]

    model = CulturalGame(
        L=config.L, initial_coop_ratio=config.initial_coop_ratio, b=config.b, K=config.K,
        C_dist=config.C_dist, mu=config.mu, sigma=config.sigma, seed=run_seed,
        K_C=config.K_C, p_update_C=config.p_update_C, p_mut=config.p_mut
    )

    # Run the model (silence tqdm progress bar within parallel runs if desired)
    # model.run_model(config.steps) # Assumes run_model doesn't use tqdm or is managed
    for _ in range(config.steps): # Manual loop to avoid nested tqdm
        model.step()


    model_df = model.datacollector.get_model_vars_dataframe()
    results = {}
    n_rows = len(model_df)

    if n_rows > 0:
        # Determine the window for averaging
        start_idx = max(0, n_rows - steady_state_window)
        window_df = model_df.iloc[start_idx:]

        for col in model.datacollector.model_reporters.keys():
            if col in model_df.columns:
                if col in NON_AVERAGE_REPORTERS:
                    # --- Take the LAST value for specific reporters ---
                    results[f"{col}"] = model_df[col].iloc[-1] # Store raw last value
                    # Keep a consistent naming? Or make it clear it's not averaged?
                    # Let's use the plain name for now, processing script will handle it.
                    # results[f"last_{col}"] = model_df[col].iloc[-1] # Alternative naming
                elif not window_df.empty:
                    # --- Calculate Averages for standard reporters ---
                    results[f"avg_{col}"] = window_df[col].mean()
                    results[f"std_{col}"] = window_df[col].std()
                else: # Should not happen if n_rows > 0, but for safety
                    results[f"avg_{col}"] = np.nan
                    results[f"std_{col}"] = np.nan
                    if col in NON_AVERAGE_REPORTERS:
                        results[f"{col}"] = None # Or appropriate null value like {} or []
            else:
                # Handle case where reporter column wasn't created (e.g., error in reporter func)
                 print(f"Warning: Reporter column '{col}' not found in model_df for config {config.param_set_id}, run {config.run_id}.")
                 results[f"avg_{col}"] = np.nan
                 results[f"std_{col}"] = np.nan
                 if col in NON_AVERAGE_REPORTERS:
                      results[f"{col}"] = None

    else: # No data collected
        for col in model.datacollector.model_reporters.keys():
             results[f"avg_{col}"] = np.nan
             results[f"std_{col}"] = np.nan
             if col in NON_AVERAGE_REPORTERS:
                  results[f"{col}"] = None


    end_time = time.time()

    # Combine config and results
    result_dict = config.to_dict()
    result_dict.update(results) # Add calculated results
    result_dict["runtime_seconds"] = end_time - start_time

    return result_dict


def batch_run_parallel(config_list: List[SimConfig],
                         num_workers: int = None,
                         steady_state_window: int = 100) -> pd.DataFrame:
    """Runs simulations in parallel."""
    if num_workers is None:
        cpu_cores = mp.cpu_count()
        num_workers = max(1, cpu_cores - 1 if cpu_cores > 1 else 1)

    print(f"Starting batch run with {len(config_list)} configurations using {num_workers} workers...")

    run_func = partial(run_single_simulation, steady_state_window=steady_state_window)

    pool = mp.Pool(processes=num_workers)
    results = []
    try:
        results = list(tqdm(pool.imap_unordered(run_func, config_list), total=len(config_list), desc="Simulations"))
    except Exception as e:
         print(f"\n--- Error during parallel execution ---")
         print(e)
         import traceback
         traceback.print_exc()
         print("--- Trying to collect partial results ---")
    finally:
        pool.close()
        pool.join()

    print(f"Batch run completed. Collected {len(results)} results.")
    if not results:
        return pd.DataFrame()
    else:
        # Convert results to DataFrame, handle potential missing columns carefully
        return pd.DataFrame(results)

# Example Usage (if needed for testing parallel.py itself)
if __name__ == '__main__':
    print("Testing parallel execution...")
    test_configs = [
        SimConfig(L=10, b=1.2, steps=20, seed=101, run_id=0, param_set_id="test1"),
        SimConfig(L=10, b=1.8, steps=20, seed=102, run_id=0, param_set_id="test2"),
        SimConfig(L=15, b=1.2, steps=20, seed=103, run_id=0, param_set_id="test3"),
        SimConfig(L=15, b=1.8, steps=20, seed=104, run_id=0, param_set_id="test4"),
    ]
    results_df = batch_run_parallel(test_configs, num_workers=2, steady_state_window=5)

    print("\n--- Batch Run Results ---")
    if not results_df.empty:
        print(results_df.head())
        print("\nColumns:", results_df.columns.tolist())
        # Check if cluster data looks like a dictionary
        if 'ClusterSizeDistribution' in results_df.columns:
            print("\nSample ClusterSizeDistribution entry:")
            print(results_df['ClusterSizeDistribution'].iloc[0])
        else:
            print("\nClusterSizeDistribution column not found in results.")
    else:
        print("No results dataframe generated.")
