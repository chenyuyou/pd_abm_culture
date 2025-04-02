# <ATTACHMENT_FILE>
# <FILE_INDEX>File 1</FILE_INDEX>
# <FILE_NAME>main_batch.py</FILE_NAME>
# <FILE_CONTENT>
import sys
import os
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# --- Path Setup (ensure this works for your project structure) ---
# Add the project root directory to the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# --- End Path Setup ---

# Now your regular imports should work using absolute paths
try:
    from utils.parallel import batch_run_parallel
    from utils.config import SimConfig
except ImportError as e:
    print(f"Error importing modules. Check PYTHONPATH and file structure: {e}")
    sys.exit(1)


if __name__ == "__main__":
    print("Setting up batch simulation for Co-evolution Model...")
    start_batch_time = time.time()

    # 1. Define Base Configuration and Sweep Parameters
    base_params = {
        "L": 50,
        "initial_coop_ratio": 0.1,
        "K": 1,               # Strategy noise
        "steps": 500,          # Simulation steps per run (increase if needed for convergence)
        "C_dist": "uniform",    # Initial cultural distribution
        "mu": 0.5,              # Initial mean (relevant for normal/bimodal/fixed)
        "sigma": 0.2,           # Initial std dev (relevant for normal)
        # --- Default Cultural Evolution Parameters ---
        "K_C": 15,             # Cultural noise (default same as K)
        "p_update_C": 0.1,      # Cultural update probability
        "p_mut": 0.0001,         # Cultural mutation rate
        # --- Seed ---
        "seed": None            # Set a base seed if strict reproducibility across the whole batch is needed,
                                # but parallel runs might need individual seeding (handled in parallel.py if seed is None)
    }

    # --- Define Parameters to Sweep ---
    sweep_params = {
        "b": np.linspace(1.0, 5, 9),       # Sweep temptation parameter 'b' (adjust range/steps)
        "p_update_C": [0.01, 0.1, 1.0],      # Sweep cultural update speed (slow, medium, fast=strategy speed)
        # Example: Sweep cultural noise relative to strategy noise
        "K_C": [base_params["K"] * 0.5, base_params["K"], base_params["K"] * 2.0],
        # Example: Fixed mutation rate for this sweep
        # "p_mut": [0.001, 0.01], # Could sweep mutation too
    }

    # 2. Generate the list of configurations
    config_list = SimConfig.generate_param_sweep(base_config=base_params,
                                                 sweep_params=sweep_params)
    print(f"Generated {len(config_list)} simulation configurations.")

    # Optional: Filter specific configurations if needed
    # config_list = [c for c in config_list if ...]

    # 3. Run the Batch Simulation in Parallel
    # Use num_workers=None to auto-detect, or specify a number e.g., num_workers=4
    # steady_state_window defines how many final steps to average for the result
    results_df = batch_run_parallel(config_list, num_workers=None, steady_state_window=200) # Use larger window maybe

    if results_df.empty:
        print("Batch run finished but produced no results. Exiting.")
        sys.exit(1)

    # 4. Save the Results
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    results_filename = f"coevo_game_results_{timestamp}.csv"
    try:
        results_df.to_csv(results_filename, index=False)
        print(f"Results saved to {results_filename}")
    except Exception as e:
        print(f"Error saving results to CSV: {e}")

    # 5. Basic Analysis and Visualization (Updated Examples)
    print("\n--- Basic Analysis ---")
    # Display potentially useful columns
    display_cols = [p for p in sweep_params.keys()] + \
                   ['avg_CooperationRate', 'avg_AverageCulture', 'avg_StdCulture', 'runtime_seconds']
    # Filter out columns not present just in case
    display_cols = [c for c in display_cols if c in results_df.columns]
    print(results_df[display_cols].head())
    print(f"\nAverage Runtime per Simulation: {results_df['runtime_seconds'].mean():.2f} seconds")

    # --- Example Plot 1: Cooperation Rate vs. 'b' for different Cultural Update Speeds (p_update_C) ---
    try:
        # Choose a fixed value for other swept params if needed for clarity
        fixed_Kc = base_params["K"] # Example: Plot for Kc = K
        plot_df_1 = results_df[np.isclose(results_df['K_C'], fixed_Kc)] # Use isclose for float comparison

        if not plot_df_1.empty:
            plt.figure(figsize=(10, 6))
            sns.lineplot(data=plot_df_1, x='b', y='avg_CooperationRate', hue='p_update_C', marker='o', errorbar=('ci', 95)) # Show confidence interval
            plt.title(f'Avg Cooperation Rate vs. Temptation (b)\n(Strategy K={base_params["K"]}, Culture K_C={fixed_Kc})')
            plt.xlabel('Temptation (b)')
            plt.ylabel('Average Cooperation Rate')
            plt.grid(True)
            plt.ylim(-0.05, 1.05)
            plt.legend(title='Culture Update Prob (p_update_C)')
            plot_filename_1 = f"coop_vs_b_by_Pup_{timestamp}.png"
            plt.savefig(plot_filename_1)
            print(f"Plot 1 saved to {plot_filename_1}")
            plt.close() # Close figure to prevent display if not interactive
        else:
             print(f"No data found for K_C={fixed_Kc} to generate plot 1.")

    except Exception as e:
        print(f"Could not generate plot 1: {e}")


    # --- Example Plot 2: Average Evolved Culture vs. 'b' for different Cultural Noises (K_C) ---
    try:
        # Choose a fixed value for other swept params if needed
        fixed_Pup = 0.1 # Example: Plot for p_update_C = 0.1
        plot_df_2 = results_df[np.isclose(results_df['p_update_C'], fixed_Pup)]

        if not plot_df_2.empty:
             plt.figure(figsize=(10, 6))
             sns.lineplot(data=plot_df_2, x='b', y='avg_AverageCulture', hue='K_C', marker='^', linestyle='--', errorbar=('ci', 95))
#             plt.title(f'Average Evolved Culture ($\overline{{C}}$) vs. Temptation (b)\n(Strategy K={base_params["K"]}, p_update_C={fixed_Pup})')
             plt.xlabel('Temptation (b)')
#             plt.ylabel('Average Culture ($\overline{C}$)')
             plt.grid(True)
             plt.ylim(-0.05, 1.05) # Culture is between 0 and 1
             plt.legend(title='Culture Noise (K_C)')
             plot_filename_2 = f"culture_vs_b_by_Kc_{timestamp}.png"
             plt.savefig(plot_filename_2)
             print(f"Plot 2 saved to {plot_filename_2}")
             plt.close()
        else:
             print(f"No data found for p_update_C={fixed_Pup} to generate plot 2.")

    except Exception as e:
         print(f"Could not generate plot 2: {e}")

    # --- Example Plot 3: Heatmap of Cooperation vs. b and p_update_C ---
    try:
        fixed_Kc_heatmap = base_params["K"] # Choose a fixed Kc for the heatmap
        plot_df_heatmap = results_df[np.isclose(results_df['K_C'], fixed_Kc_heatmap)]

        if not plot_df_heatmap.empty and len(plot_df_heatmap['p_update_C'].unique()) > 1 and len(plot_df_heatmap['b'].unique()) > 1:
             # Check if 'avg_CooperationRate' exists before pivoting
             if 'avg_CooperationRate' not in plot_df_heatmap.columns:
                 raise ValueError("'avg_CooperationRate' column not found for heatmap.")

             pivot_table = plot_df_heatmap.pivot_table(index='p_update_C', columns='b', values='avg_CooperationRate', aggfunc='mean')
             plt.figure(figsize=(12, 7))
             sns.heatmap(pivot_table, annot=True, fmt=".2f", cmap="viridis", linewidths=.5)
             plt.title(f'Avg Cooperation Rate (K_C={fixed_Kc_heatmap:.2f})')
             plt.xlabel('Temptation (b)')
             plt.ylabel('Culture Update Prob (p_update_C)')
             heatmap_filename = f"heatmap_coop_vs_b_Pup_{timestamp}.png"
             plt.savefig(heatmap_filename)
             print(f"Heatmap plot saved to {heatmap_filename}")
             plt.close()
        elif plot_df_heatmap.empty:
             print(f"No data found for K_C={fixed_Kc_heatmap} to generate heatmap.")
        else:
             print("Insufficient variation in 'p_update_C' or 'b' for the selected K_C to generate heatmap.")


    except Exception as e:
         print(f"Could not generate heatmap: {e}")


    end_batch_time = time.time()
    print(f"\nTotal batch simulation time: {(end_batch_time - start_batch_time)/60:.2f} minutes")
    print("-----------------------")

# </FILE_CONTENT>
# </ATTACHMENT_FILE>
