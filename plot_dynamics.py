# plot_dynamics.py

import sys
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns # Optional: for potentially nicer plot styling

# --- Path Setup (copy from main_batch.py) ---
# Ensure this works for your project structure
project_root = os.path.dirname(os.path.abspath(__file__))
# If plot_dynamics.py is in the root, project_root is correct.
# If it's in a subfolder (e.g., 'analysis'), adjust accordingly:
# project_root = os.path.dirname(project_root) # Go up one level if needed
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# --- End Path Setup ---

# Now import your model components
try:
    from core.model import CulturalGame
    # Reporters might be defined within model.py or imported separately
    # If defined in model.py, you don't need to import them here explicitly
    # from utils.reporters import get_average_culture, get_std_culture, get_cooperation_rate
except ImportError as e:
    print(f"Error importing modules. Check PYTHONPATH and file structure: {e}")
    print(f"Project root used: {project_root}")
    sys.exit(1)

if __name__ == "__main__":
    print("--- Running Single Simulation for Dynamics Plot ---")

    # 1. Define Simulation Parameters for THIS RUN
    sim_params = {
        "L": 50,
        "initial_coop_ratio": 0.1,
        "b": 1.5,
        "K": 1,
        "steps": 2500,              # <<< Keep steps here, we need it later
        "C_dist": "fixed",
        "mu": 0.0,
        "sigma": 0.001,
        "K_C": 1.5,
        "p_update_C": 0.01,
        "p_mut": 0.1,
        "seed": 422
    }
    print("Using parameters:")
    for key, value in sim_params.items():
        print(f"  {key}: {value}")

    # 2. Instantiate and Run the Model
    start_time = time.time()

    # --- CORRECTED PART ---
    # Create a temporary dictionary containing only the parameters
    # that CulturalGame.__init__ actually accepts.
    model_init_params = {k: v for k, v in sim_params.items() if k != 'steps'}

    # Pass only the valid initialization parameters
    model = CulturalGame(**model_init_params)
    # --- END CORRECTION ---

    print(f"\nStarting model run for {sim_params['steps']} steps...")
    # Use the 'steps' value from the original dictionary here
    model.run_model(sim_params['steps'])
    end_time = time.time()
    print(f"Model run duration: {end_time - start_time:.2f} seconds")

    # 3. Get the Time Series Data
    model_df = model.datacollector.get_model_vars_dataframe()

    if model_df.empty:
        print("Model run completed but the datacollector DataFrame is empty.")
        sys.exit(1)

    # 4. Prepare Data for Plotting
    steps = model_df.index # Use the DataFrame index for steps
    try:
        coop_rate_ts = model_df['CooperationRate']
        avg_culture_ts = model_df['AverageCulture']
        std_culture_ts = model_df['StdCulture']
    except KeyError as e:
        print(f"Error: Column missing from datacollector results: {e}")
        print(f"Available columns: {model_df.columns.tolist()}")
        sys.exit(1)

    # 5. Create the Plot
    print("\nGenerating plot...")
    try:
        fig, ax1 = plt.subplots(figsize=(12, 7)) # Use ax1 for the primary axis

        # Plot Average Culture and Std Dev on primary axis (ax1)
        color1 = 'tab:blue'
        ax1.set_xlabel('Simulation Steps')
        ax1.set_ylabel('Average Culture', color=color1)
        line1, = ax1.plot(steps, avg_culture_ts, color=color1, linewidth=2, label='Average Culture')
        fill1 = ax1.fill_between(steps,
                                 avg_culture_ts - std_culture_ts,
                                 avg_culture_ts + std_culture_ts,
                                 color=color1, alpha=0.2, label='Std Dev Culture')
        ax1.tick_params(axis='y', labelcolor=color1)
        ax1.set_ylim(-0.05, 1.05)
        ax1.grid(True, axis='y', linestyle=':', color=color1, alpha=0.5) # Grid for primary axis

        # Create secondary axis (ax2) sharing the same x-axis
        ax2 = ax1.twinx()
        color2 = 'tab:red'
        ax2.set_ylabel('Cooperation Rate', color=color2)
        line2, = ax2.plot(steps, coop_rate_ts, color=color2, linestyle='--', linewidth=1.5, label='Cooperation Rate')
        ax2.tick_params(axis='y', labelcolor=color2)
        ax2.set_ylim(-0.05, 1.05)

        # Title and Legend
        plt.title(f'Dynamics of Culture and Cooperation\n'
                  f'(b={sim_params["b"]}, K={sim_params["K"]}, K_C={sim_params["K_C"]}, '
                  f'p_up_C={sim_params["p_update_C"]}, p_mut={sim_params["p_mut"]})',
                  fontsize=12) # Smaller font for title if long
        # Combine legends from both axes
        handles = [line1, fill1, line2]
        labels = [h.get_label() for h in handles]
        ax1.legend(handles, labels, loc='center right') # Place legend based on ax1

        fig.tight_layout() # Adjust layout to prevent overlap

        # 6. Save the Plot
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        plot_filename = f"simulation_dynamics_{timestamp}.png"
        plt.savefig(plot_filename)
        print(f"Dynamics plot saved to {plot_filename}")
        # plt.show() # Uncomment to display the plot interactively
        plt.close() # Close the plot window

    except Exception as e:
        print(f"Error during plotting: {e}")
        import traceback
        traceback.print_exc()

    print("\n--- Script Finished ---")
