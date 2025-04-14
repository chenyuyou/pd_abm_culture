import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import time
from tqdm import tqdm # For progress bars
import pickle # For saving/loading data

# --- Assuming your project structure allows these imports ---
# If not, adjust sys.path as needed
try:
    from core.model import CulturalGame
    from utils.reporters import get_cooperation_rate # Import others if needed directly
    # Make sure the modified reporters are available
except ImportError as e:
    print(f"Error importing core modules: {e}")
    print("Ensure model.py and reporters.py (with modifications) are accessible.")
    exit()
# ---

# ==============================================================================
# Simulation Parameters
# ==============================================================================
PARAMS = {
    'L': 50,
    'initial_coop_ratio': 0.5,
    'K': 0.1,           # Strategy noise
    'steps': 500,       # Simulation steps per run
    'steady_state_window': 100, # Steps at the end to average over
    'runs_per_b': 5,    # Number of independent runs for each 'b' value to average
    'b_values': np.linspace(1.0, 2.0, 11), # Range of temptation 'b'

    # --- Baseline Specific ---
    'baseline_C_values': {'Individualist (C=0.1)': 0.1, 'Collectivist (C=0.9)': 0.9},

    # --- Heterogeneous Specific ---
    'hetero_C_dist': 'bimodal', # Initial distribution for Figs 3, 4, 5
    'hetero_mu': 0.5,         # Initial mix (e.g., 50/50 for bimodal)
    'hetero_sigma': 0.1,      # (Not used for bimodal, relevant if 'normal')
    'hetero_K_C': 0.1,        # Cultural noise
    'hetero_p_update_C': 0.1, # Cultural update probability
    'hetero_p_mut': 0.001,    # Cultural mutation rate

    # --- Snapshot Specific ---
    'snapshot_b_values': [1.1, 1.6], # Example b values for snapshots
    'snapshot_hetero_params': { # Use hetero params for snapshot runs
         'C_dist': 'bimodal', 'mu': 0.5, 'sigma': 0.1,
         'K_C': 0.1, 'p_update_C': 0.1, 'p_mut': 0.001
    }
}

DATA_SAVE_DIR = "simulation_data" # Directory to save intermediate data
PLOT_SAVE_DIR = "plots"           # Directory to save generated plots

os.makedirs(DATA_SAVE_DIR, exist_ok=True)
os.makedirs(PLOT_SAVE_DIR, exist_ok=True)

# ==============================================================================
# Simulation Running Functions
# ==============================================================================

def run_single_sim(params, run_id=0, seed_offset=0):
    """Runs one simulation instance and returns averaged results."""
    model_seed = None if params.get('seed', None) is None else params['seed'] + seed_offset + run_id

    model = CulturalGame(
        L=params['L'],
        initial_coop_ratio=params['initial_coop_ratio'],
        b=params['b'],
        K=params['K'],
        C_dist=params['C_dist'],
        mu=params['mu'],
        sigma=params['sigma'],
        seed=model_seed,
        K_C=params.get('K_C', 0), # Default to 0 if not provided (baseline)
        p_update_C=params.get('p_update_C', 0), # Default to 0
        p_mut=params.get('p_mut', 0)           # Default to 0
    )
    model.run_model(params['steps'])
    model_df = model.datacollector.get_model_vars_dataframe()

    # Calculate average results over the steady state window
    avg_results = {}
    n_rows = len(model_df)
    window = params['steady_state_window']
    if n_rows >= window:
        window_df = model_df.iloc[-window:]
    elif n_rows > 0:
        window_df = model_df.iloc[-n_rows:]
    else:
        window_df = pd.DataFrame()

    # Dynamically average all collected reporters
    for col in model.datacollector.model_reporters.keys():
        if not window_df.empty and col in window_df.columns:
            avg_results[f"avg_{col}"] = window_df[col].mean()
            avg_results[f"std_{col}"] = window_df[col].std() # Stdev within the window
        else:
            avg_results[f"avg_{col}"] = np.nan
            avg_results[f"std_{col}"] = np.nan

    # Combine input params and results
    result_dict = params.copy()
    result_dict.update(avg_results)
    result_dict['run_id'] = run_id
    return result_dict


def run_simulation_set(config_func, filename_tag):
    """Runs a set of simulations defined by config_func and saves/loads data."""
    data_filename = os.path.join(DATA_SAVE_DIR, f"data_{filename_tag}.pkl")

    if os.path.exists(data_filename):
        print(f"Loading existing data from {data_filename}...")
        with open(data_filename, 'rb') as f:
            all_results_df = pickle.load(f)
    else:
        print(f"Running simulations for {filename_tag}...")
        all_results = []
        param_list = config_func() # Get list of parameter dicts for this set
        with tqdm(total=len(param_list) * PARAMS['runs_per_b'], desc=f"Sims ({filename_tag})") as pbar:
            for i, sim_params in enumerate(param_list):
                for run_idx in range(PARAMS['runs_per_b']):
                    result = run_single_sim(sim_params, run_id=run_idx, seed_offset=i*100) # Offset seeds
                    all_results.append(result)
                    pbar.update(1)

        all_results_df = pd.DataFrame(all_results)
        with open(data_filename, 'wb') as f:
            pickle.dump(all_results_df, f)
        print(f"Simulation data saved to {data_filename}")

    return all_results_df

# --- Configuration Generators ---
def generate_baseline_configs():
    configs = []
    for label, c_val in PARAMS['baseline_C_values'].items():
        for b in PARAMS['b_values']:
            params = {
                'L': PARAMS['L'], 'initial_coop_ratio': PARAMS['initial_coop_ratio'],
                'K': PARAMS['K'], 'steps': PARAMS['steps'], 'steady_state_window': PARAMS['steady_state_window'],
                'b': b, 'C_dist': 'fixed', 'mu': c_val, 'sigma': 0, # Fixed C
                'K_C': 0, 'p_update_C': 0, 'p_mut': 0, # No evolution
                'label': label # Add label for plotting
            }
            configs.append(params)
    return configs

def generate_heterogeneous_configs():
    configs = []
    for b in PARAMS['b_values']:
        params = {
            'L': PARAMS['L'], 'initial_coop_ratio': PARAMS['initial_coop_ratio'],
            'K': PARAMS['K'], 'steps': PARAMS['steps'], 'steady_state_window': PARAMS['steady_state_window'],
            'b': b, 'C_dist': PARAMS['hetero_C_dist'], 'mu': PARAMS['hetero_mu'], 'sigma': PARAMS['hetero_sigma'],
            'K_C': PARAMS['hetero_K_C'], 'p_update_C': PARAMS['hetero_p_update_C'], 'p_mut': PARAMS['hetero_p_mut'],
            'label': 'Heterogeneous Mix'
        }
        configs.append(params)
    return configs

# ==============================================================================
# Plotting Functions
# ==============================================================================

def plot_fig1(baseline_data, filename="plot_fig1_baseline_coop_vs_b.png"):
    """Plots Fig 1: Baseline Homogeneous Cooperation."""
    print("Plotting Fig 1: Baseline Cooperation...")
    if 'avg_CooperationRate' not in baseline_data.columns:
         print("Error plotting Fig 1: 'avg_CooperationRate' not found in baseline data.")
         return
    if 'label' not in baseline_data.columns:
         print("Error plotting Fig 1: 'label' column missing in baseline data.")
         return

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=baseline_data, x='b', y='avg_CooperationRate', hue='label',
                 marker='o', errorbar=('ci', 95)) # Aggregates runs_per_b automatically
    plt.title('Fig 1: Baseline Homogeneous Culture Cooperation vs. Temptation (b)')
    plt.xlabel('Temptation (b)')
    plt.ylabel('Average Cooperation Rate')
    plt.grid(True)
    plt.ylim(-0.05, 1.05)
    plt.legend(title='Culture Type')
    save_path = os.path.join(PLOT_SAVE_DIR, filename)
    plt.savefig(save_path)
    print(f"Fig 1 saved to {save_path}")
    plt.close()

def plot_fig3(hetero_data, filename="plot_fig3_segregation_vs_b.png"):
    """Plots Fig 3: Segregation Index."""
    print("Plotting Fig 3: Segregation Index...")
    required_col = 'avg_SegregationIndex' # Make sure this matches reporter output key
    if required_col not in hetero_data.columns:
        print(f"Error plotting Fig 3: '{required_col}' not found in heterogeneous data.")
        print("Did you add the 'SegregationIndex' reporter and rerun simulations?")
        return

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=hetero_data, x='b', y=required_col, marker='o',
                 errorbar=('ci', 95), label='Simulated Segregation')

    # Add random baseline (assuming 50/50 mix initially, p_A=0.5)
    random_baseline = 0.5
    plt.axhline(random_baseline, color='r', linestyle='--', label=f'Random Mixing Baseline ({random_baseline:.2f})')

    plt.title('Fig 3: Spatial Segregation Index vs. Temptation (b)')
    plt.xlabel('Temptation (b)')
    plt.ylabel('Average Segregation Index (S)')
    plt.grid(True)
    # plt.ylim(0, 1.05) # Segregation index is between 0 and 1
    plt.legend(title='Segregation')
    save_path = os.path.join(PLOT_SAVE_DIR, filename)
    plt.savefig(save_path)
    print(f"Fig 3 saved to {save_path}")
    plt.close()

def plot_fig4(baseline_data, hetero_data, filename="plot_fig4_hetero_vs_homo_coop.png"):
    """Plots Fig 4: Heterogeneous vs Homogeneous Cooperation."""
    print("Plotting Fig 4: Heterogeneous vs Homogeneous...")
    if 'avg_CooperationRate' not in baseline_data.columns or 'avg_CooperationRate' not in hetero_data.columns:
         print("Error plotting Fig 4: 'avg_CooperationRate' missing from baseline or hetero data.")
         return
    if 'label' not in baseline_data.columns or 'label' not in hetero_data.columns:
        print("Error plotting Fig 4: 'label' column missing.")
        return

    combined_df = pd.concat([baseline_data, hetero_data], ignore_index=True)

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=combined_df, x='b', y='avg_CooperationRate', hue='label',
                 style='label', markers=True, dashes=True, errorbar=('ci', 95))
    plt.title('Fig 4: Cooperation Rate - Heterogeneous vs. Homogeneous Cultures')
    plt.xlabel('Temptation (b)')
    plt.ylabel('Average Cooperation Rate')
    plt.grid(True)
    plt.ylim(-0.05, 1.05)
    plt.legend(title='Scenario')
    save_path = os.path.join(PLOT_SAVE_DIR, filename)
    plt.savefig(save_path)
    print(f"Fig 4 saved to {save_path}")
    plt.close()

def plot_fig5(hetero_data, filename="plot_fig5_group_coop_vs_b.png"):
    """Plots Fig 5: Group-Specific Cooperation."""
    print("Plotting Fig 5: Group Cooperation...")
    y_var_A = 'avg_CoopRate_A' # Must match reporter key
    y_var_B = 'avg_CoopRate_B' # Must match reporter key
    if y_var_A not in hetero_data.columns or y_var_B not in hetero_data.columns:
        print(f"Error plotting Fig 5: '{y_var_A}' or '{y_var_B}' not found.")
        print("Did you add group cooperation reporters (e.g., CoopRate_A/B) and rerun?")
        return

    plot_df_melted = hetero_data.melt(
        id_vars=['b', 'run_id'], # Keep b and run_id for aggregation
        value_vars=[y_var_A, y_var_B],
        var_name='Group',
        value_name='CooperationRate'
    )
    # Map reporter names to nicer labels
    plot_df_melted['Group'] = plot_df_melted['Group'].map({
        y_var_A: 'Group A (e.g., C < 0.5)',
        y_var_B: 'Group B (e.g., C >= 0.5)'
    })


    plt.figure(figsize=(10, 6))
    sns.lineplot(data=plot_df_melted, x='b', y='CooperationRate', hue='Group',
                 marker='o', errorbar=('ci', 95))
    plt.title('Fig 5: Group-Specific Cooperation Rate vs. Temptation (b)')
    plt.xlabel('Temptation (b)')
    plt.ylabel('Average Group Cooperation Rate')
    plt.grid(True)
    plt.ylim(-0.05, 1.05)
    plt.legend(title='Cultural Group')
    save_path = os.path.join(PLOT_SAVE_DIR, filename)
    plt.savefig(save_path)
    print(f"Fig 5 saved to {save_path}")
    plt.close()

# ==============================================================================
# Snapshot Generation and Plotting
# ==============================================================================
def run_and_save_snapshot(params, b_value, filename_tag):
    """Runs a single simulation and saves the final grid state."""
    snapshot_filename = os.path.join(DATA_SAVE_DIR, f"snapshot_{filename_tag}_b{b_value:.2f}.pkl")

    if os.path.exists(snapshot_filename):
        print(f"Snapshot data already exists: {snapshot_filename}")
        return snapshot_filename

    print(f"Running simulation for snapshot (b={b_value:.2f})...")
    sim_params = {
        'L': PARAMS['L'], 'initial_coop_ratio': PARAMS['initial_coop_ratio'],
        'K': PARAMS['K'], 'steps': PARAMS['steps'],
        'b': b_value,
        'C_dist': params['C_dist'], 'mu': params['mu'], 'sigma': params['sigma'],
        'K_C': params['K_C'], 'p_update_C': params['p_update_C'], 'p_mut': params['p_mut'],
        'seed': int(time.time()) # Use a different seed for snapshot run
    }

    model = CulturalGame(**sim_params)
    model.run_model(params['steps'])

    # Extract final grid state
    grid_state = np.zeros((model.grid.width, model.grid.height, 3)) # x, y, [strategy, C, type]
    agent_types = {} # Store type definition if needed

    # Define Type A/B based on C value (consistent with reporters)
    threshold = 0.5 # Example threshold
    TYPE_A = 1
    TYPE_B = 2

    for agent in model.schedule.agents:
        x, y = agent.pos
        strategy = agent.strategy # 0 or 1
        culture = agent.C         # 0 to 1
        agent_type = TYPE_A if culture < threshold else TYPE_B
        grid_state[x, y, 0] = strategy
        grid_state[x, y, 1] = culture
        grid_state[x, y, 2] = agent_type
        if agent_type not in agent_types:
             agent_types[agent_type] = f"Type {'A' if agent_type == TYPE_A else 'B'} (C {'<' if agent_type == TYPE_A else '>='} {threshold})"


    snapshot_data = {'grid': grid_state, 'params': sim_params, 'types': agent_types}
    with open(snapshot_filename, 'wb') as f:
        pickle.dump(snapshot_data, f)
    print(f"Snapshot data saved to {snapshot_filename}")
    return snapshot_filename


def plot_fig2_snapshot(snapshot_data_file, filename="plot_fig2_snapshot.png"):
    """Plots Fig 2: Spatial Snapshot."""
    print(f"Plotting Fig 2: Snapshot from {snapshot_data_file}...")
    try:
        with open(snapshot_data_file, 'rb') as f:
            snapshot_data = pickle.load(f)
        grid = snapshot_data['grid']
        params = snapshot_data['params']
        # type_map = snapshot_data.get('types', {1:'A', 2:'B'}) # Get type map
        L = params['L']
        b_val = params['b']
    except Exception as e:
        print(f"Error loading snapshot data from {snapshot_data_file}: {e}")
        return

    # Create an image representation
    # Color mapping: Combine strategy and type
    # (A, D): Blue, (A, C): Cyan, (B, D): Red, (B, C): Yellow (Example)
    img = np.zeros((L, L, 3)) # RGB image
    TYPE_A = 1
    TYPE_B = 2
    STRAT_D = 0
    STRAT_C = 1

    color_map = {
        (TYPE_A, STRAT_D): [0, 0, 1], # Blue
        (TYPE_A, STRAT_C): [0, 1, 1], # Cyan
        (TYPE_B, STRAT_D): [1, 0, 0], # Red
        (TYPE_B, STRAT_C): [1, 1, 0], # Yellow
        'Empty':           [0, 0, 0]  # Black for empty cells if any
    }

    for x in range(L):
        for y in range(L):
            strat = int(grid[x, y, 0])
            agent_type = int(grid[x, y, 2])
            key = (agent_type, strat)
            if key in color_map:
                 img[x, y, :] = color_map[key]
            # Handle case where grid might have empty spots or unexpected values if needed


    plt.figure(figsize=(8, 8))
    plt.imshow(img, origin='lower', interpolation='nearest') # origin='lower' puts (0,0) at bottom-left

    # Create custom legend handles
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=color_map[(TYPE_A, STRAT_D)], label='Type A (C<0.5), Defect'),
        Patch(facecolor=color_map[(TYPE_A, STRAT_C)], label='Type A (C<0.5), Cooperate'),
        Patch(facecolor=color_map[(TYPE_B, STRAT_D)], label='Type B (C>=0.5), Defect'),
        Patch(facecolor=color_map[(TYPE_B, STRAT_C)], label='Type B (C>=0.5), Cooperate'),
    ]
    plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.title(f'Fig 2: Spatial Snapshot (b={b_val:.2f})')
    plt.xticks([])
    plt.yticks([])
    save_path = os.path.join(PLOT_SAVE_DIR, filename)
    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make space for legend
    plt.savefig(save_path)
    print(f"Fig 2 saved to {save_path}")
    plt.close()


# ==============================================================================
# Main Execution
# ==============================================================================
if __name__ == "__main__":
    print("Starting Simulation Runs and Plotting...")
    start_time = time.time()

    # --- Run Simulations (or load data) ---
    # IMPORTANT: Ensure reporters are added before running heterogeneous set!
    baseline_results_df = run_simulation_set(generate_baseline_configs, "baseline")
    heterogeneous_results_df = run_simulation_set(generate_heterogeneous_configs, "heterogeneous")

    # --- Generate Plots 1, 3, 4, 5 ---
    if not baseline_results_df.empty:
        plot_fig1(baseline_results_df)
    else:
        print("Skipping Fig 1 due to missing baseline data.")

    if not heterogeneous_results_df.empty:
        plot_fig3(heterogeneous_results_df)
        plot_fig5(heterogeneous_results_df)
    else:
        print("Skipping Figs 3 & 5 due to missing heterogeneous data.")

    if not baseline_results_df.empty and not heterogeneous_results_df.empty:
        plot_fig4(baseline_results_df, heterogeneous_results_df)
    else:
        print("Skipping Fig 4 due to missing baseline or heterogeneous data.")

    # --- Generate Snapshots (Fig 2) ---
    snapshot_files = []
    for b_snap in PARAMS['snapshot_b_values']:
        fname = run_and_save_snapshot(PARAMS['snapshot_hetero_params'], b_snap, "hetero")
        snapshot_files.append(fname)

    # --- Plot Snapshots ---
    for i, snap_file in enumerate(snapshot_files):
         if snap_file:
              # Extract b value from filename or load data to get it
              try:
                  b_val_str = snap_file.split('_b')[-1].replace('.pkl','')
                  b_val = float(b_val_str)
                  plot_fig2_snapshot(snap_file, filename=f"plot_fig2_snapshot_b{b_val:.2f}.png")
              except Exception as e:
                   print(f"Could not generate plot from {snap_file}: {e}")


    end_time = time.time()
    print(f"\nTotal script execution time: {(end_time - start_time):.2f} seconds")
    print("--- Finished ---")
