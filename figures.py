# plot_figures.py
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import numpy as np
import os
import time
from tqdm import tqdm
import pickle
from matplotlib.patches import Patch
from matplotlib import cm # Explicitly import cm

# --- Import Core Simulation Logic ---
RUN_WITH_PARALLEL = False # Default to False, check for parallel module
try:
    # Use the new parallel runner and config structure
    from utils.parallel import batch_run_parallel
    from utils.config import SimConfig
    from core.model import CulturalGame # Needed for snapshot run
    # Import necessary reporters if used directly (usually not needed)
    # Import _get_agent_type from reporters if it's used locally (it is for snapshots)
    from utils.reporters import _get_agent_type

    RUN_WITH_PARALLEL = True
    print("Using parallel.py and SimConfig for simulation runs.")
except ImportError as e:
    print(f"Error importing core modules: {e}")
    print("Ensure model.py, config.py, parallel.py, reporters.py are accessible.")
    # Define a placeholder _get_agent_type if import fails, to allow script to run partially
    def _get_agent_type(agent, threshold=0.5):
         if not hasattr(agent, 'C'): return None
         return 'A' if agent.C < threshold else 'B'
    print("Warning: Using placeholder for _get_agent_type.")
    # exit() # Exit if core functionality is missing for batch runs

# ==============================================================================
# Simulation Parameters (Adapted for SimConfig structure)
# ==============================================================================
PARAMS = {
    # --- Base Parameters (used across different runs unless overridden) ---
    'L': 50,
    'initial_coop_ratio': 0.5,
    'K': 0.1,           # Strategy noise
    'steps': 500,       # Simulation steps per run
    'steady_state_window': 100, # Steps at the end to average over (used by parallel runner)
    'runs_per_config': 5, # Number of independent runs for each parameter combination

    # --- Scan Parameters ---
    'b_values': np.linspace(1.0, 2.5, 16), # Refined range for 'b'

    # --- Baseline Specific (Fixed C) ---
    # Use fewer points for baseline C for clarity in plots, but keep extremes
    'baseline_C_values': {f'C={c:.1f}': c for c in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]},

    # --- Heterogeneous Specific (Evolving C) ---
    'hetero_C_dist': 'bimodal', # Initial distribution
    'hetero_mu': 0.5,          # Initial mix (e.g., 50/50 for bimodal)
    'hetero_sigma': 0.1,       # (Not used for bimodal)
    'hetero_K_C': 0.1,         # Cultural noise
    'hetero_p_update_C': 0.1,  # Cultural update probability
    'hetero_p_mut': 0.001,     # Cultural mutation rate

    # --- Snapshot Specific ---
    'snapshot_b_values': [1.2, 1.6, 1.8, 2.2], # Example b values for snapshots
    # Snapshot uses heterogeneous parameters by default now in run_and_save_snapshot
}

DATA_SAVE_DIR = "simulation_data_physicaA" # Use the Physica A specific dir
PLOT_SAVE_DIR = "plots_physicaA"           # Directory to save generated plots

os.makedirs(DATA_SAVE_DIR, exist_ok=True)
os.makedirs(PLOT_SAVE_DIR, exist_ok=True)

# Use specific data filenames
BASELINE_DATA_FILENAME = os.path.join(DATA_SAVE_DIR, "baseline_scan_data.pkl")
HETERO_DATA_FILENAME = os.path.join(DATA_SAVE_DIR, "hetero_scan_data.pkl")
SNAPSHOT_DATA_DIR = os.path.join(DATA_SAVE_DIR, "snapshots") # Separate dir for snapshot files
os.makedirs(SNAPSHOT_DATA_DIR, exist_ok=True)


# ==============================================================================
# Physica A Style Settings (Apply globally)
# ==============================================================================
mpl.rcParams.update({
    'font.size': 10, 'axes.labelsize': 12, 'axes.titlesize': 14,
    'xtick.labelsize': 10, 'ytick.labelsize': 10, 'legend.fontsize': 9,
    'legend.title_fontsize': 10, 'figure.dpi': 300, 'savefig.dpi': 300,
    'savefig.format': 'png', 'savefig.bbox': 'tight',
    'lines.linewidth': 1.5, 'lines.markersize': 4, # Base marker size
    'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
    'mathtext.fontset': 'cm', 'axes.grid': False,
})
# Define palette here but apply in plot functions to avoid affecting snapshots
N_BASELINE_TYPES = len(PARAMS['baseline_C_values'])
COLOR_PALETTE_BASELINE = sns.color_palette("viridis", n_colors=N_BASELINE_TYPES)
COLOR_HETERO = sns.color_palette("rocket", 1)[0] # A distinct color for hetero


# ==============================================================================
# Simulation Running Functions (Using parallel.py)
# ==============================================================================

def generate_baseline_configs():
    """Generates SimConfig objects for the baseline (fixed C) scan."""
    configs = []
    base_params = {
        'L': PARAMS['L'], 'initial_coop_ratio': PARAMS['initial_coop_ratio'],
        'K': PARAMS['K'], 'steps': PARAMS['steps'],
        'C_dist': 'fixed', 'sigma': 0, # Fixed C specific
        'K_C': 0, 'p_update_C': 0, 'p_mut': 0, # No evolution
    }
    # Sort C values for consistent legend order
    sorted_c_items = sorted(PARAMS['baseline_C_values'].items(), key=lambda item: item[1])

    total_configs_generated = 0 # Counter for unique seeds
    for label, c_val in sorted_c_items:
        for b in PARAMS['b_values']:
            param_set_id_base = f"Baseline_C{c_val:.1f}_b{b:.2f}"
            for i in range(PARAMS['runs_per_config']):
                run_params = base_params.copy()
                run_params['b'] = b
                run_params['mu'] = c_val # 'mu' holds the fixed C value here
                run_params['param_set_id'] = f"{param_set_id_base}_run{i}" # More unique ID
                run_params['run_id'] = i
                run_params['seed'] = total_configs_generated # Assign unique seed
                run_params['label'] = label # Store the descriptive label (e.g., 'C=0.1')

                try:
                    configs.append(SimConfig(**run_params))
                    total_configs_generated += 1 # Increment for next unique seed
                except TypeError as e:
                    print(f"Error creating SimConfig (Baseline). Params: {run_params}\nError: {e}")

    print(f"Generated {len(configs)} SimConfig objects for baseline scan.")
    return configs


def generate_heterogeneous_configs():
    """Generates SimConfig objects for the heterogeneous (evolving C) scan."""
    configs = []
    base_params = {
        'L': PARAMS['L'], 'initial_coop_ratio': PARAMS['initial_coop_ratio'],
        'K': PARAMS['K'], 'steps': PARAMS['steps'],
        'C_dist': PARAMS['hetero_C_dist'], 'mu': PARAMS['hetero_mu'], 'sigma': PARAMS['hetero_sigma'],
        'K_C': PARAMS['hetero_K_C'], 'p_update_C': PARAMS['hetero_p_update_C'], 'p_mut': PARAMS['hetero_p_mut'],
        'label': 'Heterogeneous' # Consistent label for all these runs
    }

    total_configs_generated = 0 # Counter for unique seeds
    seed_offset = 50000 # Ensure seeds don't overlap with baseline

    for b in PARAMS['b_values']:
        param_set_id_base = f"Hetero_b{b:.2f}"
        for i in range(PARAMS['runs_per_config']):
            run_params = base_params.copy()
            run_params['b'] = b
            run_params['param_set_id'] = f"{param_set_id_base}_run{i}"
            run_params['run_id'] = i
            run_params['seed'] = seed_offset + total_configs_generated # Unique seed
            try:
                configs.append(SimConfig(**run_params))
                total_configs_generated += 1
            except TypeError as e:
                print(f"Error creating SimConfig (Hetero). Params: {run_params}\nError: {e}")

    print(f"Generated {len(configs)} SimConfig objects for heterogeneous scan.")
    return configs


def run_simulation_batch(config_generator, data_filename, force_rerun=False):
    """
    Runs a batch of simulations using parallel.py or loads existing data.
    Returns the raw DataFrame containing results from all runs.
    """
    if not force_rerun and os.path.exists(data_filename):
        print(f"Loading existing data from {data_filename}...")
        try:
            with open(data_filename, 'rb') as f:
                results_df = pickle.load(f)
            print(f"Loaded {len(results_df)} results.")
            # Basic validation
            if not isinstance(results_df, pd.DataFrame):
                 print("Error: Loaded data is not a Pandas DataFrame. Rerunning...")
                 force_rerun = True
            elif results_df.empty:
                 print("Warning: Loaded DataFrame is empty. Rerunning...")
                 force_rerun = True
            else:
                 # Check for essential columns maybe?
                 if 'run_id' not in results_df.columns or 'label' not in results_df.columns:
                      print("Warning: 'run_id' or 'label' missing from loaded data. Consider rerunning.")
                 return results_df # Return loaded data
        except Exception as e:
            print(f"Error loading data from {data_filename}: {e}. Rerunning...")
            force_rerun = True # Force rerun if loading fails

    # --- Proceed with running simulations ---
    print(f"Running simulations for {data_filename}...")
    sim_configs = config_generator()
    if not sim_configs:
         print("Warning: No configurations generated.")
         return pd.DataFrame()

    # Ensure parallel execution is available
    if not RUN_WITH_PARALLEL:
         print("Error: Parallel execution module not loaded. Cannot run batch simulations.")
         return pd.DataFrame() # Or raise an exception

    results_df = batch_run_parallel(
        sim_configs,
        steady_state_window=PARAMS['steady_state_window']
        # num_workers defaults in batch_run_parallel
    )

    # Save the results
    if not results_df.empty:
        try:
            with open(data_filename, 'wb') as f:
                pickle.dump(results_df, f)
            print(f"Simulation data saved to {data_filename}")
        except Exception as e:
            print(f"Error saving data to {data_filename}: {e}")
    else:
         print(f"Warning: No results generated for {data_filename}.")

    return results_df


# ==============================================================================
# Plotting Functions (Adapted from user's file, using new data structure)
# ==============================================================================

def save_plot(fig, base_filename, plot_dir=PLOT_SAVE_DIR):
    """Helper function to save plot in multiple formats."""
    png_path = os.path.join(plot_dir, f"{base_filename}.png")
    pdf_path = os.path.join(plot_dir, f"{base_filename}.pdf") # Save PDF for quality
    try:
        fig.savefig(png_path, dpi=mpl.rcParams['savefig.dpi'], bbox_inches='tight')
        fig.savefig(pdf_path, bbox_inches='tight')
        # print(f"Plot saved to {png_path} and {pdf_path}") # Reduce console noise
    except Exception as e:
        print(f"Error saving plot {base_filename}: {e}")

def plot_fig1(baseline_data, filename_base="fig1_baseline_coop_vs_b"):
    """Plots Baseline Homogeneous Cooperation vs. Temptation (b)."""
    print("Plotting Fig 1: Baseline Cooperation...")
    # Input `baseline_data` is the raw DF with multiple runs per config
    if baseline_data.empty:
        print("Error plotting Fig 1: Baseline data is empty.")
        return
    if 'avg_CooperationRate' not in baseline_data.columns:
         print("Error plotting Fig 1: 'avg_CooperationRate' column not found.")
         return
    if 'label' not in baseline_data.columns:
         print("Error plotting Fig 1: 'label' column missing.")
         return

    fig, ax = plt.subplots(figsize=(6, 4))
    # Use seaborn to plot, it will aggregate over runs for CI
    # Ensure 'label' column has sorted unique values if needed before plotting
    hue_order = sorted(baseline_data['label'].unique(), key=lambda x: float(x.split('=')[1]))
    sns.lineplot(data=baseline_data, x='b', y='avg_CooperationRate', hue='label',
                 hue_order=hue_order, # Ensure legend order matches C value
                 palette=COLOR_PALETTE_BASELINE, # Use specific palette
                 marker='o',
                 markersize=mpl.rcParams['lines.markersize'],
                 linewidth=mpl.rcParams['lines.linewidth'],
                 errorbar=('ci', 95), # Calculate 95% CI across runs
                 err_style="bars",
                 err_kws={'capsize': 3},
                 ax=ax)

    ax.set_title('Baseline Cooperation Rate (Fixed C)')
    ax.set_xlabel('Temptation ($b$)')
    ax.set_ylabel('Average Cooperation Rate $\\langle f_C \\rangle$')
    ax.set_ylim(-0.05, 1.05)
    ax.legend(title='Culture Type (Fixed C)', bbox_to_anchor=(1.02, 1), loc='upper left')
    ax.grid(False)

    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout for legend
    save_plot(fig, filename_base)
    plt.close(fig)

def plot_fig3(hetero_data, filename_base="fig3_segregation_vs_b"):
    """Plots Segregation Index vs. Temptation (b) for Heterogeneous case."""
    print("Plotting Fig 3: Segregation Index...")
    # Input `hetero_data` is the raw DF with multiple runs per config
    if hetero_data.empty:
        print("Error plotting Fig 3: Heterogeneous data is empty.")
        return
    required_col = 'avg_SegregationIndex'
    if required_col not in hetero_data.columns:
        print(f"Error plotting Fig 3: '{required_col}' column not found.")
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.lineplot(data=hetero_data, x='b', y=required_col, marker='s',
                 color=COLOR_HETERO, # Use specific color for hetero
                 markersize=mpl.rcParams['lines.markersize'],
                 linewidth=mpl.rcParams['lines.linewidth'],
                 errorbar=('ci', 95),
                 err_style="bars",
                 err_kws={'capsize': 3},
                 label='Simulated (Heterogeneous)', # Label for the line
                 ax=ax)

    # Add random baseline (adjust if needed based on mu)
    # For two types with equal initial probability (mu=0.5), expected neighbors of same type is 0.5
    # Segregation Index = Avg(fraction_same) -> baseline is 0.5
    random_baseline = 0.5 # For mu=0.5 initial condition
    # More general: p_same = mu^2 + (1-mu)^2. Here mu=0.5 => 0.25+0.25=0.5
    ax.axhline(random_baseline, color='grey', linestyle='--', linewidth=1.2,
               label=f'Random Mixing Baseline ({random_baseline:.2f})')

    ax.set_title('Spatial Segregation (Heterogeneous Culture)')
    ax.set_xlabel('Temptation ($b$)')
    ax.set_ylabel('Average Segregation Index $\\langle S \\rangle$')
    ax.set_ylim(bottom=-0.05) # Allow slightly below 0 for error bars
    ax.legend(title='Segregation Measure')
    ax.grid(False)

    plt.tight_layout()
    save_plot(fig, filename_base)
    plt.close(fig)

def plot_fig4(baseline_data, hetero_data, filename_base="fig4_hetero_vs_homo_coop"):
    """Plots Comparison: Heterogeneous vs Selected Homogeneous Cooperation."""
    print("Plotting Fig 4: Heterogeneous vs Homogeneous Comparison...")
    if baseline_data.empty or hetero_data.empty:
        print("Error plotting Fig 4: Baseline or Heterogeneous data is empty.")
        return
    if 'avg_CooperationRate' not in baseline_data.columns or 'avg_CooperationRate' not in hetero_data.columns:
         print("Error plotting Fig 4: 'avg_CooperationRate' column missing.")
         return
    if 'label' not in baseline_data.columns or 'label' not in hetero_data.columns:
         print("Error plotting Fig 4: 'label' column missing.")
         return

    # --- Data Preparation ---
    # Select specific baseline C values to compare against
    # Choose representative values, e.g., extremes and middle
    baseline_labels_to_plot = ['C=0.0', 'C=0.4', 'C=1.0'] # Adjust as needed from PARAMS['baseline_C_values']
    baseline_subset = baseline_data[baseline_data['label'].isin(baseline_labels_to_plot)].copy()

    hetero_data_copy = hetero_data.copy()
    hetero_label = 'Heterogeneous' # Get label from data
    combined_df = pd.concat([baseline_subset, hetero_data_copy], ignore_index=True)

    # Define markers and dash styles dynamically
    unique_labels = sorted(combined_df['label'].unique(), key=lambda x: (x != hetero_label, float(x.split('=')[1]) if '=' in x else -1))
    n_unique = len(unique_labels)
    palette = sns.color_palette("viridis", n_colors=n_unique-1) + [COLOR_HETERO] # Assign hetero color last
    markers = {label: m for label, m in zip(unique_labels, ['o', '^', 'D', 'v', 's'][:n_unique])}
    dash_styles = {label: (4, 1.5) if label == hetero_label else "" for label in unique_labels}


    fig, ax = plt.subplots(figsize=(6, 4))
    sns.lineplot(data=combined_df,
                 x='b', y='avg_CooperationRate',
                 hue='label', hue_order=unique_labels, palette=palette,
                 style='label', style_order=unique_labels, markers=markers, dashes=dash_styles,
                 markersize=mpl.rcParams['lines.markersize'] + 1,
                 linewidth=mpl.rcParams['lines.linewidth'],
                 errorbar=('ci', 95),
                 err_style="bars", err_kws={'capsize': 3},
                 ax=ax)

    ax.set_title('Cooperation Rate Comparison')
    ax.set_xlabel('Temptation ($b$)')
    ax.set_ylabel('Average Cooperation Rate $\\langle f_C \\rangle$')
    ax.set_ylim(-0.05, 1.05)
    ax.legend(title='Scenario')
    ax.grid(False)

    plt.tight_layout()
    save_plot(fig, filename_base)
    plt.close(fig)


def plot_fig5(hetero_data, filename_base="fig5_group_coop_vs_b"):
    """Plots Group-Specific Cooperation Rates vs. Temptation (b) (Heterogeneous)."""
    print("Plotting Fig 5: Group Cooperation (Heterogeneous)...")
    if hetero_data.empty:
        print("Error plotting Fig 5: Heterogeneous data is empty.")
        return
    # Columns generated by reporters: avg_CoopRate_A, avg_CoopRate_B
    y_var_A = 'avg_CoopRate_A'
    y_var_B = 'avg_CoopRate_B'
    if y_var_A not in hetero_data.columns or y_var_B not in hetero_data.columns:
        print(f"Error plotting Fig 5: '{y_var_A}' or '{y_var_B}' not found.")
        return

    # Melt the DataFrame for easier plotting with seaborn hue
    plot_df_melted = hetero_data.melt(
        id_vars=['b', 'run_id'], # Keep b and run_id for CI calculation
        value_vars=[y_var_A, y_var_B],
        var_name='Group Reporter', # Temporary column name
        value_name='CooperationRate'
    )
    # Map reporter names to nicer labels
    plot_df_melted['Group Type'] = plot_df_melted['Group Reporter'].map({
        y_var_A: 'Group A ($C < 0.5$)', # Assuming 0.5 threshold in reporter
        y_var_B: 'Group B ($C \\geq 0.5$)'
    })

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.lineplot(data=plot_df_melted, x='b', y='CooperationRate', hue='Group Type',
                 palette=sns.color_palette("coolwarm_r", 2), # Use a diverging palette (e.g., blue/red)
                 marker='o',
                 markersize=mpl.rcParams['lines.markersize'],
                 linewidth=mpl.rcParams['lines.linewidth'],
                 errorbar=('ci', 95), # Uses underlying run_id variation
                 err_style="bars", err_kws={'capsize': 3},
                 ax=ax)

    ax.set_title('Group-Specific Cooperation (Heterogeneous)')
    ax.set_xlabel('Temptation ($b$)')
    ax.set_ylabel('Average Group Cooperation Rate')
    ax.set_ylim(-0.05, 1.05)
    ax.legend(title='Cultural Group')
    ax.grid(False)

    plt.tight_layout()
    save_plot(fig, filename_base)
    plt.close(fig)

# ==============================================================================
# Snapshot Generation and Plotting (Adapted Physica A Style)
# ==============================================================================
def run_and_save_snapshot(b_value, filename_tag="hetero"):
    """Runs a single simulation for snapshot and saves final grid state."""
    snapshot_filename = os.path.join(SNAPSHOT_DATA_DIR, f"snapshot_{filename_tag}_b{b_value:.2f}.pkl")

    # Overwrite snapshot if force_rerun is True (add flag if needed)
    # if FORCE_RERUN_SNAPSHOTS and os.path.exists(snapshot_filename): os.remove(...)
    if os.path.exists(snapshot_filename):
        print(f"Skipping run, snapshot exists: {snapshot_filename}")
        return snapshot_filename

    print(f"Running simulation for snapshot (b={b_value:.2f})...")

    # Build parameters dictionary for this specific run (using Hetero params)
    # Need to ensure all parameters required by CulturalGame.__init__ are present
    model_params = {
        'L': PARAMS['L'], 'initial_coop_ratio': PARAMS['initial_coop_ratio'],
        'K': PARAMS['K'], 'b': b_value,
        'C_dist': PARAMS['hetero_C_dist'], 'mu': PARAMS['hetero_mu'], 'sigma': PARAMS['hetero_sigma'],
        'K_C': PARAMS['hetero_K_C'], 'p_update_C': PARAMS['hetero_p_update_C'], 'p_mut': PARAMS['hetero_p_mut'],
        'seed': int(time.time() * 1000 + b_value * 100) % (2**32 - 1) # Unique-ish seed
    }
    steps_to_run = PARAMS['steps']

    try:
        # Run directly without parallel overhead for single snapshot
        model = CulturalGame(**model_params)
        # model.run_model(steps_to_run) # run_model might have tqdm, run manually
        for _ in range(steps_to_run):
            model.step()

    except Exception as e:
        print(f"Error running snapshot simulation (b={b_value}): {e}")
        import traceback
        traceback.print_exc() # Print full traceback for debugging
        return None

    # Extract Grid State (using _get_agent_type helper)
    grid_state = np.zeros((model.grid.width, model.grid.height, 3)) # x, y, [strategy, C, type_id]
    agent_types_map = {} # Map type_id to description
    threshold = 0.5 # Threshold used in reporters, keep consistent
    TYPE_A_ID = 1
    TYPE_B_ID = 2

    for agent in model.schedule.agents:
        x, y = agent.pos
        if x is None or y is None: continue
        strat = agent.strategy # 0 or 1
        cult = agent.C         # 0 to 1
        agent_type = _get_agent_type(agent, threshold) # Returns 'A' or 'B' or None
        if agent_type is None:
            agent_type_id = 0 # Use 0 for undefined type
        else:
            agent_type_id = TYPE_A_ID if agent_type == 'A' else TYPE_B_ID

        grid_state[x, y, 0] = strat
        grid_state[x, y, 1] = cult
        grid_state[x, y, 2] = agent_type_id

        if agent_type_id not in agent_types_map and agent_type_id != 0:
             desc = f"Group {agent_type} (C {'<' if agent_type == 'A' else '>='} {threshold})"
             agent_types_map[agent_type_id] = desc


    snapshot_params_saved = model_params.copy()
    snapshot_params_saved['steps_run'] = steps_to_run

    snapshot_data = {'grid': grid_state, 'params': snapshot_params_saved, 'types': agent_types_map}
    try:
        with open(snapshot_filename, 'wb') as f:
            pickle.dump(snapshot_data, f)
        print(f"Snapshot data saved to {snapshot_filename}")
        return snapshot_filename
    except Exception as e:
        print(f"Error saving snapshot data to {snapshot_filename}: {e}")
        return None


def plot_fig2_snapshot(snapshot_data_file, filename_base_prefix="fig2_snapshot"):
    """Plots Spatial Snapshot with Physica A style."""
    if not snapshot_data_file or not os.path.exists(snapshot_data_file):
         print(f"Snapshot file not found or invalid: {snapshot_data_file}")
         return

    print(f"Plotting Snapshot from {os.path.basename(snapshot_data_file)}...")
    try:
        with open(snapshot_data_file, 'rb') as f:
            snapshot_data = pickle.load(f)
        grid = snapshot_data['grid']
        params = snapshot_data['params']
        type_desc = snapshot_data['types'] # {1: 'Group A (C<0.5)', 2: 'Group B (C>=0.5)'}
        L = params['L']
        b_val = params['b']
    except Exception as e:
        print(f"Error loading snapshot data from {snapshot_data_file}: {e}")
        return

    img = np.zeros((L, L, 3)) # RGB image
    TYPE_A_ID = 1
    TYPE_B_ID = 2
    STRAT_D = 0 # Defector
    STRAT_C = 1 # Cooperator

    # Define colors (adjust for clarity/preference)
    color_map = {
        (TYPE_A_ID, STRAT_D): np.array([0.0, 0.0, 0.6]), # Dark Blue (A, Defect)
        (TYPE_A_ID, STRAT_C): np.array([0.6, 0.8, 1.0]), # Light Blue (A, Coop)
        (TYPE_B_ID, STRAT_D): np.array([0.6, 0.0, 0.0]), # Dark Red (B, Defect)
        (TYPE_B_ID, STRAT_C): np.array([1.0, 0.6, 0.6]), # Pink/Light Red (B, Coop)
        'default': np.array([0.5, 0.5, 0.5]) # Grey for unknown type_id=0
    }
    # Create labels dynamically from loaded type descriptions
    legend_labels = {}
    if TYPE_A_ID in type_desc:
        legend_labels[(TYPE_A_ID, STRAT_D)] = f"{type_desc[TYPE_A_ID]} / Defect"
        legend_labels[(TYPE_A_ID, STRAT_C)] = f"{type_desc[TYPE_A_ID]} / Cooperate"
    if TYPE_B_ID in type_desc:
        legend_labels[(TYPE_B_ID, STRAT_D)] = f"{type_desc[TYPE_B_ID]} / Defect"
        legend_labels[(TYPE_B_ID, STRAT_C)] = f"{type_desc[TYPE_B_ID]} / Cooperate"


    for x in range(L):
        for y in range(L):
            strat = int(grid[x, y, 0])
            agent_type_id = int(grid[x, y, 2])
            key = (agent_type_id, strat)
            # imshow expects (height, width) = (y, x)
            img[y, x, :] = color_map.get(key, color_map['default'])

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(img, origin='lower', interpolation='nearest')

    # Create custom legend
    valid_legend_keys = sorted([k for k in color_map if k != 'default' and k in legend_labels])
    legend_elements = [Patch(facecolor=color_map[key], edgecolor='k', linewidth=0.5, label=legend_labels[key])
                       for key in valid_legend_keys]

    # Only add legend if there are elements to show
    if legend_elements:
         ax.legend(handles=legend_elements, bbox_to_anchor=(1.02, 1), loc='upper left', title="Agent State", fontsize='small')
         right_margin = 0.75 # Leave space for legend
    else:
         right_margin = 0.95 # No legend, use more space

    ax.set_title(f'Spatial Snapshot ($L={L}, b={b_val:.2f}$)')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)

    plt.tight_layout(rect=[0, 0, right_margin, 1]) # Adjust right margin
    filename = f"{filename_base_prefix}_L{L}_b{b_val:.2f}"
    save_plot(fig, filename)
    plt.close(fig)


# ==============================================================================
# Main Execution Workflow
# ==============================================================================
if __name__ == "__main__":
    print("--- Starting Physica A Analysis Workflow ---")
    overall_start_time = time.time()

    # --- Control Flags ---
    FORCE_RERUN_BASELINE = False     # Rerun baseline simulations?
    FORCE_RERUN_HETERO = False       # Rerun heterogeneous simulations?
    FORCE_RERUN_SNAPSHOTS = False    # Rerun simulations for snapshots?

    # --- 1. Run/Load Baseline Scan ---
    print("\n--- Processing Baseline Data ---")
    baseline_raw_data = run_simulation_batch(generate_baseline_configs, BASELINE_DATA_FILENAME, force_rerun=FORCE_RERUN_BASELINE)
    # Sort data for consistent legend order in plots
    if not baseline_raw_data.empty and 'label' in baseline_raw_data.columns:
         try:
             # Extract float value from label for sorting
             baseline_raw_data['c_value_sort'] = baseline_raw_data['label'].str.extract(r'C=([\d.]+)').astype(float)
             baseline_raw_data = baseline_raw_data.sort_values('c_value_sort').drop(columns=['c_value_sort'])
         except Exception as e:
              print(f"Warning: Could not sort baseline data by label C value: {e}")


    # --- 2. Run/Load Heterogeneous Scan ---
    print("\n--- Processing Heterogeneous Data ---")
    hetero_raw_data = run_simulation_batch(generate_heterogeneous_configs, HETERO_DATA_FILENAME, force_rerun=FORCE_RERUN_HETERO)


    # --- 3. Generate/Check Snapshot Data ---
    snapshot_files = []
    print("\n--- Generating/Checking Snapshots ---")
    for b_snap in PARAMS['snapshot_b_values']:
         if FORCE_RERUN_SNAPSHOTS:
              snap_filename_temp = os.path.join(SNAPSHOT_DATA_DIR, f"snapshot_hetero_b{b_snap:.2f}.pkl")
              if os.path.exists(snap_filename_temp):
                   try:
                        os.remove(snap_filename_temp)
                        print(f"Removed existing snapshot: {snap_filename_temp}")
                   except OSError as e:
                        print(f"Error removing existing snapshot {snap_filename_temp}: {e}")

         # Snapshots use heterogeneous parameters by default in the function
         fname = run_and_save_snapshot(b_snap, "hetero") # Pass b value, tag is fixed
         if fname:
             snapshot_files.append(fname)

    # --- 4. Generate Plots ---
    print("\n--- Generating Plots ---")

    # --- Fig 1: Baseline Cooperation ---
    if not baseline_raw_data.empty:
        plot_fig1(baseline_raw_data)
    else:
        print("Skipping Fig 1 (Baseline Cooperation) due to missing/empty data.")

    # --- Fig 3: Heterogeneous Segregation ---
    if not hetero_raw_data.empty:
        plot_fig3(hetero_raw_data)
    else:
        print("Skipping Fig 3 (Heterogeneous Segregation) due to missing/empty data.")

    # --- Fig 4: Comparison Plot ---
    if not baseline_raw_data.empty and not hetero_raw_data.empty:
        plot_fig4(baseline_raw_data, hetero_raw_data)
    else:
        print("Skipping Fig 4 (Comparison) due to missing/empty baseline or heterogeneous data.")

    # --- Fig 5: Heterogeneous Group Cooperation ---
    if not hetero_raw_data.empty:
        plot_fig5(hetero_raw_data)
    else:
        print("Skipping Fig 5 (Group Cooperation) due to missing/empty data.")

    # --- Fig 2: Snapshots ---
    print("\n--- Plotting Snapshots ---")
    if not snapshot_files:
        print("No snapshot files found or generated to plot.")
    else:
        plot_count = 0
        for snap_file in snapshot_files:
             plot_fig2_snapshot(snap_file) # Uses default filename prefix "fig2_snapshot"
             plot_count += 1
        print(f"Plotted {plot_count} snapshots.")


    overall_end_time = time.time()
    print(f"\n--- Workflow finished in {(overall_end_time - overall_start_time):.2f} seconds ---")
    print(f"Data saved in: '{DATA_SAVE_DIR}'")
    print(f"Plots saved in: '{PLOT_SAVE_DIR}'")
