import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl # Import main matplotlib library
import seaborn as sns
import numpy as np
import os
import time
from tqdm import tqdm # For progress bars
import pickle # For saving/loading data
from matplotlib.patches import Patch # For custom legends

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
# Simulation Parameters (Keep as is)
# ==============================================================================
PARAMS = {
    'L': 50,
    'initial_coop_ratio': 0.5,
    'K': 0.1,           # Strategy noise
    'steps': 500,       # Simulation steps per run
    'steady_state_window': 100, # Steps at the end to average over
    'runs_per_b': 5,    # Number of independent runs for each 'b' value to average
    'b_values': np.linspace(1.0, 6, 6), # Range of temptation 'b'

    # --- Baseline Specific ---
    'baseline_C_values': {f'C={c:.1f}': c for c in np.arange(0, 1.1, 0.1)},
    # --- Heterogeneous Specific ---
    'hetero_C_dist': 'bimodal', # Initial distribution for Figs 3, 4, 5
    'hetero_mu': 0.5,         # Initial mix (e.g., 50/50 for bimodal)
    'hetero_sigma': 0.1,      # (Not used for bimodal, relevant if 'normal')
    'hetero_K_C': 0.1,        # Cultural noise
    'hetero_p_update_C': 0.1, # Cultural update probability
    'hetero_p_mut': 0.001,    # Cultural mutation rate

    # --- Snapshot Specific ---
    'snapshot_b_values': [1.2, 1.6, 2.0, 2.4], # Example b values for snapshots
    'snapshot_hetero_params': { # Use hetero params for snapshot runs
         'C_dist': 'bimodal', 'mu': 0.5, 'sigma': 0.1,
         'K_C': 0.1, 'p_update_C': 0.1, 'p_mut': 0.001
    }
}

DATA_SAVE_DIR = "simulation_data" # Directory to save intermediate data
PLOT_SAVE_DIR = "plots_physicaA"           # Directory to save generated plots

os.makedirs(DATA_SAVE_DIR, exist_ok=True)
os.makedirs(PLOT_SAVE_DIR, exist_ok=True)

# ==============================================================================
# Physica A Style Settings (Apply globally)
# ==============================================================================
# You might need to install fonts like Times New Roman if not available
# Or choose a common sans-serif like 'Arial' or 'DejaVu Sans'
# plt.style.use('seaborn-v0_8-paper') # Start with a base style suitable for papers


mpl.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'legend.title_fontsize': 11,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.format': 'png',
    'savefig.bbox': 'tight',
    'lines.linewidth': 1.5,
    'lines.markersize': 5,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
    'mathtext.fontset': 'cm',
    # --- 修改这里 ---
    'axes.grid': False,             # 关闭网格
    # 'grid.linestyle': ':',       # 这两行现在无效，可以注释掉或删除
    # 'grid.alpha': 0.7,           # 这两行现在无效，可以注释掉或删除
    # --- 结束修改 ---
})

# Consistent color palette (example: seaborn's colorblind)
# You can choose others like 'viridis', 'magma', 'tab10', etc.
COLOR_PALETTE = sns.color_palette("colorblind")
sns.set_palette(COLOR_PALETTE)

# ==============================================================================
# Simulation Running Functions (Keep as is)
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
        window_df = pd.DataFrame() # Handle empty dataframe case

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

        # Handle case where no results were generated
        if not all_results:
            print(f"Warning: No results generated for {filename_tag}. Creating empty DataFrame.")
            all_results_df = pd.DataFrame()
        else:
             all_results_df = pd.DataFrame(all_results)

        # Only save if DataFrame is not empty
        if not all_results_df.empty:
            with open(data_filename, 'wb') as f:
                pickle.dump(all_results_df, f)
            print(f"Simulation data saved to {data_filename}")
        else:
             print(f"Skipping save for empty results: {filename_tag}")

    return all_results_df

# --- Configuration Generators (Keep as is) ---
def generate_baseline_configs():
    configs = []
    # Sort C values for consistent legend order
    sorted_c_items = sorted(PARAMS['baseline_C_values'].items(), key=lambda item: item[1])
    # for label, c_val in PARAMS['baseline_C_values'].items():
    for label, c_val in sorted_c_items:
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
            'label': 'Heterogeneous Mix' # Consistent label
        }
        configs.append(params)
    return configs

# ==============================================================================
# Plotting Functions (Physica A Style)
# ==============================================================================

def save_plot(fig, base_filename, plot_dir=PLOT_SAVE_DIR):
    """Helper function to save plot in multiple formats."""
    png_path = os.path.join(plot_dir, f"{base_filename}.png")
    pdf_path = os.path.join(plot_dir, f"{base_filename}.pdf")
    # Save PNG (high-res)
    fig.savefig(png_path, dpi=mpl.rcParams['savefig.dpi'], bbox_inches='tight')
    # Save PDF (vector format, often preferred by journals)
    fig.savefig(pdf_path, bbox_inches='tight')
    print(f"Plot saved to {png_path} and {pdf_path}")

def plot_fig1(baseline_data, filename_base="fig1_baseline_coop_vs_b"):
    """Plots Baseline Homogeneous Cooperation vs. Temptation (b)."""
    print("Plotting Fig 1: Baseline Cooperation...")
    if 'avg_CooperationRate' not in baseline_data.columns:
         print("Error plotting Fig 1: 'avg_CooperationRate' not found.")
         return
    if 'label' not in baseline_data.columns:
         print("Error plotting Fig 1: 'label' column missing.")
         return

    fig, ax = plt.subplots(figsize=(6, 4)) # Adjust figsize as needed
    sns.lineplot(data=baseline_data, x='b', y='avg_CooperationRate', hue='label',
                 marker='o', # Use markers
                 markersize=mpl.rcParams['lines.markersize'],
                 linewidth=mpl.rcParams['lines.linewidth'],
                 errorbar=('ci', 95),
                 err_style="bars", # Use bars for error bars
                 err_kws={'capsize': 3}, # Add caps to error bars
                 ax=ax) # Plot on the created axes

    ax.set_title('Baseline Cooperation Rate') # Concise title
    ax.set_xlabel('Temptation ($b$)') # Use math formatting for b
    ax.set_ylabel('Average Cooperation Rate $\\langle f_C \\rangle$') # Example with math notation
#    ax.grid(False, linestyle=mpl.rcParams['grid.linestyle'], alpha=mpl.rcParams['grid.alpha'])
    ax.set_ylim(-0.05, 1.05)
    ax.legend(title='Culture Type (C)', bbox_to_anchor=(1.02, 1), loc='upper left') # Move legend outside slightly

    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout slightly if legend is outside
    save_plot(fig, filename_base)
    plt.close(fig) # Close the figure to free memory

def plot_fig3(hetero_data, filename_base="fig3_segregation_vs_b"):
    """Plots Segregation Index vs. Temptation (b)."""
    print("Plotting Fig 3: Segregation Index...")
    required_col = 'avg_SegregationIndex'
    if required_col not in hetero_data.columns:
        print(f"Error plotting Fig 3: '{required_col}' not found.")
        print("Ensure 'SegregationIndex' reporter was run.")
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.lineplot(data=hetero_data, x='b', y=required_col, marker='s', # Use square markers
                 markersize=mpl.rcParams['lines.markersize'],
                 linewidth=mpl.rcParams['lines.linewidth'],
                 errorbar=('ci', 95),
                 err_style="bars",
                 err_kws={'capsize': 3},
                 label='Simulated Segregation', # Add label for line
                 ax=ax)

    # Add random baseline
    random_baseline = 0.5 # Adjust if initial mix is different
    ax.axhline(random_baseline, color='red', linestyle='--', linewidth=1.2,
               label=f'Random Mixing Baseline ({random_baseline:.1f})')

    ax.set_title('Spatial Segregation Index')
    ax.set_xlabel('Temptation ($b$)')
    ax.set_ylabel('Average Segregation Index $\\langle S \\rangle$')
#    ax.grid(True, linestyle=mpl.rcParams['grid.linestyle'], alpha=mpl.rcParams['grid.alpha'])
    # ax.set_ylim(bottom=0) # Ensure y-axis starts at 0 or slightly below
    ax.legend(title='Segregation Measure')

    plt.tight_layout()
    save_plot(fig, filename_base)
    plt.close(fig)

def plot_fig4(baseline_data, hetero_data, filename_base="fig4_hetero_vs_homo_coop"):
    """Plots Comparison: Heterogeneous vs Homogeneous Cooperation."""
    print("Plotting Fig 4: Heterogeneous vs Homogeneous...")
    # ... (error checking code remains the same) ...

    # --- Data Preparation ---
    baseline_labels_to_plot = ['C=0.1', 'C=0.5', 'C=0.9']
    baseline_subset = baseline_data[baseline_data['label'].isin(baseline_labels_to_plot)].copy()

    hetero_data_copy = hetero_data.copy()
    hetero_label = 'Heterogeneous'
    hetero_data_copy['label'] = hetero_label
    combined_df = pd.concat([baseline_subset, hetero_data_copy], ignore_index=True)

    # --- Define markers and dash styles ---
    markers = {'C=0.1': 'o', 'C=0.5': '^', 'C=0.9': 'D', hetero_label: 's'}

    # Define dash styles: Use "" (empty string) for solid lines instead of False
    # Keys MUST match the unique values in the 'label' column
    dash_styles = {'C=0.1': "", 'C=0.5': "", 'C=0.9': "", hetero_label: (4, 1.5)} # Use "" for solid

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.lineplot(data=combined_df,
                 x='b',
                 y='avg_CooperationRate',
                 hue='label',       # Color based on label
                 style='label',     # Vary line style AND marker based on label
                 markers=markers,   # Map labels to specific markers
                 dashes=dash_styles,# Map labels to specific dash patterns ("" or tuple)
                 markersize=mpl.rcParams['lines.markersize'] + 1,
                 linewidth=mpl.rcParams['lines.linewidth'],
                 errorbar=('ci', 95),
                 err_style="bars",
                 err_kws={'capsize': 3},
                 ax=ax)

    ax.set_title('Cooperation Rate Comparison')
    ax.set_xlabel('Temptation ($b$)')
    ax.set_ylabel('Average Cooperation Rate $\\langle f_C \\rangle$')
#    ax.grid(True, linestyle=mpl.rcParams['grid.linestyle'], alpha=mpl.rcParams['grid.alpha'])
    ax.set_ylim(-0.05, 1.05)
    ax.legend(title='Scenario')

    plt.tight_layout()
    save_plot(fig, filename_base)
    plt.close(fig)




def plot_fig5(hetero_data, filename_base="fig5_group_coop_vs_b"):
    """Plots Group-Specific Cooperation Rates vs. Temptation (b)."""
    print("Plotting Fig 5: Group Cooperation...")
    y_var_A = 'avg_CoopRate_A' # Must match reporter key
    y_var_B = 'avg_CoopRate_B' # Must match reporter key
    if y_var_A not in hetero_data.columns or y_var_B not in hetero_data.columns:
        print(f"Error plotting Fig 5: '{y_var_A}' or '{y_var_B}' not found.")
        print("Ensure group cooperation reporters were run.")
        return

    # Use melt for easier plotting with seaborn
    plot_df_melted = hetero_data.melt(
        id_vars=['b', 'run_id'], # Keep b and run_id for aggregation if needed
        value_vars=[y_var_A, y_var_B],
        var_name='Group Type',
        value_name='CooperationRate'
    )
    # Map reporter names to nicer labels for the legend
    plot_df_melted['Group Type'] = plot_df_melted['Group Type'].map({
        y_var_A: 'Group A ($C < 0.5$)', # Example definition
        y_var_B: 'Group B ($C \\geq 0.5$)' # Example definition
    })

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.lineplot(data=plot_df_melted, x='b', y='CooperationRate', hue='Group Type',
                 marker='o', # Use markers
                 markersize=mpl.rcParams['lines.markersize'],
                 linewidth=mpl.rcParams['lines.linewidth'],
                 errorbar=('ci', 95), # Uses underlying run_id variation
                 err_style="bars",
                 err_kws={'capsize': 3},
                 ax=ax)

    ax.set_title('Group-Specific Cooperation Rates')
    ax.set_xlabel('Temptation ($b$)')
    ax.set_ylabel('Average Group Cooperation Rate')
#    ax.grid(True, linestyle=mpl.rcParams['grid.linestyle'], alpha=mpl.rcParams['grid.alpha'])
    ax.set_ylim(-0.05, 1.05)
    ax.legend(title='Cultural Group') # Legend title

    plt.tight_layout()
    save_plot(fig, filename_base)
    plt.close(fig)

# ==============================================================================
# Snapshot Generation and Plotting (Physica A Style)
# ==============================================================================
def run_and_save_snapshot(params, b_value, filename_tag):
    """Runs a single simulation and saves the final grid state."""
    snapshot_filename = os.path.join(DATA_SAVE_DIR, f"snapshot_{filename_tag}_b{b_value:.2f}.pkl")

    if os.path.exists(snapshot_filename):
        print(f"Snapshot data already exists: {snapshot_filename}")
        return snapshot_filename

    print(f"Running simulation for snapshot (b={b_value:.2f})...")

    # Parameters for model initialization
    model_init_params = {
        'L': PARAMS['L'], 'initial_coop_ratio': PARAMS['initial_coop_ratio'],
        'K': PARAMS['K'], 'b': b_value, 'C_dist': params['C_dist'],
        'mu': params['mu'], 'sigma': params['sigma'], 'K_C': params['K_C'],
        'p_update_C': params['p_update_C'], 'p_mut': params['p_mut'],
        'seed': int(time.time() * 1000 + b_value * 10) % (2**32 - 1) # Seed variation
    }
    steps_to_run = PARAMS['steps']

    model = CulturalGame(**model_init_params)
    model.run_model(steps_to_run)

    grid_state = np.zeros((model.grid.width, model.grid.height, 3)) # x, y, [strategy, C, type]
    agent_types = {}

    threshold = 0.5 # Threshold for defining Type A/B
    TYPE_A = 1
    TYPE_B = 2

    for agent in model.schedule.agents:
        x, y = agent.pos
        if x is None or y is None: continue
        strategy = agent.strategy # 0 or 1
        culture = agent.C         # 0 to 1
        agent_type = TYPE_A if culture < threshold else TYPE_B
        grid_state[x, y, 0] = strategy
        grid_state[x, y, 1] = culture
        grid_state[x, y, 2] = agent_type
        if agent_type not in agent_types:
             agent_types[agent_type] = f"Type {'A' if agent_type == TYPE_A else 'B'} (C {'<' if agent_type == TYPE_A else '>='} {threshold})"

    snapshot_params_saved = model_init_params.copy()
    snapshot_params_saved['steps_run'] = steps_to_run

    snapshot_data = {'grid': grid_state, 'params': snapshot_params_saved, 'types': agent_types}
    with open(snapshot_filename, 'wb') as f:
        pickle.dump(snapshot_data, f)
    print(f"Snapshot data saved to {snapshot_filename}")
    return snapshot_filename

def plot_fig2_snapshot(snapshot_data_file, filename_base="fig2_snapshot"):
    """Plots Spatial Snapshot with Physica A style."""
    print(f"Plotting Fig 2: Snapshot from {snapshot_data_file}...")
    try:
        with open(snapshot_data_file, 'rb') as f:
            snapshot_data = pickle.load(f)
        grid = snapshot_data['grid']
        params = snapshot_data['params']
        L = params['L']
        b_val = params['b']
    except Exception as e:
        print(f"Error loading snapshot data from {snapshot_data_file}: {e}")
        return

    img = np.zeros((L, L, 3)) # RGB image
    TYPE_A = 1
    TYPE_B = 2
    STRAT_D = 0 # Defector
    STRAT_C = 1 # Cooperator

    # Define clearer colors, potentially check colorblind accessibility
    color_map = {
        (TYPE_A, STRAT_D): np.array([0.0, 0.0, 1.0]), # Blue (Indiv, Defect)
        (TYPE_A, STRAT_C): np.array([0.4, 0.8, 1.0]), # Light Blue (Indiv, Coop)
        (TYPE_B, STRAT_D): np.array([1.0, 0.0, 0.0]), # Red (Collect, Defect)
        (TYPE_B, STRAT_C): np.array([1.0, 0.6, 0.6]), # Pink/Light Red (Collect, Coop)
        # 'Empty':           [0, 0, 0]  # Black for empty cells if any
    }

    for x in range(L):
        for y in range(L):
            # Grid is (width, height) but imshow expects (height, width)
            # Access grid[x, y, :] but plot img[y, x, :] if using origin='lower'
            # Or transpose the grid/image later
            strat = int(grid[x, y, 0])
            agent_type = int(grid[x, y, 2])
            key = (agent_type, strat)
            if key in color_map:
                 # If origin='lower', y is the row index, x is the column index
                 img[y, x, :] = color_map[key]
            # else: img[y, x, :] = color_map['Empty'] # Handle empty if necessary

    fig, ax = plt.subplots(figsize=(5, 5)) # Square aspect ratio often good for grids
    # Use origin='lower' so (0,0) is bottom-left, consistent with grid coords
    # Use transpose img.transpose(1,0,2) if your img[x,y] setup requires it
    ax.imshow(img, origin='lower', interpolation='nearest')

    # Create custom legend handles - more robust
    legend_elements = [
        Patch(facecolor=color_map[(TYPE_A, STRAT_D)], edgecolor='k', linewidth=0.5, label='Group A / Defect'),
        Patch(facecolor=color_map[(TYPE_A, STRAT_C)], edgecolor='k', linewidth=0.5, label='Group A / Cooperate'),
        Patch(facecolor=color_map[(TYPE_B, STRAT_D)], edgecolor='k', linewidth=0.5, label='Group B / Defect'),
        Patch(facecolor=color_map[(TYPE_B, STRAT_C)], edgecolor='k', linewidth=0.5, label='Group B / Cooperate'),
    ]
    # Place legend outside the plot
    ax.legend(handles=legend_elements, bbox_to_anchor=(1.02, 1), loc='upper left', title="Agent State")

    ax.set_title(f'Spatial Snapshot ($b={b_val:.1f}$)') # Concise title with math
    ax.set_xticks([]) # No ticks for spatial grid
    ax.set_yticks([])

    # Adjust layout to prevent legend overlap
    plt.tight_layout(rect=[0, 0, 0.78, 1]) # Adjust right boundary for legend space
    save_plot(fig, f"{filename_base}_b{b_val:.1f}") # Include b in filename
    plt.close(fig)


# ==============================================================================
# Main Execution
# ==============================================================================
if __name__ == "__main__":
    print("Starting Simulation Runs and Plotting (Physica A Style)...")
    start_time = time.time()

    # --- Run Simulations (or load data) ---
    baseline_results_df = run_simulation_set(generate_baseline_configs, "baseline")
    heterogeneous_results_df = run_simulation_set(generate_heterogeneous_configs, "heterogeneous")

    # --- Generate Plots ---
    # Ensure dataframes are not empty before plotting
    if not baseline_results_df.empty:
        # Sort baseline data by C value for consistent plot legend order
        baseline_results_df['c_value'] = baseline_results_df['label'].str.extract(r'C=([\d.]+)').astype(float)
        baseline_results_df = baseline_results_df.sort_values('c_value')
        plot_fig1(baseline_results_df) # Uses default filename base
    else:
        print("Skipping Fig 1 due to missing/empty baseline data.")

    if not heterogeneous_results_df.empty:
        plot_fig3(heterogeneous_results_df) # Uses default filename base
        plot_fig5(heterogeneous_results_df) # Uses default filename base
    else:
        print("Skipping Figs 3 & 5 due to missing/empty heterogeneous data.")

    if not baseline_results_df.empty and not heterogeneous_results_df.empty:
         # Pass the sorted baseline data
        plot_fig4(baseline_results_df, heterogeneous_results_df) # Uses default filename base
    else:
        print("Skipping Fig 4 due to missing/empty baseline or heterogeneous data.")

    # --- Generate and Plot Snapshots (Fig 2) ---
    snapshot_files = []
    print("\nGenerating Snapshots...")
    for b_snap in PARAMS['snapshot_b_values']:
        fname = run_and_save_snapshot(PARAMS['snapshot_hetero_params'], b_snap, "hetero")
        if fname: # Check if file was actually created or loaded
            snapshot_files.append(fname)

    print("\nPlotting Snapshots...")
    if not snapshot_files:
        print("No snapshot files found or generated to plot.")
    for snap_file in snapshot_files:
         # Filename base for snapshot plots will be derived inside the function
         plot_fig2_snapshot(snap_file) # Function now handles filename generation


    end_time = time.time()
    print(f"\nTotal script execution time: {(end_time - start_time):.2f} seconds")
    print(f"--- Finished. Plots saved in '{PLOT_SAVE_DIR}' directory ---")
