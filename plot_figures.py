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
from scipy.optimize import curve_fit # For fitting P(s) if needed
from collections import Counter # For processing cluster sizes

# --- Import Core Simulation Logic ---
RUN_WITH_PARALLEL = False # Default to False, check for parallel module
try:
    from utils.parallel import batch_run_parallel
    from utils.config import SimConfig
    from core.model import CulturalGame
    RUN_WITH_PARALLEL = True
    print("Using parallel.py for simulation runs.")
except ImportError:
    print("Warning: parallel.py or SimConfig not found. Falling back to sequential runs.")
    try:
        from core.model import CulturalGame
        from utils.config import SimConfig # Still need SimConfig for sequential mode
    except ImportError as e:
        print(f"Error importing core modules for sequential fallback: {e}")
        exit()

# --- Ensure Reporters are Accessible (Needed by snapshot logic) ---
try:
    from utils.reporters import _get_agent_type
except ImportError:
    def _get_agent_type(agent, threshold=0.5):
         if not hasattr(agent, 'C'): return None
         return 'A' if agent.C < threshold else 'B'
    print("Warning: Could not import reporters._get_agent_type locally, using placeholder.")


# ==============================================================================
# Simulation Parameters (Revised for Physica A Analysis)
# ==============================================================================
PARAMS = {
    # --- Core Model Parameters ---
    'initial_coop_ratio': 0.5,
    'K': 0.1,           # Strategy noise
    'K_C': 0.1,         # Cultural noise
    'p_update_C': 0.1,  # Cultural update probability
    'p_mut': 0.001,     # Cultural mutation rate
    'C_dist': 'bimodal',# Initial distribution ('uniform', 'normal', 'bimodal', 'fixed')
    'mu': 0.5,          # Meaning depends on C_dist (e.g., p(C=1) for bimodal)
    'sigma': 0.1,       # Std Dev for C_dist='normal'

    # --- Simulation Control ---
    'steps': 2000,        # Adjust steps for steady state
    'steady_state_window': 100, # Average over last N steps (for avg reporters)
    'runs_per_parameter_set': 10, # CRUCIAL FOR STATS & CHI

    # --- Scan Parameters ---
    'L_values': [20, 30, 40, 50], # System sizes for FSS (adjust as needed)
    'b_values': np.linspace(1.3, 7, 12), # Temptation 'b' (refine near transition!)

    # --- Snapshot Specific ---
    'snapshot_L': 50, # L for snapshots (usually one of the larger ones)
    'snapshot_b_values': [1.8, 2.3, 2.8], # Example b values

    # --- Cluster Analysis Specific ---
    'cluster_analysis_b_values': [1.6, 1.7, 1.8, 1.9, 2.0], # b values near transition for P(s)

    # --- Phase Diagram Specific (Example: Scan b vs K_C) ---
    'phasediagram_L': 40,        # Fixed L for phase diagram
    'phasediagram_param1_name': 'b',
    'phasediagram_param1_values': np.linspace(1.3, 7.0, 12),
    'phasediagram_param2_name': 'K_C',
    'phasediagram_param2_values': np.linspace(0.01, 0.5, 16),
    'phasediagram_target_reporter': 'avg_CooperationRate', # What to plot

    # --- FSS Parameters (REVISE after seeing peaks!) ---
    'fss_bc_estimate': 2.1,          # Initial guess for critical b (Cooperation)
    # IMPORTANT: These should be estimates for the *actual* exponents, not ratios
    'fss_beta_estimate': 0.125,      # Example: ~Ising 2D beta=1/8
    'fss_gamma_estimate': 1.75,      # Example: ~Ising 2D gamma=7/4
    'fss_nu_estimate': 1.0,          # Example: ~Ising 2D nu=1
}

DATA_SAVE_DIR = "simulation_data_physicaA"
PLOT_SAVE_DIR = "plots_physicaA"
os.makedirs(DATA_SAVE_DIR, exist_ok=True)
os.makedirs(PLOT_SAVE_DIR, exist_ok=True)

# Use specific data filenames
MAIN_DATA_FILENAME = os.path.join(DATA_SAVE_DIR, "main_scan_data.pkl")
# CLUSTER_DATA_FILENAME = os.path.join(DATA_SAVE_DIR, "cluster_data.pkl") # Cluster data is now part of main data
PHASEDIAGRAM_DATA_FILENAME = os.path.join(DATA_SAVE_DIR, "phasediagram_data.pkl")
SNAPSHOT_DATA_DIR = os.path.join(DATA_SAVE_DIR, "snapshots")
os.makedirs(SNAPSHOT_DATA_DIR, exist_ok=True)


# ==============================================================================
# Physica A Style Settings
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
MARKERS = ['o', 's', '^', 'D', 'v', 'p', '*', 'X']
LINESTYLES = ['-', '--', ':', '-.']

def get_style_kwargs(index, num_items, base_markersize=4):
    """ Get color, marker, linestyle based on index """
    palette = sns.color_palette("viridis", n_colors=num_items)
    return {
        'color': palette[index % num_items],
        'marker': MARKERS[index % len(MARKERS)],
        'linestyle': LINESTYLES[(index // len(MARKERS)) % len(LINESTYLES)],
        'markersize': base_markersize # Return markersize too
    }

# ==============================================================================
# Simulation Running Functions
# ==============================================================================

def generate_main_scan_configs():
    """Generates SimConfig objects for the main L vs b scan."""
    configs = []
    # Base parameters excluding those being swept or set per run
    # ADD 'steady_state_window' to the exclusion list
    base_params = {k: v for k, v in PARAMS.items() if k not in ['L_values', 'b_values', 'runs_per_parameter_set', 'steps', 'seed', 'param_set_id', 'run_id', 'steady_state_window'] and not k.startswith('snapshot_') and not k.startswith('cluster_') and not k.startswith('phasediagram_') and not k.startswith('fss_')}
    base_params['steps'] = PARAMS['steps'] # Add steps back

    param_list = []
    for L in PARAMS['L_values']:
        for b in PARAMS['b_values']:
            current_params = base_params.copy()
            current_params['L'] = L
            current_params['b'] = b
            current_params['param_set_id'] = f"L{L}_b{b:.3f}" # Unique ID for (L, b)
            param_list.append(current_params)

    # Create SimConfig objects for each parameter set, repeated for each run
    for i in range(PARAMS['runs_per_parameter_set']):
         run_seed_base = i * 10000 # Base seed for this run index
         for idx, params in enumerate(param_list):
              run_params = params.copy()
              run_params['seed'] = run_seed_base + idx # Unique seed per run
              run_params['run_id'] = i # Store the run index
              try:
                   configs.append(SimConfig(**run_params))
              except TypeError as e:
                   print(f"Error creating SimConfig. Params: {run_params}\nError: {e}")
                   # You might want to investigate why this fails if it happens

    print(f"Generated {len(configs)} SimConfig objects for main scan.")
    return configs


def generate_phasediagram_configs():
    """ Generates configs for phase diagram scan """
    configs = []
    # Base parameters excluding those being swept or set per run
    # ADD 'steady_state_window' to the exclusion list
    base_params = {k: v for k, v in PARAMS.items() if k not in ['L_values', 'b_values', 'runs_per_parameter_set', 'steps', 'seed', 'param_set_id', 'run_id', 'steady_state_window'] and not k.startswith('snapshot_') and not k.startswith('cluster_') and not k.startswith('phasediagram_') and not k.startswith('fss_')}
    base_params['steps'] = PARAMS['steps']
    base_params['L'] = PARAMS['phasediagram_L'] # Fixed L for phase diagram

    param_list = []
    p1_name = PARAMS['phasediagram_param1_name']
    p2_name = PARAMS['phasediagram_param2_name']
    p1_values = PARAMS['phasediagram_param1_values']
    p2_values = PARAMS['phasediagram_param2_values']

    for p1_val in p1_values:
        for p2_val in p2_values:
            current_params = base_params.copy()
            current_params[p1_name] = p1_val
            current_params[p2_name] = p2_val
            current_params['param_set_id'] = f"PhaseD_L{current_params['L']}_{p1_name}{p1_val:.2f}_{p2_name}{p2_val:.3f}"
            param_list.append(current_params)

    # Fewer runs might be okay for phase diagram, adjust if needed
    num_runs = max(1, PARAMS['runs_per_parameter_set'] // 2)

    for i in range(num_runs):
         run_seed_base = i * 10000 + 7000 # Offset seed
         for idx, params in enumerate(param_list):
              run_params = params.copy()
              run_params['seed'] = run_seed_base + idx
              run_params['run_id'] = i
              configs.append(SimConfig(**run_params))
    print(f"Generated {len(configs)} SimConfig objects for phase diagram (runs per point: {num_runs}).")
    return configs


def run_simulation_batch(config_generator, data_filename, force_rerun=False):
    """Runs a batch of simulations using parallel.py or sequentially."""
    if not force_rerun and os.path.exists(data_filename):
        print(f"Loading existing data from {data_filename}...")
        try:
            with open(data_filename, 'rb') as f:
                results_df = pickle.load(f)
            print(f"Loaded {len(results_df)} results.")
            return results_df
        except Exception as e:
            print(f"Error loading data from {data_filename}: {e}. Rerunning...")
            force_rerun = True # Force rerun if loading fails

    print(f"Running simulations for {data_filename}...")
    sim_configs = config_generator()
    if not sim_configs:
         print("Warning: No configurations generated.")
         return pd.DataFrame()

    if RUN_WITH_PARALLEL:
        results_df = batch_run_parallel(
            sim_configs,
            steady_state_window=PARAMS['steady_state_window']
            # num_workers defaults in batch_run_parallel
        )
    else:
        # Fallback Sequential Run
        print("Running sequentially (This might be slow)...")
        all_results = []
        # NON_AVERAGE_REPORTERS must match the one in parallel.py's run_single_simulation
        NON_AVERAGE_REPORTERS = ["ClusterSizeDistribution"]
        with tqdm(total=len(sim_configs), desc=f"Sims ({os.path.basename(data_filename)})") as pbar:
             for config in sim_configs:
                 try:
                      model = CulturalGame(**config.to_dict())
                      for _ in range(config.steps):
                          model.step()

                      model_df = model.datacollector.get_model_vars_dataframe()
                      result_dict = config.to_dict()
                      n_rows = len(model_df)

                      if n_rows > 0:
                          start_idx = max(0, n_rows - PARAMS['steady_state_window'])
                          window_df = model_df.iloc[start_idx:]

                          for col in model.datacollector.model_reporters.keys():
                              if col in model_df.columns:
                                  if col in NON_AVERAGE_REPORTERS:
                                      result_dict[f"{col}"] = model_df[col].iloc[-1]
                                  elif not window_df.empty:
                                      result_dict[f"avg_{col}"] = window_df[col].mean()
                                      result_dict[f"std_{col}"] = window_df[col].std()
                                  else:
                                      result_dict[f"avg_{col}"] = np.nan
                                      result_dict[f"std_{col}"] = np.nan
                                      if col in NON_AVERAGE_REPORTERS: result_dict[f"{col}"] = None
                              else: # Reporter column missing
                                  result_dict[f"avg_{col}"] = np.nan
                                  result_dict[f"std_{col}"] = np.nan
                                  if col in NON_AVERAGE_REPORTERS: result_dict[f"{col}"] = None
                      else: # No data collected
                           for col in model.datacollector.model_reporters.keys():
                                result_dict[f"avg_{col}"] = np.nan
                                result_dict[f"std_{col}"] = np.nan
                                if col in NON_AVERAGE_REPORTERS: result_dict[f"{col}"] = None

                      all_results.append(result_dict)

                 except Exception as e:
                      print(f"\nError running config {config.param_set_id} (run {config.run_id}): {e}")
                      import traceback
                      traceback.print_exc()
                      # Optionally store None or skip this run's result
                 finally:
                      pbar.update(1)

        results_df = pd.DataFrame(all_results)

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
# Data Processing Functions
# ==============================================================================

def process_main_data(raw_df):
    """
    Calculates mean, SEM, susceptibility, and processes cluster data from raw run data.
    Cluster data ('ClusterSizeDistribution') is expected to be present as raw output
    (e.g., dict from last step) for each run.
    """
    if raw_df.empty:
        return pd.DataFrame(), None # Return empty DF and None for clusters

    processed_data = []
    cluster_results = {} # Aggregated cluster sizes: Key: (L, b), Value: {'A': all_sizes, 'B': all_sizes}

    # Group by the parameters that define a unique simulation setting
    grouping_params = ['L', 'b'] # Add others if they are swept in the main scan

    # Reporters for mean/SEM/Susceptibility calculation
    # Identify reporters that have 'avg_' prefix (meaning they were averaged by default)
    avg_reporters = [col for col in raw_df.columns if col.startswith('avg_')]
    avg_reporter_names = [col.split('avg_')[1] for col in avg_reporters] # e.g., 'CooperationRate'

    # Reporters for which we want susceptibility (typically order parameters)
    susceptibility_target_names = ['CooperationRate', 'SegregationIndex'] # Base names

    # Special handling for non-averaged reporters
    cluster_col_name = "ClusterSizeDistribution" # Exact name from parallel/sequential run output

    grouped = raw_df.groupby(grouping_params)

    print("Processing main scan data (means, SEM, susceptibility, clusters)...")
    for name, group in tqdm(grouped, desc="Processing (L, b) groups"):
        processed_point = dict(zip(grouping_params, name)) # Store L, b
        L = processed_point['L']
        N = L * L # Number of agents

        # --- Calculate Mean and SEM for averaged reporters ---
        for reporter_name in avg_reporter_names:
            avg_col = f"avg_{reporter_name}"
            if avg_col in group.columns:
                 mean_val = group[avg_col].mean()
                 sem_val = group[avg_col].sem() # Standard Error of the Mean
                 processed_point[avg_col] = mean_val
                 processed_point[f"sem_{reporter_name}"] = sem_val
            else: # Should not happen if data generation is correct
                 processed_point[avg_col] = np.nan
                 processed_point[f"sem_{reporter_name}"] = np.nan

        # --- Calculate Susceptibility (Chi) = N * Var(<Reporter>) ---
        for reporter_name in susceptibility_target_names:
             avg_col = f"avg_{reporter_name}" # Column containing the mean value *for each run*
             if avg_col in group.columns:
                 # Variance of the run averages
                 variance_across_runs = group[avg_col].var()
                 # Susceptibility
                 chi_val = N * variance_across_runs
                 processed_point[f"chi_{reporter_name}"] = chi_val
             else:
                 processed_point[f"chi_{reporter_name}"] = np.nan
                 # print(f"Warning: Column {avg_col} not found for susceptibility calc at {name}.")


        # --- Aggregate Cluster Data ---
        if cluster_col_name in group.columns:
            all_sizes_A = []
            all_sizes_B = []
            # Iterate through each run in the group
            for idx, row in group.iterrows():
                 dist_dict = row[cluster_col_name] # Get the dict {'A':[], 'B':[]} from this run
                 if isinstance(dist_dict, dict):
                     all_sizes_A.extend(dist_dict.get('A', []))
                     all_sizes_B.extend(dist_dict.get('B', []))
                 # else: print(f"Warning: Unexpected data type in {cluster_col_name} at {name}, run {row.get('run_id', '?')}")

            # Store aggregated sizes for this (L, b) point
            if name not in cluster_results: cluster_results[name] = {'A': [], 'B': []}
            cluster_results[name]['A'].extend(all_sizes_A)
            cluster_results[name]['B'].extend(all_sizes_B)
        # else: print(f"Warning: Column '{cluster_col_name}' not found at {name}.")


        processed_data.append(processed_point)

    processed_df = pd.DataFrame(processed_data)

    # --- Calculate P(s) from aggregated cluster_results ---
    print("Calculating P(s) from aggregated cluster sizes...")
    processed_ps = {} # Key: (L, b, type), Value: (bin_centers, ps_values)
    # Use log binning parameters (adjust factor as needed)
    log_bin_factor = 1.4

    for (L_b_tuple), sizes_dict in tqdm(cluster_results.items(), desc="Calculating P(s)"):
        L, b = L_b_tuple
        for type_label, all_sizes in sizes_dict.items():
            if not all_sizes: continue

            counts = Counter(all_sizes)
            total_clusters = len(all_sizes) # Total clusters of this type for this (L,b) across all runs
            if total_clusters == 0: continue

            s_values = np.array(list(counts.keys()))
            raw_counts = np.array([counts[s] for s in s_values]) # Raw counts for each size s

            # --- Log Binning ---
            if len(s_values) < 2: # Not enough data for log binning
                 if len(s_values) == 1 and total_clusters > 0: # Single size found
                      # P(s) = count(s) / total_clusters (bin width is implicitly 1)
                      processed_ps[(L, b, type_label)] = (s_values, raw_counts / total_clusters)
                 continue

            min_s, max_s = s_values.min(), s_values.max()
            if max_s <= min_s or min_s <= 0: continue # Need positive range

            log_min = np.log10(min_s)
            log_max = np.log10(max_s)
            # Estimate number of bins
            num_bins = int(np.ceil((log_max - log_min) / np.log10(log_bin_factor)))
            if num_bins <= 0: num_bins = 1

            # Create bin edges on log scale, convert back, ensure uniqueness and integer type
            bins = np.unique(np.int64(np.ceil(np.logspace(log_min, log_max, num_bins + 1))))

            if len(bins) < 2 : continue # Need at least two bin edges

            # Bin the data points (s_values) according to counts (raw_counts)
            # We need to sum counts within each bin
            bin_indices = np.digitize(s_values, bins, right=False) - 1 # Get bin index for each s_value

            binned_s_centers = []
            binned_ps_values = []

            for i in range(len(bins) - 1): # Iterate through bins
                bin_mask = (bin_indices == i)
                s_in_bin = s_values[bin_mask]
                counts_in_bin = raw_counts[bin_mask]

                total_count_in_bin = np.sum(counts_in_bin)
                if total_count_in_bin == 0: continue

                # Calculate bin center (geometric mean recommended for log bins)
                # Avoid log(0) if s can be 0 (though cluster size shouldn't be)
                log_s_in_bin = np.log10(s_in_bin + 1e-9)
                avg_log_s = np.average(log_s_in_bin, weights=counts_in_bin)
                bin_center = 10**avg_log_s

                # Calculate bin width (linear scale)
                bin_width = bins[i+1] - bins[i]
                if bin_width <= 0: continue # Skip zero-width bins

                # Calculate P(s) density for the bin
                # P(s) = (total count in bin) / (total number of clusters * bin width)
                ps_value = total_count_in_bin / (total_clusters * bin_width)

                binned_s_centers.append(bin_center)
                binned_ps_values.append(ps_value)

            if binned_s_centers:
                processed_ps[(L, b, type_label)] = (np.array(binned_s_centers), np.array(binned_ps_values))
            # --- End Log Binning ---

    print(f"Finished processing. Found {len(processed_df)} (L,b) points.")
    print(f"Calculated P(s) for {len(processed_ps)} cases.")
    return processed_df, processed_ps # Return both processed averages and P(s) data


# ==============================================================================
# Plotting Functions (Physica A Style)
# ==============================================================================

def save_plot(fig, base_filename, plot_dir=PLOT_SAVE_DIR):
    """Helper function to save plot."""
    png_path = os.path.join(plot_dir, f"{base_filename}.png")
    pdf_path = os.path.join(plot_dir, f"{base_filename}.pdf") # Save PDF for quality
    try:
        fig.savefig(png_path, dpi=300, bbox_inches='tight')
        fig.savefig(pdf_path, bbox_inches='tight')
    except Exception as e:
        print(f"Error saving plot {base_filename}: {e}")

# --- Figure 1: Order Parameter vs b (Multi-L) ---
def plot_fig1_order_param(data, order_param_key_avg, sem_key, ylabel, filename_base):
    """ Plots an order parameter vs b for different L values. """
    print(f"Plotting Order Parameter: {order_param_key_avg}...")
    if data.empty or order_param_key_avg not in data.columns:
        print(f"Error plotting {filename_base}: Data missing or '{order_param_key_avg}' column not found.")
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    Ls = sorted(data['L'].unique())
    base_markersize = mpl.rcParams['lines.markersize']

    for i, L in enumerate(Ls):
        L_data = data[data['L'] == L].sort_values('b')
        if L_data.empty: continue
        style_kwargs = get_style_kwargs(i, len(Ls), base_markersize)
        label = f'$L={L}$'

        # Check if SEM data exists and is valid
        yerr = None
        if sem_key in L_data.columns:
            yerr_data = L_data[sem_key].replace([np.inf, -np.inf], np.nan).fillna(0) # Handle NaN/inf SEM
            if not np.all(yerr_data == 0): # Only plot error bars if SEM is meaningful
                 yerr = yerr_data

        ax.errorbar(L_data['b'], L_data[order_param_key_avg],
                    yerr=yerr,
                    label=label,
                    linestyle='-', # Line connecting points
                    capsize=3, elinewidth=1,
                    # Pass style kwargs directly
                    color=style_kwargs['color'],
                    marker=style_kwargs['marker'],
#                    linestyle=style_kwargs['linestyle'],
                    markersize=style_kwargs['markersize']
                   )

    ax.set_xlabel('Temptation ($b$)')
    ax.set_ylabel(ylabel)
    title_str = ylabel.split("$\\langle$")[-1].split("$\\rangle$")[0] # Extract name like f_C or S
    ax.set_title(f'Order Parameter $\\langle {title_str} \\rangle$ vs. Temptation $b$')
    ax.legend(title='System Size $L$', loc='best', frameon=False)
    # Sensible limits (adjust if needed)
    if "Rate" in ylabel or "Index" in ylabel: ax.set_ylim(-0.05, 1.05)
    ax.grid(False) # Physica A often prefers no grid

    save_plot(fig, filename_base)
    plt.close(fig)


# --- Figure 2: Susceptibility vs b (Multi-L) ---
def plot_fig2_susceptibility(data, chi_key, ylabel, filename_base):
    """ Plots susceptibility vs b for different L values. """
    print(f"Plotting Susceptibility: {chi_key}...")
    if data.empty or chi_key not in data.columns:
        print(f"Error plotting {filename_base}: Data missing or '{chi_key}' column not found.")
        return pd.Series(dtype=float) # Return empty Series if error

    fig, ax = plt.subplots(figsize=(6, 4))
    Ls = sorted(data['L'].unique())
    base_markersize = mpl.rcParams['lines.markersize']
    peak_locs = {} # Store peak locations {L: b_peak}

    for i, L in enumerate(Ls):
        L_data = data[data['L'] == L].sort_values('b')
        L_data = L_data.replace([np.inf, -np.inf], np.nan).dropna(subset=[chi_key]) # Clean data
        if L_data.empty: continue

        style_kwargs = get_style_kwargs(i, len(Ls), base_markersize)
        label = f'$L={L}$'
        ax.plot(L_data['b'], L_data[chi_key], label=label, **style_kwargs) # Apply style

        # Find peak location
        if not L_data[chi_key].empty:
             peak_idx = L_data[chi_key].idxmax()
             # Ensure index exists before accessing loc
             if peak_idx in L_data.index:
                  peak_locs[L] = L_data.loc[peak_idx, 'b']
                  # Optional: Mark peaks
                  # ax.plot(peak_locs[L], L_data.loc[peak_idx, chi_key], marker='x', color=style_kwargs['color'], markersize=8)
             else:
                  peak_locs[L] = np.nan # Indicate peak not found


    ax.set_xlabel('Temptation ($b$)')
    ax.set_ylabel(ylabel)
#    title_str = ylabel.split("$\\chi_{")[-1].split("}$")[0] # Extract name like f_C or S
#    ax.set_title(f'Susceptibility $\\chi_{{{param_name}}}$ vs. Temptation $b$') 
    ax.legend(title='System Size $L$', loc='best', frameon=False)
    # ax.set_yscale('log') # Consider log scale if peaks vary widely
    ax.grid(False)

    save_plot(fig, filename_base)
    plt.close(fig)
    print(f"Susceptibility Peak Locations ({chi_key}): {peak_locs}")
    return pd.Series(peak_locs) # Return as Series

# --- Figures 3, 4, 5: Finite-Size Scaling (Plotting Function Corrected) ---
def plot_fss_analysis(
    data: pd.DataFrame,
    op_col_avg: str, # e.g., 'avg_CooperationRate'
    sus_col: str,    # e.g., 'chi_CooperationRate'
    b_param_col: str, # Name of column holding 'b' values, e.g., 'b'
    L_param_col: str, # Name of column holding 'L' values, e.g., 'L'
    b_c: float,       # Critical point estimate
    beta: float,      # Critical exponent beta
    gamma: float,     # Critical exponent gamma
    nu: float,        # Critical exponent nu
    filename_base_prefix: str = "fss_analysis",
    output_dir: str = PLOT_SAVE_DIR, # Use global plot dir
    op_label: str = r"$\langle O \rangle$", # Generic labels, customize if needed
    sus_label: str = r"$\chi$",
    b_label: str = r"$b$",
    op_scaled_label: str = r"$\langle O \rangle L^{\beta/\nu}$",
    sus_scaled_label: str = r"$\chi L^{-\gamma/\nu}$",
    x_scaled_label: str = r"$(b - b_c) L^{1/\nu}$",
    base_markersize: int = mpl.rcParams['lines.markersize'], # Use base size
    collapse_markersize_scale: float = 1.0 # Scale factor for collapse markers (1.0 means same size)
):
    """
    Performs Finite-Size Scaling (FSS) analysis and plots:
    1. Order parameter at b_c vs L (log-log) -> Fig 3 estimate beta/nu
    2. Susceptibility peak vs L (log-log) -> Fig 4 estimate gamma/nu
    3. Data collapse plots (linear-linear axes) -> Fig 5
    """
    print(f"Performing FSS analysis for op='{op_col_avg}', sus='{sus_col}' around b_c={b_c:.4f}")
    if data.empty or not all(col in data.columns for col in [op_col_avg, sus_col, b_param_col, L_param_col]):
        print(f"Error in FSS: DataFrame empty or missing required columns: {op_col_avg}, {sus_col}, {b_param_col}, {L_param_col}")
        return

    # --- Preparations ---
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    unique_Ls = sorted(data[L_param_col].unique())
    num_L = len(unique_Ls)
    if num_L == 0:
        print("Error in FSS: No unique L values found in data.")
        return

    # Data storage for log-log plots
    fss_loglog_data = {'L': [], 'op_at_bc': [], 'sus_peak': [], 'sus_peak_b': []}

    # --- Create Figure for Collapse (Fig 5) ---
    fig_collapse, axes_collapse = plt.subplots(1, 2, figsize=(12, 5))
    ax_op_collapse, ax_sus_collapse = axes_collapse

    # --- Loop through L to gather data and plot collapse ---
    print("Processing data for FSS plots...")
    all_b_values = data[b_param_col].unique() # For finding closest b to bc
    for i, L in enumerate(tqdm(unique_Ls, desc="FSS per L")):
        subset = data[data[L_param_col] == L].sort_values(by=b_param_col).reset_index()
        if subset.empty: continue

        # --- Data for Log-Log Plots (Fig 3 & 4) ---
        # Order parameter at b_c (find closest b value in data)
        b_closest_idx = (subset[b_param_col] - b_c).abs().idxmin()
        op_at_bc = subset.loc[b_closest_idx, op_col_avg]

        # Susceptibility peak value and location
        sus_peak_idx = subset[sus_col].idxmax()
        sus_peak = subset.loc[sus_peak_idx, sus_col]
        sus_peak_b = subset.loc[sus_peak_idx, b_param_col]

        # Store log-log data if valid
        if pd.notna(op_at_bc) and pd.notna(sus_peak) and pd.notna(L):
            fss_loglog_data['L'].append(L)
            fss_loglog_data['op_at_bc'].append(op_at_bc)
            fss_loglog_data['sus_peak'].append(sus_peak)
            fss_loglog_data['sus_peak_b'].append(sus_peak_b)

        # --- Data for Collapse Plot (Fig 5) ---
        b_values = subset[b_param_col].values
        op_values = subset[op_col_avg].values
        sus_values = subset[sus_col].values

        # Filter invalid values for scaling (e.g., NaN, Inf, non-positive for logs)
        valid_indices = pd.notna(op_values) & pd.notna(sus_values) & (op_values > 1e-9) & (sus_values > 1e-9)
        if not np.any(valid_indices) or np.abs(nu) < 1e-9:
             continue # Skip if no valid data or nu is zero

        b_valid = b_values[valid_indices]
        op_valid = op_values[valid_indices]
        sus_valid = sus_values[valid_indices]

        # Calculate scaled variables
        t = b_valid - b_c # Reduced parameter
        x_scaled = t * (L**(1/nu))
        y_op_scaled = op_valid * (L**(beta/nu))
        y_sus_scaled = sus_valid * (L**(-gamma/nu))

        # Plot collapse data
        style_kwargs = get_style_kwargs(i, num_L, base_markersize)
        # Make collapse markers slightly different if desired
        current_collapse_markersize = base_markersize * collapse_markersize_scale
        ax_op_collapse.plot(x_scaled, y_op_scaled,
                            marker=style_kwargs['marker'],
                            color=style_kwargs['color'],
                            linestyle='', # Points only for collapse
                            markersize=current_collapse_markersize,
                            label=f"$L={L}$" if i == 0 else "_nolegend_") # Label only first
        ax_sus_collapse.plot(x_scaled, y_sus_scaled,
                             marker=style_kwargs['marker'],
                             color=style_kwargs['color'],
                             linestyle='',
                             markersize=current_collapse_markersize,
                             label=f"$L={L}$" if i == 0 else "_nolegend_")


    # --- Finalize Collapse Plot (Fig 5) ---
    ax_op_collapse.set_xlabel(x_scaled_label)
    ax_op_collapse.set_ylabel(op_scaled_label)
    ax_op_collapse.set_title(f"Scaled Order Parameter ({op_label})")
    ax_op_collapse.grid(True, linestyle=':', alpha=0.6)
    # ax_op_collapse.legend(title='$L$', fontsize='small') # Optional legend for collapse

    ax_sus_collapse.set_xlabel(x_scaled_label)
    ax_sus_collapse.set_ylabel(sus_scaled_label)
    ax_sus_collapse.set_title(f"Scaled Susceptibility ({sus_label})")
    ax_sus_collapse.grid(True, linestyle=':', alpha=0.6)
    # ax_sus_collapse.legend(title='$L$', fontsize='small')

    fig_collapse.suptitle(f"Data Collapse ($b_c={b_c:.4f}, \\beta={beta:.3f}, \\gamma={gamma:.3f}, \\nu={nu:.3f}$)", fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_plot(fig_collapse, f"{filename_base_prefix}_fig5_collapse")
    plt.close(fig_collapse)

    # --- Create and Plot Log-Log Plots (Fig 3 & 4) ---
    if not fss_loglog_data['L']:
        print("Warning: No valid data collected for FSS log-log plots.")
        return

    df_loglog = pd.DataFrame(fss_loglog_data).replace(0, 1e-9) # Avoid log(0)

    # --- Fig 3: Order Parameter Scaling ---
    fig_log_op, ax_log_op = plt.subplots(figsize=(6, 4.5))
    ax_log_op.plot(df_loglog['L'], df_loglog['op_at_bc'], marker='o', linestyle='None', color='blue')

    # Fit line: log(op) = C - (beta/nu) * log(L)
    if len(df_loglog) > 1:
        log_L = np.log10(df_loglog['L'])
        log_op = np.log10(df_loglog['op_at_bc'])
        valid_fit = pd.notna(log_L) & pd.notna(log_op)
        if np.sum(valid_fit) > 1:
             coeffs = np.polyfit(log_L[valid_fit], log_op[valid_fit], 1)
             beta_nu_fit = -coeffs[0]
             fit_line = 10**(coeffs[1] + coeffs[0] * log_L)
             ax_log_op.plot(df_loglog['L'], fit_line, 'r--', label=f'Fit: $\\beta/\\nu \\approx {beta_nu_fit:.3f}$')
             ax_log_op.legend()
        else: beta_nu_fit = np.nan
    else: beta_nu_fit = np.nan

    ax_log_op.set_xlabel('System Size $L$')
    ax_log_op.set_ylabel(f'Order Parameter at $b_c$, {op_label}$(b_c)$')
    ax_log_op.set_xscale('log')
    ax_log_op.set_yscale('log')
    ax_log_op.set_title(f'Fig 3: Order Parameter Scaling at $b_c$')
    ax_log_op.grid(True, which='both', linestyle=':', alpha=0.6)
    save_plot(fig_log_op, f"{filename_base_prefix}_fig3_op_scaling")
    plt.close(fig_log_op)

    # --- Fig 4: Susceptibility Scaling ---
    fig_log_sus, ax_log_sus = plt.subplots(figsize=(6, 4.5))
    ax_log_sus.plot(df_loglog['L'], df_loglog['sus_peak'], marker='s', linestyle='None', color='green')

    # Fit line: log(chi) = C + (gamma/nu) * log(L)
    if len(df_loglog) > 1:
        log_L = np.log10(df_loglog['L'])
        log_sus = np.log10(df_loglog['sus_peak'])
        valid_fit = pd.notna(log_L) & pd.notna(log_sus)
        if np.sum(valid_fit) > 1:
             coeffs = np.polyfit(log_L[valid_fit], log_sus[valid_fit], 1)
             gamma_nu_fit = coeffs[0]
             fit_line = 10**(coeffs[1] + coeffs[0] * log_L)
             ax_log_sus.plot(df_loglog['L'], fit_line, 'r--', label=f'Fit: $\\gamma/\\nu \\approx {gamma_nu_fit:.3f}$')
             ax_log_sus.legend()
        else: gamma_nu_fit = np.nan
    else: gamma_nu_fit = np.nan

    ax_log_sus.set_xlabel('System Size $L$')
    ax_log_sus.set_ylabel(f'Susceptibility Peak Value, max({sus_label})')
    ax_log_sus.set_xscale('log')
    ax_log_sus.set_yscale('log')
    ax_log_sus.set_title(f'Fig 4: Susceptibility Peak Scaling')
    ax_log_sus.grid(True, which='both', linestyle=':', alpha=0.6)
    save_plot(fig_log_sus, f"{filename_base_prefix}_fig4_sus_scaling")
    plt.close(fig_log_sus)

    print(f"FSS Fits: beta/nu ~ {beta_nu_fit:.3f}, gamma/nu ~ {gamma_nu_fit:.3f}")
    print("Compare these fits to your input beta/nu and gamma/nu estimates.")
    # Optional: Plot peak location bc(L) vs L -> estimate nu
    # fig_bcL, ax_bcL = plt.subplots(figsize=(6, 4))
    # ax_bcL.plot(df_loglog['L']**(-1/nu), df_loglog['sus_peak_b'], 'd-') # Example plot bc(L) vs L^(-1/nu)
    # save_plot(fig_bcL, f"{filename_base_prefix}_figX_bc_scaling")
    # plt.close(fig_bcL)


# --- Figure 6: Cluster Size Distribution P(s) ---
def plot_fig6_cluster_dist(processed_ps_data, filename_base):
    """ Plots P(s) vs s on log-log axes for selected parameters. """
    print("Plotting Cluster Size Distribution P(s)...")
    if not processed_ps_data:
        print("Error plotting Fig 6: Processed P(s) data is missing or empty.")
        return

    # Select parameters to plot (e.g., near critical point, largest L?)
    # Let's plot multiple L for a specific b near the estimated critical point
    target_b = PARAMS.get('fss_bc_estimate', PARAMS['cluster_analysis_b_values'][0])
    # Find closest b in the actual data if needed
    available_bs = sorted(list(set(k[1] for k in processed_ps_data.keys())))
    if available_bs:
         target_b_actual = min(available_bs, key=lambda x:abs(x-target_b))
         print(f"Plotting P(s) for b closest to estimate: {target_b_actual:.3f}")
    else:
         print("Error: No b values found in processed P(s) data.")
         return

    target_type = 'B' # Plot Type B clusters (adjust if needed)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    Ls = sorted(list(set(k[0] for k in processed_ps_data.keys())))
    base_markersize = mpl.rcParams['lines.markersize']
    plotted_data = False

    for i, L in enumerate(Ls):
        key = (L, target_b_actual, target_type)
        if key in processed_ps_data:
            s_values, ps_values = processed_ps_data[key]
            # Ensure data is valid for log-log plot
            valid_indices = (s_values > 0) & (ps_values > 0) & pd.notna(s_values) & pd.notna(ps_values)
            if np.any(valid_indices):
                 s_plot = s_values[valid_indices]
                 ps_plot = ps_values[valid_indices]
                 style_kwargs = get_style_kwargs(i, len(Ls), base_markersize)
                 # Use markers only for P(s)
                 ax.plot(s_plot, ps_plot,
                         marker=style_kwargs['marker'],
                         color=style_kwargs['color'],
                         linestyle='', # No line
                         markersize=style_kwargs['markersize'], # Use base size
                         label=f'$L={L}$')
                 plotted_data = True

    if not plotted_data:
         print(f"Error: No valid P(s) data found to plot for b={target_b_actual:.3f}, Type={target_type}.")
         plt.close(fig)
         return

    # --- Optional: Power Law Fit (Example on largest L data) ---
    largest_L = Ls[-1]
    key_large = (largest_L, target_b_actual, target_type)
    if key_large in processed_ps_data:
         s_large, ps_large = processed_ps_data[key_large]
         valid_large = (s_large > 0) & (ps_large > 0) & pd.notna(s_large) & pd.notna(ps_large)
         if np.any(valid_large):
              s_fit_all = s_large[valid_large]
              ps_fit_all = ps_large[valid_large]
              # Select a fitting range (heuristic, needs tuning!)
              # Avoid very small and very large clusters (finite size effects)
              fit_mask = (s_fit_all > 3) & (s_fit_all < (largest_L * largest_L / 20))
              if np.sum(fit_mask) > 2: # Need at least 3 points for fit
                  s_to_fit = s_fit_all[fit_mask]
                  ps_to_fit = ps_fit_all[fit_mask]
                  try:
                      log_s = np.log10(s_to_fit)
                      log_ps = np.log10(ps_to_fit)
                      # Fit: log(P) = log(C) - tau * log(s)
                      coeffs, cov = np.polyfit(log_s, log_ps, 1, cov=True)
                      tau_fit = -coeffs[0]
                      log_C_fit = coeffs[1]
                      # Estimate error (simple example)
                      tau_err = np.sqrt(np.diag(cov)[0]) if cov is not None else np.nan

                      # Plot the fitted line only in the fitted range
                      s_line = np.logspace(np.log10(s_to_fit.min()), np.log10(s_to_fit.max()), 50)
                      ps_line = (10**log_C_fit) * (s_line ** (-tau_fit))
                      ax.plot(s_line, ps_line, 'r--', linewidth=1.5,
                              label=f'Fit $(L={largest_L}): \\tau \\approx {tau_fit:.2f} \\pm {tau_err:.2f}$') # Include error if desired
                      print(f"Fitted P(s) exponent tau ~ {tau_fit:.3f} for L={largest_L}, b={target_b_actual:.3f}")
                  except Exception as e:
                      print(f"Could not perform/plot power-law fit for P(s): {e}")
              # else: print("Not enough points in fitting range for power-law fit.")

    ax.set_xlabel('Cluster Size ($s$)')
    ax.set_ylabel('Probability Density ($P(s)$)')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title(f'Cluster Size Distribution ($b={target_b_actual:.2f}$, Type {target_type})')
    ax.legend(title='System Size $L$', loc='best', frameon=False)
    ax.grid(True, which='both', linestyle=':', alpha=0.4) # Use grid for log-log

    save_plot(fig, f"{filename_base}_b{target_b_actual:.2f}_Type{target_type}")
    plt.close(fig)

# --- Figure 7: Boundary Effects ---
def plot_fig7_boundary_effects(data, filename_base):
    """ Plots boundary fraction and coop rates (boundary vs bulk) vs b. """
    print("Plotting Boundary Effects...")
    if data.empty:
        print("Error plotting Fig 7: Data is empty.")
        return

    # Define keys based on output of process_main_data
    boundary_frac_key = 'avg_BoundaryFraction'
    boundary_coop_key = 'avg_BoundaryCoopRate'
    bulk_coop_key = 'avg_BulkCoopRate'
    sem_boundary_frac = 'sem_BoundaryFraction'
    sem_boundary_coop = 'sem_BoundaryCoopRate'
    sem_bulk_coop = 'sem_BulkCoopRate'

    required_keys = [boundary_frac_key, boundary_coop_key, bulk_coop_key,
                     sem_boundary_frac, sem_boundary_coop, sem_bulk_coop]
    if not all(key in data.columns for key in required_keys):
        print(f"Error plotting Fig 7: Missing one or more required columns: {required_keys}")
        print(f"Available columns: {data.columns.tolist()}")
        return

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True)
    base_markersize = mpl.rcParams['lines.markersize']

    # --- Plot 1: Boundary Fraction ---
    ax1 = axes[0]
    Ls = sorted(data['L'].unique())
    for i, L in enumerate(Ls):
        L_data = data[data['L'] == L].sort_values('b')
        if L_data.empty: continue
        style_kwargs = get_style_kwargs(i, len(Ls), base_markersize)
        label = f'$L={L}$'
        yerr = L_data[sem_boundary_frac].replace([np.inf, -np.inf], np.nan).fillna(0)
        ax1.errorbar(L_data['b'], L_data[boundary_frac_key], yerr=yerr if np.any(yerr) else None,
                    label=label,  capsize=3, elinewidth=1, **style_kwargs)
#        ax1.errorbar(L_data['b'], L_data[boundary_frac_key], yerr=yerr if np.any(yerr) else None,
#                    fmt='-',  **style_kwargs)

    ax1.set_xlabel('Temptation ($b$)')
    ax1.set_ylabel('Boundary Fraction $\\langle f_{bound} \\rangle$')
    ax1.set_title('Fraction of Boundary Agents')
    ax1.legend(title='$L$', loc='best', frameon=False)
    ax1.set_ylim(bottom=-0.05)
    ax1.grid(False)

    # --- Plot 2: Boundary vs Bulk Cooperation ---
    ax2 = axes[1]
    # Plot for a representative L (e.g., largest)
    if not Ls: return # No L values found
    L_plot = Ls[-1]
    L_data_plot = data[data['L'] == L_plot].sort_values('b')

    if not L_data_plot.empty:
         # Boundary Coop
         yerr_bnd = L_data_plot[sem_boundary_coop].replace([np.inf, -np.inf], np.nan).fillna(0)
         ax2.errorbar(L_data_plot['b'], L_data_plot[boundary_coop_key],
                      yerr=yerr_bnd if np.any(yerr_bnd) else None,
                      label=f'Boundary Coop.',
                      fmt='-', capsize=3, elinewidth=1, color='red', marker='o', markersize=base_markersize)
         # Bulk Coop
         yerr_blk = L_data_plot[sem_bulk_coop].replace([np.inf, -np.inf], np.nan).fillna(0)
         ax2.errorbar(L_data_plot['b'], L_data_plot[bulk_coop_key],
                      yerr=yerr_blk if np.any(yerr_blk) else None,
                      label=f'Bulk Coop.',
                      fmt='-', capsize=3, elinewidth=1, color='blue', marker='s', markersize=base_markersize)
    else:
         print(f"Warning: No data found for L={L_plot} for boundary/bulk coop plot.")

    ax2.set_xlabel('Temptation ($b$)')
    ax2.set_ylabel('Avg. Cooperation Rate $\\langle f_C \\rangle$')
    ax2.set_title(f'Boundary vs. Bulk Cooperation ($L={L_plot}$)')
    ax2.legend(loc='best', frameon=False)
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(False)

    plt.tight_layout()
    save_plot(fig, filename_base)
    plt.close(fig)

# --- Figure 8: Phase Diagram ---
def plot_fig8_phase_diagram(data, filename_base):
    """ Plots a 2D phase diagram (heatmap). """
    print("Plotting Phase Diagram...")
    if data.empty:
        print("Error plotting Fig 8: Phase diagram data is empty.")
        return

    p1_name = PARAMS['phasediagram_param1_name']
    p2_name = PARAMS['phasediagram_param2_name']
    target_reporter_avg = PARAMS['phasediagram_target_reporter'] # e.g., 'avg_CooperationRate'

    if not all(k in data.columns for k in [p1_name, p2_name, target_reporter_avg]):
        print(f"Error plotting Fig 8: Missing required columns. Need: {p1_name}, {p2_name}, {target_reporter_avg}")
        print(f"Available cols: {data.columns.tolist()}")
        return

    # Average results across runs if multiple runs per point were done
    try:
        # Group by the parameters defining a point and calculate the mean of the target reporter
        grouped_pd = data.groupby([p1_name, p2_name])[target_reporter_avg].mean().reset_index()
        heatmap_data = grouped_pd.pivot(index=p2_name, columns=p1_name, values=target_reporter_avg)
    except Exception as e:
        print(f"Error pivoting data for heatmap: {e}")
        # Fallback if averaging fails (e.g., only one run per point)
        try:
             print("Attempting pivot without explicit averaging...")
             heatmap_data = data.pivot(index=p2_name, columns=p1_name, values=target_reporter_avg)
        except Exception as e2:
              print(f"Could not pivot data: {e2}")
              return

    fig, ax = plt.subplots(figsize=(7, 5.5))
    sns.heatmap(heatmap_data, ax=ax, cmap="viridis", # Use perceptually uniform colormap
                annot=False, # Annotate values only if grid is small
                # fmt=".2f", # Format for annotation
                cbar_kws={'label': target_reporter_avg.replace('avg_', '$\\langle$') + '$\\rangle$'}) # LaTeX label

    # Improve axis labels
    ax.set_xlabel(f'{p1_name}' if p1_name != 'b' else 'Temptation ($b$)')
    ax.set_ylabel(f'{p2_name}' if p2_name != 'K_C' else 'Cultural Noise ($K_C$)')
    ax.set_title(f'Phase Diagram ($L={PARAMS["phasediagram_L"]}$)')

    # Adjust tick labels if needed
    # plt.xticks(rotation=45, ha='right')
    # plt.yticks(rotation=0)

    # Ensure correct orientation (heatmap index often becomes y-axis)
    if heatmap_data.index[0] > heatmap_data.index[-1]: # If index decreases downwards
        ax.invert_yaxis()

    plt.tight_layout()
    save_plot(fig, filename_base)
    plt.close(fig)

# --- Snapshot Plotting (Adapted from original) ---
# plot_figures.py -> run_and_save_snapshot function

def run_and_save_snapshot(b_value, L_snap, filename_tag="snap"):
    """ Runs a single simulation and saves the final grid state. """
    snapshot_filename = os.path.join(SNAPSHOT_DATA_DIR, f"snapshot_{filename_tag}_L{L_snap}_b{b_value:.2f}.pkl")

    # Check if snapshot exists (logic remains the same)
    if os.path.exists(snapshot_filename):
        print(f"Skipping run, snapshot exists: {snapshot_filename}")
        return snapshot_filename

    print(f"Running simulation for snapshot (L={L_snap}, b={b_value:.2f})...")
    # Build parameters dictionary EXCLUDING 'steps' for the constructor
    model_params = {k: v for k, v in PARAMS.items() if k not in ['L_values', 'b_values', 'runs_per_parameter_set', 'seed', 'param_set_id', 'run_id', 'steps', 'steady_state_window'] and not k.startswith('snapshot_') and not k.startswith('cluster_') and not k.startswith('phasediagram_') and not k.startswith('fss_')}
    # Add the specific L, b, and seed for this snapshot run
    model_params['L'] = L_snap
    model_params['b'] = b_value
    model_params['seed'] = int(time.time() * 1000 + b_value * 100 + L_snap) % (2**32 - 1)

    # --- REMOVE OR COMMENT OUT THIS LINE ---
    # model_params['steps'] = PARAMS['steps']
    # ---------------------------------------

    # Get the number of steps needed for the run from PARAMS
    steps_to_run = PARAMS['steps']

    try:
        # Initialize the model with only the valid parameters
        model = CulturalGame(**model_params)

        # Run the model silently using the correct number of steps
        for _ in range(steps_to_run): # Use the variable defined above
            model.step()

    except Exception as e:
        print(f"Error running snapshot simulation (L={L_snap}, b={b_value}): {e}")
        import traceback # Add traceback for better debugging
        traceback.print_exc()
        return None

    # --- Grid state extraction logic remains the same ---
    grid_state = np.zeros((model.grid.width, model.grid.height, 3)) # x, y, [strategy, C, type_id]
    agent_types_map = {} # Map type_id to description
    TYPE_A_ID = 1
    TYPE_B_ID = 2
    threshold = 0.5 # Assuming fixed threshold for snapshot visualization

    if TYPE_A_ID not in agent_types_map: agent_types_map[TYPE_A_ID] = f"Type A (C<{threshold})"
    if TYPE_B_ID not in agent_types_map: agent_types_map[TYPE_B_ID] = f"Type B (C>={threshold})"

    # Ensure _get_agent_type is defined (you might already have this logic)



    for agent in model.schedule.agents:
        x, y = agent.pos
        if x is None or y is None: continue
        strategy = agent.strategy # 0 or 1
        culture = agent.C         # 0 to 1
        # Call _get_agent_type correctly
        agent_type = _get_agent_type(agent, threshold=threshold) # Pass threshold explicitly
        if agent_type == 'A':
             agent_type_id = TYPE_A_ID
        elif agent_type == 'B':
             agent_type_id = TYPE_B_ID
        else: # Handle None case if necessary
             agent_type_id = 0 # Or some other indicator

        grid_state[x, y, 0] = strategy
        grid_state[x, y, 1] = culture
        grid_state[x, y, 2] = agent_type_id

    # Also save the actual steps run in the snapshot data
    snapshot_params_saved = model_params.copy()
    snapshot_params_saved['steps_run'] = steps_to_run

    snapshot_data = {'grid': grid_state, 'params': snapshot_params_saved, 'types': agent_types_map}
    # --- Saving logic remains the same ---
    try:
        with open(snapshot_filename, 'wb') as f:
            pickle.dump(snapshot_data, f)
        print(f"Snapshot data saved to {snapshot_filename}")
        return snapshot_filename
    except Exception as e:
        print(f"Error saving snapshot data to {snapshot_filename}: {e}")
        return None




def plot_snapshot(snapshot_data_file, filename_base_prefix="snapshot"):
    """ Plots Spatial Snapshot with Physica A style. """
    if not snapshot_data_file or not os.path.exists(snapshot_data_file):
         print(f"Snapshot file not found or invalid: {snapshot_data_file}")
         return

    print(f"Plotting Snapshot from {os.path.basename(snapshot_data_file)}...")
    try:
        with open(snapshot_data_file, 'rb') as f:
            snapshot_data = pickle.load(f)
        grid = snapshot_data['grid']
        params = snapshot_data['params']
        type_desc = snapshot_data['types'] # {1: 'Type A...', 2: 'Type B...'}
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
        'default': np.array([0.5, 0.5, 0.5]) # Grey for unknown
    }
    legend_labels = {
        (TYPE_A_ID, STRAT_D): f"{type_desc.get(TYPE_A_ID, 'Type A')} / Defect",
        (TYPE_A_ID, STRAT_C): f"{type_desc.get(TYPE_A_ID, 'Type A')} / Cooperate",
        (TYPE_B_ID, STRAT_D): f"{type_desc.get(TYPE_B_ID, 'Type B')} / Defect",
        (TYPE_B_ID, STRAT_C): f"{type_desc.get(TYPE_B_ID, 'Type B')} / Cooperate",
    }

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
    # Create custom legend
    # 1. Filter out 'default' and keys not in legend_labels *before* sorting
    valid_legend_keys = [key for key in color_map.keys() if key != 'default' and key in legend_labels]
    # 2. Sort only the valid tuple keys
    sorted_legend_keys = sorted(valid_legend_keys)
    # 3. Create legend elements using the sorted keys
    legend_elements = [Patch(facecolor=color_map[key], edgecolor='k', linewidth=0.5, label=legend_labels[key])
                       for key in sorted_legend_keys]

    ax.legend(handles=legend_elements, bbox_to_anchor=(1.02, 1), loc='upper left', title="Agent State", fontsize='small')

    ax.set_title(f'Spatial Snapshot ($L={L}, b={b_val:.2f}$)')
    ax.set_xticks([])
    ax.set_yticks([])

    plt.tight_layout(rect=[0, 0, 0.75, 1]) # Adjust right margin for legend
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
    FORCE_RERUN_MAIN = False         # Rerun L-b scan for Figs 1,2,FSS,7?
    FORCE_RERUN_PHASEDIAGRAM = False # Rerun phase diagram scan?
    FORCE_RERUN_SNAPSHOTS = False    # Regenerate snapshot simulation data?

    # --- 1. Run/Load Main L vs b Scan ---
    main_raw_data = run_simulation_batch(generate_main_scan_configs, MAIN_DATA_FILENAME, force_rerun=FORCE_RERUN_MAIN)

    # --- 2. Process Main Scan Data (Includes Cluster Aggregation) ---
    main_processed_data, processed_ps_data = process_main_data(main_raw_data)

    # --- 3. Run/Load Phase Diagram Scan ---
    phasediagram_raw_data = run_simulation_batch(generate_phasediagram_configs, PHASEDIAGRAM_DATA_FILENAME, force_rerun=FORCE_RERUN_PHASEDIAGRAM)

    # --- 4. Generate/Check Snapshot Data ---
    snapshot_files = []
    print("\n--- Generating/Checking Snapshots ---")
    snap_L = PARAMS['snapshot_L']
    for b_snap in PARAMS['snapshot_b_values']:
         snap_filename = os.path.join(SNAPSHOT_DATA_DIR, f"snapshot_hetero_L{snap_L}_b{b_snap:.2f}.pkl")
         if FORCE_RERUN_SNAPSHOTS and os.path.exists(snap_filename):
              try: os.remove(snap_filename)
              except OSError as e: print(f"Error removing old snapshot {snap_filename}: {e}")

         # Pass only necessary info if needed, or let function use PARAMS
         fname = run_and_save_snapshot(b_snap, snap_L, "hetero")
         if fname:
             snapshot_files.append(fname)


    # --- 5. Generate Plots ---
    print("\n--- Generating Plots ---")
    if not main_processed_data.empty:
        # --- Fig 1: Order Parameters ---
        plot_fig1_order_param(main_processed_data, 'avg_CooperationRate', 'sem_CooperationRate',
                              'Avg. Cooperation Rate $\\langle f_C \\rangle$', "fig1a_coop_rate_vs_b")
        plot_fig1_order_param(main_processed_data, 'avg_SegregationIndex', 'sem_SegregationIndex',
                              'Avg. Segregation Index $\\langle S \\rangle$', "fig1b_segregation_vs_b")

        # --- Fig 2: Susceptibilities ---
        peaks_fc = plot_fig2_susceptibility(main_processed_data, 'chi_CooperationRate',
                                           'Susceptibility $\\chi_{f_C}$', "fig2a_susc_coop_rate")
        peaks_s = plot_fig2_susceptibility(main_processed_data, 'chi_SegregationIndex',
                                  r'Susceptibility $\chi_S$', "fig2b_susc_segregation")

        # --- Figs 3, 4, 5: FSS Analysis ---
        print("\n--- Performing FSS Analysis ---")
        # Use estimated exponents from PARAMS. Refine b_c based on peaks if desired.
        # Example: Use mean peak location from largest L susceptibility if peaks are clear
        bc_refined_fc = PARAMS['fss_bc_estimate'] # Start with param
        if not peaks_fc.empty and pd.notna(peaks_fc.iloc[-1]):
             # bc_refined_fc = peaks_fc.iloc[-1] # Use peak from largest L
             # Or average over larger Ls: bc_refined_fc = peaks_fc[peaks_fc.index >= sorted(peaks_fc.index)[-2]].mean()
             print(f"Using b_c estimate: {bc_refined_fc:.4f} (from PARAMS)")
        else:
             print(f"Warning: Could not refine b_c from susceptibility peaks. Using estimate: {bc_refined_fc:.4f}")

        # Call FSS plot function (corrected call)
        plot_fss_analysis(main_processed_data,
                            op_col_avg='avg_CooperationRate', # Avg value column
                            sus_col='chi_CooperationRate',    # Chi value column
                            b_param_col='b',                  # Name of b column in df
                            L_param_col='L',                  # Name of L column in df
                            b_c=bc_refined_fc,                # Critical point estimate
                            beta=PARAMS['fss_beta_estimate'], # Actual beta estimate
                            gamma=PARAMS['fss_gamma_estimate'],# Actual gamma estimate
                            nu=PARAMS['fss_nu_estimate'],     # Actual nu estimate
                            filename_base_prefix="fss_cooperation", # Prefix for output files
                            op_label=r"$\langle f_C \rangle$", # Specific label
                            sus_label=r"$\chi_{f_C}$",         # Specific label
                            b_label=r"$b$")                   # Specific label

        # Optional: FSS for Segregation Index (if it shows criticality)
        # Find peak for S, e.g., bc_refined_s = peaks_s.mean() or PARAMS estimate
        # plot_fss_analysis(main_processed_data, 'avg_SegregationIndex', 'chi_SegregationIndex', ...)


        # --- Fig 7: Boundary Effects ---
        plot_fig7_boundary_effects(main_processed_data, "fig7_boundary_effects")

    else:
        print("Skipping plots based on main scan data (Figs 1, 2, FSS, 7) as data is empty.")


    # --- Fig 6: Cluster Distribution ---
    if processed_ps_data:
        plot_fig6_cluster_dist(processed_ps_data, "fig6_cluster_dist")
    else:
        print("Skipping Fig 6 (Cluster Distribution) due to missing/empty processed P(s) data.")


    # --- Fig 8: Phase Diagram ---
    if not phasediagram_raw_data.empty:
        plot_fig8_phase_diagram(phasediagram_raw_data, "fig8_phase_diagram")
    else:
        print("Skipping Fig 8 (Phase Diagram) due to missing/empty data.")


    # --- Plot Snapshots ---
    print("\n--- Plotting Snapshots ---")
    if not snapshot_files:
        print("No snapshot files found or generated to plot.")
    else:
        for snap_file in snapshot_files:
             plot_snapshot(snap_file) # Uses default filename prefix


    overall_end_time = time.time()
    print(f"\n--- Workflow finished in {(overall_end_time - overall_start_time):.2f} seconds ---")
    print(f"Data saved in: '{DATA_SAVE_DIR}'")
    print(f"Plots saved in: '{PLOT_SAVE_DIR}'")
