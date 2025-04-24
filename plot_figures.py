# plot_figures.py
import numpy as np  # 导入 numpy 用于处理 geomspace 数据
from matplotlib.ticker import FormatStrFormatter  # 导入 FormatStrFormatter
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
from matplotlib import cm  # Explicitly import cm
from scipy.optimize import curve_fit  # For fitting P(s) if needed
from collections import Counter  # For processing cluster sizes
from dataclasses import fields  # Import fields to introspect SimConfig
from typing import Dict, Any  # Add this line


# --- Import Core Simulation Logic ---
RUN_WITH_PARALLEL = False  # Default to False, check for parallel module
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
        from utils.config import SimConfig  # Still need SimConfig for sequential mode
    except ImportError as e:
        print(f"Error importing core modules for sequential fallback: {e}")
        exit()

# --- Ensure Reporters are Accessible (Needed by snapshot logic) ---
try:
    from utils.reporters import _get_agent_type
except ImportError:
    def _get_agent_type(agent, threshold=0.5):
        if not hasattr(agent, 'C'):
            return None
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
    'p_update_C': 0.1,  # Strategy update probability (模仿更新文化的概率)

    # --- Mutation Parameters (New Names) ---
    'p_mut_culture': 0.01,      # Probability of random cultural mutation per step
    'p_mut_strategy': 0.001,    # Probability of random strategy mutation per step

    # Initial distribution ('uniform', 'normal', 'bimodal', 'fixed')
    'C_dist': 'bimodal',
    'mu': 0.5,          # Meaning depends on C_dist (e.g., p(C=1) for bimodal)
    'sigma': 0.1,       # Std Dev for C_dist='normal'

    # --- Simulation Control ---
    'steps': 2000,        # Adjust steps for steady state  建议30000
    # Average over last N steps (for avg reporters)
    'steady_state_window': 500,             # 建议1000
    'runs_per_parameter_set': 10,  # CRUCIAL FOR STATS & CHI

    # --- Scan Parameters ---
    'L_values': [20, 30, 40, 50],  # System sizes for FSS (adjust as needed)
    #    'b_values': np.linspace(1.3, 7.3, 13), # Temptation 'b' (refine near transition!)
    'b_values': np.unique(np.concatenate((
        np.linspace(1.3, 1.9, 4),      # 低 b 区: 1.3, 1.5, 1.7, 1.9 (步长~0.2)
        # 临界区: 2.0, 2.05, 2.1, ..., 2.95, 3.0 (步长 0.05)
        np.linspace(2.0, 3.0, 21),
        np.linspace(3.2, 4.0, 5),      # 过渡区: 3.2, 3.4, 3.6, 3.8, 4.0 (步长 0.2)
        # 高 b 区: 4.5, 5.0, 5.5, 6.0, 6.5, 7.0 (步长 0.5)
        np.linspace(4.5, 7.0, 6)
        # np.linspace(3.5, 7.0, 8) # 另一种高 b 区选择 (步长 0.5)
    ))),
    # --- Snapshot Specific ---
    'snapshot_L': 50,  # L for snapshots (usually one of the larger ones)
    # Example b values
    'snapshot_b_values': [1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0],

    # --- Cluster Analysis Specific ---
    # b values near transition for P(s)
    'cluster_analysis_b_values': [2.2, 2.3, 2.4, 2.45, 2.5, 2.55, 2.6],

    # --- Phase Diagram Specific (Example: Scan b vs K_C) ---
    'phasediagram_L': 40,        # Fixed L for phase diagram
    'phasediagram_param1_name': 'b',
    'phasediagram_param1_values': np.linspace(1.3, 7.3, 13),
    'phasediagram_param2_name': 'K_C',
    'phasediagram_param2_values': np.geomspace(0.0001, 1.0, num=13),
    #    'phasediagram_param2_values': np.round(np.linspace(0.01, 0.91, 10), 2),
    'phasediagram_target_reporter': 'avg_CooperationRate',  # What to plot

    # --- FSS Parameters (REVISE after seeing peaks!) ---
    # Initial guess for critical b (Cooperation)
    'fss_bc_estimate': 2.5,
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
PHASEDIAGRAM_DATA_FILENAME = os.path.join(
    DATA_SAVE_DIR, "phasediagram_data.pkl")
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
    'lines.linewidth': 1.5, 'lines.markersize': 4,  # Base marker size
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
        'markersize': base_markersize  # Return markersize too
    }


def generate_main_scan_configs():
    """Generates SimConfig objects for the main L vs b scan."""
    configs = []
    # Base parameters excluding those being swept or set per run
    # Add all parameters from PARAMS as base, then exclude sweep/run-specific ones
    base_params = PARAMS.copy()
    # Exclude lists:
    sweep_params_main = ['L_values', 'b_values']
    run_specific_params = ['runs_per_parameter_set', 'steps',
                           'seed', 'param_set_id', 'run_id', 'steady_state_window']
    plot_specific_params = [k for k in PARAMS if k.startswith('snapshot_') or k.startswith(
        'cluster_') or k.startswith('phasediagram_') or k.startswith('fss_')]

    # Filter base_params
    for key_list in [sweep_params_main, run_specific_params, plot_specific_params]:
        for key in key_list:
            if key in base_params:
                del base_params[key]

    # Add steps back as it's part of SimConfig
    base_params['steps'] = PARAMS['steps']

    # Ensure all fields required by SimConfig are present in base_params
    # If not, default values from SimConfig will be used
    sim_config_fields = {f.name for f in fields(SimConfig)}
    valid_base_params = {k: v for k,
                         v in base_params.items() if k in sim_config_fields}

    param_list = []
    for L in PARAMS['L_values']:
        for b in PARAMS['b_values']:
            current_params = valid_base_params.copy()  # Start with filtered base
            current_params['L'] = L
            current_params['b'] = b
            # Generate a unique ID for this parameter set
            # Include other fixed parameters in ID if you want to distinguish data files/plot titles later
            id_parts = [f"L{L}", f"b{b:.3f}".replace('.', 'p')]
            # Add fixed mutation rates to ID for clarity in data files/plotting
            if 'p_mut_culture' in PARAMS:
                id_parts.append(
                    f"pmC{PARAMS['p_mut_culture']:.3f}".replace('.', 'p'))
            if 'p_mut_strategy' in PARAMS:
                id_parts.append(
                    f"pmS{PARAMS['p_mut_strategy']:.3f}".replace('.', 'p'))

            current_params['param_set_id'] = "_".join(id_parts)
            param_list.append(current_params)

    # Create SimConfig objects for each parameter set, repeated for each run
    for i in range(PARAMS['runs_per_parameter_set']):
        run_seed_base = i * 10000  # Base seed for this run index
        for idx, params in enumerate(param_list):
            run_params = params.copy()
            run_params['seed'] = run_seed_base + idx  # Unique seed per run
            run_params['run_id'] = i  # Store the run index
            try:
                configs.append(SimConfig(**run_params))
            except TypeError as e:
                print(
                    f"Error creating SimConfig. Params: {run_params}\nError: {e}")
                # You might want to investigate why this fails if it happens
                # print(f"SimConfig fields: {sim_config_fields}")

    print(f"Generated {len(configs)} SimConfig objects for main scan.")
    return configs


def generate_phasediagram_configs():
    """ Generates configs for phase diagram scan """
    configs = []
    # Base parameters excluding those being swept or set per run
    base_params = PARAMS.copy()
    # Exclude lists:
    sweep_params_pd = [PARAMS['phasediagram_param1_name'],
                       PARAMS['phasediagram_param2_name']]
    run_specific_params = ['runs_per_parameter_set', 'steps',
                           'seed', 'param_set_id', 'run_id', 'steady_state_window']
    plot_specific_params = [k for k in PARAMS if k.startswith('snapshot_') or k.startswith(
        'cluster_') or k.startswith('phasediagram_') or k.startswith('fss_')]

    # Filter base_params
    for key_list in [sweep_params_pd, run_specific_params, plot_specific_params]:
        for key in key_list:
            if key in base_params:
                del base_params[key]

    base_params['steps'] = PARAMS['steps']
    base_params['L'] = PARAMS['phasediagram_L']  # Fixed L for phase diagram

    # Ensure all fields required by SimConfig are present in base_params
    sim_config_fields = {f.name for f in fields(SimConfig)}
    valid_base_params = {k: v for k,
                         v in base_params.items() if k in sim_config_fields}

    param_list = []
    p1_name = PARAMS['phasediagram_param1_name']
    p2_name = PARAMS['phasediagram_param2_name']
    p1_values = PARAMS['phasediagram_param1_values']
    p2_values = PARAMS['phasediagram_param2_values']

    for p1_val in p1_values:
        for p2_val in p2_values:
            current_params = valid_base_params.copy()
            current_params[p1_name] = p1_val
            current_params[p2_name] = p2_val
            # Generate a unique ID
            id_parts = [f"PhaseD_L{current_params['L']}"]
            # Use .4g for better float representation
            id_parts.append(f"{p1_name}{p1_val:.4g}".replace('.', 'p'))
            id_parts.append(f"{p2_name}{p2_val:.4g}".replace('.', 'p'))
            # Add fixed mutation rates to ID if they are not being scanned
            if p1_name not in ['p_mut_culture', 'p_mut_strategy'] and p2_name not in ['p_mut_culture', 'p_mut_strategy']:
                if 'p_mut_culture' in PARAMS:
                    id_parts.append(
                        f"pmC{PARAMS['p_mut_culture']:.3f}".replace('.', 'p'))
                if 'p_mut_strategy' in PARAMS:
                    id_parts.append(
                        f"pmS{PARAMS['p_mut_strategy']:.3f}".replace('.', 'p'))

            current_params['param_set_id'] = "_".join(id_parts)
            param_list.append(current_params)

    # Fewer runs might be okay for phase diagram, adjust if needed
    num_runs = max(1, PARAMS['runs_per_parameter_set'] // 2)

    for i in range(num_runs):
        run_seed_base = i * 10000 + 7000  # Offset seed
        for idx, params in enumerate(param_list):
            run_params = params.copy()
            run_params['seed'] = run_seed_base + idx
            run_params['run_id'] = i
            configs.append(SimConfig(**run_params))

    print(
        f"Generated {len(configs)} SimConfig objects for phase diagram (runs per point: {num_runs}).")
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
            print(
                f"Error loading data from {data_filename}: {e}. Rerunning...")
            force_rerun = True  # Force rerun if loading fails

    print(f"Running simulations for {data_filename}...")
    sim_configs = config_generator()
    if not sim_configs:
        print("Warning: No configurations generated.")
        return pd.DataFrame()

    if RUN_WITH_PARALLEL:
        # batch_run_parallel takes SimConfig objects, which contain all needed parameters
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
                    # CulturalGame(**config.to_dict()) correctly passes all parameters
                    model = CulturalGame(**config.to_dict())
                    for _ in range(config.steps):
                        model.step()

                    model_df = model.datacollector.get_model_vars_dataframe()
                    result_dict = config.to_dict()  # Ensure this dict has correct keys
                    n_rows = len(model_df)

                    if n_rows > 0:
                        start_idx = max(
                            0, n_rows - PARAMS['steady_state_window'])
                        window_df = model_df.iloc[start_idx:]

                        for col in model.datacollector.model_reporters.keys():
                            if col in model_df.columns:
                                if col in NON_AVERAGE_REPORTERS:
                                    # Store the last value directly
                                    result_dict[f"{col}"] = model_df[col].iloc[-1]
                                elif not window_df.empty:
                                    # Calculate mean and std for averaged reporters
                                    result_dict[f"avg_{col}"] = window_df[col].mean(
                                    )
                                    result_dict[f"std_{col}"] = window_df[col].std(
                                    )
                                else:  # Window is empty, but n_rows > 0 implies step > 0
                                    # This case is unlikely with steady_state_window >= 1
                                    result_dict[f"avg_{col}"] = np.nan
                                    result_dict[f"std_{col}"] = np.nan
                                    if col in NON_AVERAGE_REPORTERS:
                                        # Should be handled by first if
                                        result_dict[f"{col}"] = None
                            else:  # Reporter column missing
                                print(
                                    f"Warning: Column '{col}' not found in model_df for config {config.param_set_id}, run {config.run_id}. Setting to NaN.")
                                result_dict[f"avg_{col}"] = np.nan
                                result_dict[f"std_{col}"] = np.nan
                                if col in NON_AVERAGE_REPORTERS:
                                    result_dict[f"{col}"] = None

                        else:  # No data collected at all
                            print(
                                f"Warning: No data collected for config {config.param_set_id}, run {config.run_id}.")
                            for col in model.datacollector.model_reporters.keys():
                                # Fix: changed results_dict to result_dict
                                result_dict[f"avg_{col}"] = np.nan
                                result_dict[f"std_{col}"] = np.nan
                                if col in NON_AVERAGE_REPORTERS:
                                    result_dict[f"{col}"] = None

                    all_results.append(result_dict)

                except Exception as e:
                    print(
                        f"\nError running config {config.param_set_id} (run {config.run_id}): {e}")
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
    Groups by all simulation parameters except run-specific ones (seed, run_id).
    """
    if raw_df.empty:
        print("Warning: process_main_data received empty DataFrame.")
        return pd.DataFrame(), None  # Return empty DF and None for clusters

    processed_data = []
    # Aggregated cluster sizes: Key: grouping_tuple, Value: {'A': all_sizes, 'B': all_sizes}
    cluster_results = {}

    # Group by all columns that are NOT run-specific identifiers OR reporter results
    sim_config_fields = {f.name for f in fields(SimConfig)}
    # Identify columns that are part of SimConfig (parameters)
    parameter_cols = [
        col for col in raw_df.columns if col in sim_config_fields]
    # Exclude run-specific identifiers
    run_identifiers = ['seed', 'run_id', 'param_set_id']
    # Exclude columns that are reporter results (avg_, std_, or specific non-avg reporters)
    reporter_cols = [col for col in raw_df.columns if col.startswith('avg_') or col.startswith(
        # Add other specific non-avg reporters here if any
        'std_') or col == 'ClusterSizeDistribution']

    # The grouping columns are parameters that are NOT run identifiers and NOT reporter outputs
    grouping_params = [
        col for col in parameter_cols if col not in run_identifiers and col not in reporter_cols]

    # Add L and b explicitly to ensure they are always in grouping_params for the main scan context
    if 'L' not in grouping_params and 'L' in raw_df.columns:
        grouping_params.append('L')
    if 'b' not in grouping_params and 'b' in raw_df.columns:
        grouping_params.append('b')

    # Ensure uniqueness and sorted order for consistency
    grouping_params = sorted(list(set(grouping_params)))

    print(f"Processing main scan data, grouping by: {grouping_params}")

    # Reporters for mean/SEM/Susceptibility calculation
    # Identify reporters that have 'avg_' prefix (meaning they were averaged by default by run_single_simulation)
    avg_reporters = [col for col in raw_df.columns if col.startswith('avg_')]
    # e.g., 'CooperationRate'
    avg_reporter_names = [col.split('avg_')[1] for col in avg_reporters]

    # Reporters for which we want susceptibility (typically order parameters)
    susceptibility_target_names = [
        'CooperationRate', 'SegregationIndex']  # Base names

    # Special handling for non-averaged reporters
    # Exact name from parallel/sequential run output
    cluster_col_name = "ClusterSizeDistribution"

    grouped = raw_df.groupby(grouping_params)

    print(f"Processing {len(grouped)} unique parameter sets...")
    for name, group in tqdm(grouped, desc="Processing parameter sets"):
        # 'name' is a tuple corresponding to the grouping_params values
        processed_point = dict(zip(grouping_params, name)
                               )  # Store grouping params

        L = processed_point.get('L')  # Get L value from the grouping tuple
        if pd.notna(L) and L is not None:
            N = L * L  # Number of agents
        else:
            N = np.nan  # Cannot calculate N

        # --- Calculate Mean and SEM for averaged reporters ---
        for reporter_name in avg_reporter_names:
            avg_col = f"avg_{reporter_name}"
            if avg_col in group.columns:
                mean_val = group[avg_col].mean()
                # Standard Error of the Mean across runs
                sem_val = group[avg_col].sem()
                processed_point[avg_col] = mean_val
                processed_point[f"sem_{reporter_name}"] = sem_val
            else:  # Should not happen if data generation is correct but safety
                processed_point[avg_col] = np.nan
                processed_point[f"sem_{reporter_name}"] = np.nan

        # --- Calculate Susceptibility (Chi) = N * Var(<Reporter>) ---
        for reporter_name in susceptibility_target_names:
            # Column containing the mean value *for each run*
            avg_col = f"avg_{reporter_name}"
            if avg_col in group.columns and pd.notna(N) and N > 0:
                # Variance of the run averages
                variance_across_runs = group[avg_col].var()
                # Susceptibility
                chi_val = N * \
                    variance_across_runs if pd.notna(
                        variance_across_runs) else np.nan
                processed_point[f"chi_{reporter_name}"] = chi_val
            else:
                processed_point[f"chi_{reporter_name}"] = np.nan

        # --- Aggregate Cluster Data ---
        if cluster_col_name in group.columns:
            all_sizes_A = []
            all_sizes_B = []
            # Iterate through each run in the group
            for idx, row in group.iterrows():
                # Get the dict {'A':[], 'B':[]} from this run
                dist_dict = row[cluster_col_name]
                if isinstance(dist_dict, dict):
                    all_sizes_A.extend(dist_dict.get('A', []))
                    all_sizes_B.extend(dist_dict.get('B', []))
                # else: print(f"Warning: Unexpected data type in {cluster_col_name} at {name}, run {row.get('run_id', '?')}")

            # Store aggregated sizes for this parameter set (name tuple)
            if name not in cluster_results:
                cluster_results[name] = {'A': [], 'B': []}
            cluster_results[name]['A'].extend(all_sizes_A)
            cluster_results[name]['B'].extend(all_sizes_B)
        # else: print(f"Warning: Column '{cluster_col_name}' not found at {name}.")

        processed_data.append(processed_point)

    processed_df = pd.DataFrame(processed_data)

    # --- Calculate P(s) from aggregated cluster_results ---
    print("Calculating P(s) from aggregated cluster sizes...")
    # Key: (grouping_tuple, type_label), Value: (bin_centers, ps_values)
    processed_ps = {}

    # Use log binning parameters (adjust factor as needed)
    log_bin_factor = 1.4

    for (grouping_tuple), sizes_dict in tqdm(cluster_results.items(), desc="Calculating P(s)"):
        # Extract L and b from grouping_tuple based on grouping_params order
        L = grouping_tuple[grouping_params.index(
            'L')] if 'L' in grouping_params else None
        b = grouping_tuple[grouping_params.index(
            'b')] if 'b' in grouping_params else None
        if L is None or b is None:
            continue  # Need L and b for P(s) context

        for type_label, all_sizes in sizes_dict.items():
            if not all_sizes:
                continue

            counts = Counter(all_sizes)
            # Total clusters of this type for this set across all runs
            total_clusters = len(all_sizes)
            if total_clusters == 0:
                continue

            s_values = np.array(list(counts.keys()))
            # Raw counts for each size s
            raw_counts = np.array([counts[s] for s in s_values])

            # --- Log Binning ---
            if len(s_values) < 2:  # Not enough data for log binning
                if len(s_values) == 1 and total_clusters > 0:  # Single size found
                    processed_ps[(grouping_tuple, type_label)] = (
                        s_values, raw_counts / total_clusters)
                continue  # Skip if no valid data

            min_s, max_s = s_values.min(), s_values.max()
            if max_s <= min_s or min_s <= 0:
                continue  # Need positive range

            # Use np.logspace for more reliable log bin edges
            # Determine log scale range. Avoid log(0).
            # Start log binning at 1 or min_s
            log_min = np.log10(max(1, min_s))
            log_max = np.log10(max_s)
            if log_max <= log_min:
                continue  # Need a valid range

            # Estimate number of bins
            # This is a heuristic; adjust as needed
            # Ensure at least 5 bins if possible
            num_bins = max(
                5, int(np.ceil((log_max - log_min) / np.log10(log_bin_factor))))

            # Create bin edges on log scale, convert back, ensure uniqueness and integer type
            bins_float = np.logspace(log_min, log_max, num_bins + 1)
            # Ensure bins are integers and unique
            bins = np.unique(np.floor(bins_float)).astype(np.int64)
            bins = bins[bins >= 1]  # Ensure bins start at or above 1
            bins = np.unique(bins)  # Ensure unique after floor
            bins = np.sort(bins)  # Ensure sorted

            if len(bins) < 2:  # Re-check after cleanup
                # Maybe single bin covering everything or just the single point?
                if len(s_values) == 1 and total_clusters > 0 and bins.size >= 1:
                    processed_ps[(grouping_tuple, type_label)] = (
                        s_values, raw_counts / total_clusters)
                continue  # Skip if still not enough bins

            # Bin the data points (s_values) according to counts (raw_counts)
            # We need to sum counts within each bin
            # Use histogram function for binning counts
            counts_in_bins, bin_edges = np.histogram(
                s_values, bins=bins, weights=raw_counts)
            # bin_edges are the same as 'bins' array

            binned_s_centers = []
            binned_ps_values = []

            # Iterate through bins (number of bins = len(bins) - 1)
            for i in range(len(counts_in_bins)):
                total_count_in_bin = counts_in_bins[i]
                if total_count_in_bin == 0:
                    continue

                # Bin center calculation (geometric mean of bin edges recommended for log bins)
                bin_start = bin_edges[i]
                bin_end = bin_edges[i+1]
                # Avoid log(0)
                if bin_start > 0 and bin_end > 0:
                    bin_center = np.sqrt(bin_start * bin_end)  # Geometric mean
                elif bin_start == 0 and bin_end > 0:  # Should not happen with bins >= 1, but for safety
                    bin_center = bin_end / 2  # Linear center as fallback
                else:  # Invalid bin edges
                    continue

                # Bin width (linear scale)
                bin_width = bin_end - bin_start
                if bin_width <= 0:
                    continue  # Skip zero-width bins

                # Calculate P(s) density for the bin
                # P(s) = (total count in bin) / (total number of clusters * bin width)
                ps_value = total_count_in_bin / (total_clusters * bin_width)

                binned_s_centers.append(bin_center)
                binned_ps_values.append(ps_value)

            if binned_s_centers:
                processed_ps[(grouping_tuple, type_label)] = (
                    np.array(binned_s_centers), np.array(binned_ps_values))
            # --- End Log Binning ---

    print(
        f"Finished processing. Found {len(processed_df)} unique parameter sets.")
    print(f"Calculated P(s) for {len(processed_ps)} cases.")

    # Store grouping parameters list for later use in plotting functions
    processed_df.attrs['grouping_params'] = grouping_params
    # processed_ps doesn't have attributes, but we can store the list separately or pass it

    # Return both processed averages and P(s) data
    return processed_df, processed_ps


# Helper function to filter data based on fixed parameters
def filter_data_by_fixed_params(data, params_to_check):
    """
    Filters a DataFrame to keep only rows where columns match specific values.
    Used to select a subset of data corresponding to fixed parameters (like mutation rates).
    params_to_check: Dictionary {param_name: value}
    """
    filtered_data = data.copy()
    for param, value in params_to_check.items():
        if param in filtered_data.columns:
            # Use np.isclose for float comparison
            if pd.api.types.is_numeric_dtype(filtered_data[param]):
                # Handle potential NaN values carefully
                filtered_data = filtered_data[
                    # Match numeric
                    np.isclose(filtered_data[param], value, atol=1e-9, rtol=1e-9) |
                    # Match NaN if both are NaN
                    (pd.isna(filtered_data[param]) & pd.isna(value))
                ]
            else:  # For non-numeric (e.g., strings)
                filtered_data = filtered_data[filtered_data[param] == value]
        # If the param is not in columns, assume it was fixed at this value during data generation
        # and the data already only contains this value (based on process_main_data grouping)
        # If data contains multiple values for this param and it wasn't in grouping_params, this is a problem.
        # print(f"Warning: Filtering by '{param}' which is not in DataFrame columns.")
    return filtered_data


# Helper function to generate parameter string for plot titles/filenames
def get_fixed_params_string(data_row_or_params_dict, params_to_display=None):
    """
    Generates a string like '(p_mut_culture=0.01, p_mut_strategy=0.001)'
    from a data row (Series) or a dictionary of parameters.
    params_to_display: List of parameter names to include in the string.
                       Defaults to mutation rates if not specified.
    """
    if params_to_display is None:
        # Default to mutation rates
        params_to_display = ['p_mut_culture', 'p_mut_strategy']

    parts = []
    for param in params_to_display:
        if param in data_row_or_params_dict:
            value = data_row_or_params_dict[param]
            # Format based on type
            if isinstance(value, float):
                # Use .3g for general float fmt
                parts.append(
                    f"${{{param.replace('_', ',').replace('p,mut', 'p_{mut}')}}}={value:.3g}$")
            elif isinstance(value, int):
                parts.append(
                    f"${{{param.replace('_', ',').replace('p,mut', 'p_{mut}')}}}={value}$")
            else:  # Other types like string
                parts.append(f"{param}={value}")
        # If not found, skip or add a placeholder like param=?
    return "(" + ", ".join(parts) + ")" if parts else ""


# ==============================================================================
# Plotting Functions (Physica A Style)
# ==============================================================================

def save_plot(fig, base_filename, plot_dir=PLOT_SAVE_DIR):
    """Helper function to save plot."""
    png_path = os.path.join(plot_dir, f"{base_filename}.png")
    # Save PDF for quality
    pdf_path = os.path.join(plot_dir, f"{base_filename}.pdf")
    try:
        fig.savefig(png_path, dpi=300, bbox_inches='tight')
        fig.savefig(pdf_path, bbox_inches='tight')
        print(f"Saved plot: {png_path}")
    except Exception as e:
        print(f"Error saving plot {base_filename}: {e}")

# --- Figure 1: Order Parameter vs b (Multi-L) ---


def plot_fig1_order_param(data, order_param_key_avg, sem_key, ylabel, filename_base):
    """ Plots an order parameter vs b for different L values. """
    print(f"Plotting Order Parameter: {order_param_key_avg}...")
    if data.empty or order_param_key_avg not in data.columns:
        print(
            f"Error plotting {filename_base}: Data missing or '{order_param_key_avg}' column not found.")
        return

    # --- Filter data for fixed parameters (e.g., mutation rates) ---
    # Assuming you want to plot for the default mutation rates specified in PARAMS
    fixed_params_to_plot = {
        'p_mut_culture': PARAMS.get('p_mut_culture', None),
        'p_mut_strategy': PARAMS.get('p_mut_strategy', None)
        # Add other fixed parameters you want to filter by, e.g., 'K_C': PARAMS.get('K_C')
    }
    # Remove None values
    fixed_params_to_plot = {k: v for k,
                            v in fixed_params_to_plot.items() if v is not None}

    filtered_data = filter_data_by_fixed_params(data, fixed_params_to_plot)
    if filtered_data.empty:
        print(
            f"Warning: No data found for plotting {filename_base} with fixed parameters: {fixed_params_to_plot}")
        return

    # Get the string representation of fixed parameters for title/filename
    fixed_params_str = get_fixed_params_string(fixed_params_to_plot)
    # Add formatted string to filename
    filename_base_with_params = filename_base + "_" + fixed_params_str.replace(
        '(', '').replace(')', '').replace(', ', '_').replace('=', '').replace('.', 'p')

    fig, ax = plt.subplots(figsize=(6, 4))
    Ls = sorted(filtered_data['L'].unique())
    base_markersize = mpl.rcParams['lines.markersize']

    for i, L in enumerate(Ls):
        L_data = filtered_data[filtered_data['L'] == L].sort_values('b')
        if L_data.empty:
            continue  # Should not happen if L came from filtered_data but safety check
        style_kwargs = get_style_kwargs(i, len(Ls), base_markersize)
        label = f'$L={L}$'

        # Check if SEM data exists and is valid
        yerr = None
        if sem_key in L_data.columns:
            yerr_data = L_data[sem_key].replace(
                [np.inf, -np.inf], np.nan).fillna(0)  # Handle NaN/inf SEM
            if not np.all(yerr_data == 0):  # Only plot error bars if SEM is meaningful
                yerr = yerr_data

        ax.errorbar(L_data['b'], L_data[order_param_key_avg],
                    yerr=yerr,
                    label=label,
                    linestyle='-',  # Line connecting points
                    capsize=3, elinewidth=1,
                    color=style_kwargs['color'],
                    marker=style_kwargs['marker'],
                    markersize=style_kwargs['markersize']
                    )

    ax.set_xlabel('Temptation ($b$)')
    ax.set_ylabel(ylabel)
    # Extract name like f_C or S
    title_str = ylabel.split("$\\langle$")[-1].split("$\\rangle$")[0]
    ax.set_title(
        f'Order Parameter $\\langle {title_str} \\rangle$ vs. Temptation $b$\n{fixed_params_str}')
    ax.legend(title='System Size $L$', loc='best', frameon=False)
    # Sensible limits (adjust if needed)
    if "Rate" in ylabel or "Index" in ylabel:
        ax.set_ylim(-0.05, 1.05)
    ax.grid(False)  # Physica A often prefers no grid

    save_plot(fig, filename_base_with_params)
    plt.close(fig)


# --- Figure 2: Susceptibility vs b (Multi-L) ---
def plot_fig2_susceptibility(data, chi_key, ylabel, filename_base):
    """ Plots susceptibility vs b for different L values. """
    print(f"Plotting Susceptibility: {chi_key}...")
    if data.empty or chi_key not in data.columns:
        print(
            f"Error plotting {filename_base}: Data missing or '{chi_key}' column not found.")
        return pd.Series(dtype=float)  # Return empty Series if error

    # --- Filter data for fixed parameters ---
    fixed_params_to_plot = {
        'p_mut_culture': PARAMS.get('p_mut_culture', None),
        'p_mut_strategy': PARAMS.get('p_mut_strategy', None)
        # Add other fixed parameters you want to filter by
    }
    fixed_params_to_plot = {k: v for k,
                            v in fixed_params_to_plot.items() if v is not None}
    filtered_data = filter_data_by_fixed_params(data, fixed_params_to_plot)
    if filtered_data.empty:
        print(
            f"Warning: No data found for plotting {filename_base} with fixed parameters: {fixed_params_to_plot}")
        return pd.Series(dtype=float)

    # Get the string representation of fixed parameters for title/filename
    fixed_params_str = get_fixed_params_string(fixed_params_to_plot)
    filename_base_with_params = filename_base + "_" + fixed_params_str.replace(
        '(', '').replace(')', '').replace(', ', '_').replace('=', '').replace('.', 'p')

    fig, ax = plt.subplots(figsize=(6, 4))
    Ls = sorted(filtered_data['L'].unique())
    base_markersize = mpl.rcParams['lines.markersize']
    peak_locs = {}  # Store peak locations {L: b_peak}

    for i, L in enumerate(Ls):
        L_data = filtered_data[filtered_data['L'] == L].sort_values('b')
        L_data = L_data.replace(
            [np.inf, -np.inf], np.nan).dropna(subset=[chi_key])  # Clean data
        if L_data.empty:
            continue

        style_kwargs = get_style_kwargs(i, len(Ls), base_markersize)
        label = f'$L={L}$'
        ax.plot(L_data['b'], L_data[chi_key], label=label,
                **style_kwargs)  # Apply style

        # Find peak location
        peak_idx = None
        if not L_data.empty and chi_key in L_data.columns and not L_data[chi_key].empty:
            peak_idx = L_data[chi_key].idxmax()
            # Ensure index exists before accessing loc
            if peak_idx in L_data.index:
                peak_locs[L] = L_data.loc[peak_idx, 'b']
            else:
                peak_locs[L] = np.nan  # Indicate peak not found
        else:
            peak_locs[L] = np.nan

    ax.set_xlabel('Temptation ($b$)')
    ax.set_ylabel(ylabel)
    ax.set_title(f'Susceptibility vs. Temptation $b$\n{fixed_params_str}')
    ax.legend(title='System Size $L$', loc='best', frameon=False)
    # ax.set_yscale('log') # Consider log scale if peaks vary widely
    ax.grid(False)

    save_plot(fig, filename_base_with_params)
    plt.close(fig)
    print(f"Susceptibility Peak Locations ({chi_key}): {peak_locs}")
    # Return peaks only for L values that were actually plotted
    return pd.Series({L: peak_locs.get(L) for L in Ls if pd.notna(peak_locs.get(L))})


# --- Figures 3, 4, 5: Finite-Size Scaling ---
def plot_fss_analysis(
    data: pd.DataFrame,
    op_col_avg: str,  # e.g., 'avg_CooperationRate'
    sus_col: str,    # e.g., 'chi_CooperationRate'
    b_param_col: str,  # Name of column holding 'b' values, e.g., 'b'
    L_param_col: str,  # Name of column holding 'L' values, e.g., 'L'
    b_c: float,       # Critical point estimate
    beta: float,      # Critical exponent beta
    gamma: float,     # Critical exponent gamma
    nu: float,        # Critical exponent nu
    filename_base_prefix: str = "fss_analysis",
    output_dir: str = PLOT_SAVE_DIR,  # Use global plot dir
    op_label: str = r"$\langle O \rangle$",  # Generic labels, customize if needed
    sus_label: str = r"$\chi$",
    b_label: str = r"$b$",
    op_scaled_label: str = None,  # Can be customized, default based on op_label
    sus_scaled_label: str = None,  # Can be customized
    x_scaled_label: str = None,  # Can be customized
    base_markersize: int = mpl.rcParams['lines.markersize'],
    collapse_markersize_scale: float = 1.0,
    # Add parameters for filtering fixed values
    # e.g., {'p_mut_culture': 0.01}
    fixed_params_to_plot: Dict[str, Any] = None
):
    """
    Performs Finite-Size Scaling (FSS) analysis and plots:
    1. Order parameter at b_c vs L (log-log) -> Fig 3 estimate beta/nu
    2. Susceptibility peak vs L (log-log) -> Fig 4 estimate gamma/nu
    3. Data collapse plots (linear-linear axes) -> Fig 5
    Filters data based on fixed_params_to_plot before analysis.
    """
    print(
        f"Performing FSS analysis for op='{op_col_avg}', sus='{sus_col}' around b_c={b_c:.4f}")

    # --- Filter data for fixed parameters ---
    if fixed_params_to_plot is None:
        # Default to filtering by mutation rates if not specified
        fixed_params_to_plot = {
            'p_mut_culture': PARAMS.get('p_mut_culture', None),
            'p_mut_strategy': PARAMS.get('p_mut_strategy', None)
        }
        fixed_params_to_plot = {
            k: v for k, v in fixed_params_to_plot.items() if v is not None}

    filtered_data = filter_data_by_fixed_params(data, fixed_params_to_plot)
    if filtered_data.empty or not all(col in filtered_data.columns for col in [op_col_avg, sus_col, b_param_col, L_param_col]):
        print(
            f"Error in FSS: Filtered DataFrame empty or missing required columns: {op_col_avg}, {sus_col}, {b_param_col}, {L_param_col} for fixed params {fixed_params_to_plot}")
        return

    # Get the string representation of fixed parameters for title/filename
    fixed_params_str = get_fixed_params_string(fixed_params_to_plot)
    filename_base_with_params = filename_base_prefix + "_" + fixed_params_str.replace(
        '(', '').replace(')', '').replace(', ', '_').replace('=', '').replace('.', 'p')
    # Add bc to filename
    filename_base_with_params += f"_bc{b_c:.4f}".replace('.', 'p')

    # --- Preparations ---
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    unique_Ls = sorted(filtered_data[L_param_col].unique())
    num_L = len(unique_Ls)
    if num_L == 0:
        print(
            f"Error in FSS: No unique L values found in filtered data for fixed params {fixed_params_to_plot}.")
        return

    # Data storage for log-log plots
    fss_loglog_data = {'L': [], 'op_at_bc': [],
                       'sus_peak': [], 'sus_peak_b': []}

    # --- Create Figure for Collapse (Fig 5) ---
    fig_collapse, axes_collapse = plt.subplots(1, 2, figsize=(12, 5))
    ax_op_collapse, ax_sus_collapse = axes_collapse

    # Set default scaled labels if not customized
    if op_scaled_label is None:
        op_scaled_label = f"{op_label} $L^{{\\beta/\\nu}}$"
    if sus_scaled_label is None:
        sus_scaled_label = f"{sus_label} $L^{{-\\gamma/\\nu}}$"
    if x_scaled_label is None:
        x_scaled_label = f"$({b_label} - {b_label}_c) L^{{1/\\nu}}$"

    # --- Loop through L to gather data and plot collapse ---
    print("Processing data for FSS plots...")
    peak_idx = None
    for i, L in enumerate(tqdm(unique_Ls, desc="FSS per L")):
        # Filtered data is already for the correct fixed params, just subset by L
        subset = filtered_data[filtered_data[L_param_col]
                               == L].sort_values(by=b_param_col).reset_index()
        if subset.empty:
            continue

        # --- Data for Log-Log Plots (Fig 3 & 4) ---
        # Order parameter at b_c (find closest b value in data)
        # Ensure b_c is within the range of b_param_col values for this L subset, or extrapolation is okay
        if len(subset[b_param_col]) > 1:  # Need at least 2 b values to find closest
            # Ensure b_c is within or close to the range of b values for this L
            if b_c >= subset[b_param_col].min() and b_c <= subset[b_param_col].max():
                b_closest_idx = (subset[b_param_col] - b_c).abs().idxmin()
                op_at_bc = subset.loc[b_closest_idx, op_col_avg]
            else:
                # Handle cases where b_c is outside the range (e.g., boundary critical point)
                # Linear interpolation/extrapolation might be needed, but closest point is simpler
                print(
                    f"Warning: b_c={b_c:.4f} is outside the b range for L={L} ({subset[b_param_col].min():.3f} - {subset[b_param_col].max():.3f}). Using closest point.")
                b_closest_idx = (subset[b_param_col] - b_c).abs().idxmin()
                op_at_bc = subset.loc[b_closest_idx, op_col_avg]
                # op_at_bc = np.nan # Or set to NaN if extrapolation is not desired

            # Susceptibility peak value and location
            if sus_col in subset.columns:
                # Ensure susceptibility column has valid data before finding peak
                if subset[sus_col].replace([np.inf, -np.inf], np.nan).dropna().empty:
                    sus_peak_idx = None
                    sus_peak = np.nan
                    sus_peak_b = np.nan
                    # print(f"Warning: No valid susceptibility data found for peak detection for L={L}.")
                else:
                    sus_peak_idx = subset[sus_col].idxmax()
                    # Ensure index exists before accessing loc
                    if peak_idx in subset.index:
                        sus_peak = subset.loc[sus_peak_idx, sus_col]
                        sus_peak_b = subset.loc[sus_peak_idx, b_param_col]
                    else:
                        sus_peak = np.nan  # Should not happen with idxmax on valid data
                        sus_peak_b = np.nan
            else:
                sus_peak_idx = None
                sus_peak = np.nan
                sus_peak_b = np.nan
                # print(f"Warning: Susceptibility column '{sus_col}' not found for L={L}. Cannot plot Fig 4.")
        else:  # Not enough data points for this L
            op_at_bc = np.nan
            sus_peak = np.nan
            sus_peak_b = np.nan
            # print(f"Warning: Not enough data points ({len(subset)}) for L={L} to perform FSS analysis.")

        # Store log-log data if valid
        if pd.notna(op_at_bc) and pd.notna(L) and L > 0:
            fss_loglog_data['L'].append(L)
            fss_loglog_data['op_at_bc'].append(op_at_bc)

        if pd.notna(sus_peak) and pd.notna(L) and L > 0:
            # Only add L once if both OP and Sus are valid, but need unique entries for the peak dataframes
            fss_loglog_data['sus_peak'].append(sus_peak)
            fss_loglog_data['sus_peak_b'].append(sus_peak_b)
            # Add L here *only* if it wasn't just added for op_at_bc for the same i (implicit unique check by separate appends)
            if len(fss_loglog_data['L']) < len(fss_loglog_data['sus_peak']):
                # Add L again if this is a new L for sus_peak
                fss_loglog_data['L'].append(L)

        # --- Data for Collapse Plot (Fig 5) ---
        b_values = subset[b_param_col].values
        op_values = subset[op_col_avg].values
        sus_values = subset[sus_col].values

        # Filter invalid values for scaling (e.g., NaN, Inf, non-positive for logs if needed)
        valid_indices = pd.notna(op_values) & pd.notna(
            sus_values)  # Start with just NaN/Inf
        # Add checks for non-positive if taking log:
        # valid_indices = valid_indices & (op_values > 0) & (sus_values > 0) # For log scale

        if not np.any(valid_indices) or np.abs(nu) < 1e-9 or L is None or L <= 0 or pd.isna(L):
            # print(f"Skipping collapse plot for L={L} due to invalid data or nu={nu}.")
            continue  # Skip if no valid data, nu is zero/invalid, or L is invalid

        b_valid = b_values[valid_indices]
        op_valid = op_values[valid_indices]
        sus_valid = sus_values[valid_indices]

        # Calculate scaled variables
        t = b_valid - b_c  # Reduced parameter

        x_scaled = t * (L**(1/nu))
        y_op_scaled = op_valid * (L**(beta/nu))
        y_sus_scaled = sus_valid * (L**(-gamma/nu))

        # Plot collapse data
        style_kwargs = get_style_kwargs(i, num_L, base_markersize)
        current_collapse_markersize = base_markersize * collapse_markersize_scale

        ax_op_collapse.plot(x_scaled, y_op_scaled,
                            marker=style_kwargs['marker'],
                            color=style_kwargs['color'],
                            linestyle='',  # Points only for collapse
                            markersize=current_collapse_markersize,
                            # Label every L for collapse legend
                            label=f"$L={L}$")

        ax_sus_collapse.plot(x_scaled, y_sus_scaled,
                             marker=style_kwargs['marker'],
                             color=style_kwargs['color'],
                             linestyle='',
                             markersize=current_collapse_markersize,
                             label=f"$L={L}$")

    # --- Finalize Collapse Plot (Fig 5) ---
    ax_op_collapse.set_xlabel(x_scaled_label)
    ax_op_collapse.set_ylabel(op_scaled_label)
    # Add "Collapse" to title
    ax_op_collapse.set_title(f"Scaled Order Parameter ({op_label}) Collapse")
    ax_op_collapse.grid(True, linestyle=':', alpha=0.6)
    ax_op_collapse.legend(title='$L$', fontsize='small',
                          loc='best')  # Add legend to collapse plot

    ax_sus_collapse.set_xlabel(x_scaled_label)
    ax_sus_collapse.set_ylabel(sus_scaled_label)
    ax_sus_collapse.set_title(
        f"Scaled Susceptibility ({sus_label}) Collapse")  # Add "Collapse"
    ax_sus_collapse.grid(True, linestyle=':', alpha=0.6)
    ax_sus_collapse.legend(title='$L$', fontsize='small',
                           loc='best')  # Add legend

    fig_collapse.suptitle(
        f"Data Collapse ($b_c={b_c:.4f}, \\beta={beta:.3f}, \\gamma={gamma:.3f}, \\nu={nu:.3f}$)\n{fixed_params_str}", fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_plot(fig_collapse, f"{filename_base_with_params}_fig5_collapse")
    plt.close(fig_collapse)

    # --- Create and Plot Log-Log Plots (Fig 3 & 4) ---
    # Need to handle potential duplicates in L if both OP and Sus peaks were valid
    df_loglog_op = pd.DataFrame({k: fss_loglog_data[k] for k in [
                                'L', 'op_at_bc']}).dropna().drop_duplicates(subset=['L']).sort_values('L')
    # Ensure sus_peak_b corresponds to sus_peak, so group by L and get first valid peak/b_peak
    df_loglog_sus = pd.DataFrame({k: fss_loglog_data[k] for k in ['L', 'sus_peak', 'sus_peak_b']}).dropna(
        subset=['L', 'sus_peak']).groupby('L').first().reset_index().sort_values('L')

    beta_nu_fit = np.nan
    if not df_loglog_op.empty:
        # --- Fig 3: Order Parameter Scaling ---
        fig_log_op, ax_log_op = plt.subplots(figsize=(6, 4.5))
        ax_log_op.plot(df_loglog_op['L'], df_loglog_op['op_at_bc'],
                       marker='o', linestyle='None', color='blue')

        # Fit line: log10(op) = log10(C) - (beta/nu) * log10(L) -> op = C * L^(-beta/nu)
        if len(df_loglog_op) > 1:
            # Filter out non-positive op_at_bc if taking log
            fit_data_op = df_loglog_op[df_loglog_op['op_at_bc'] > 0].copy()
            if len(fit_data_op) > 1:
                log_L = np.log10(fit_data_op['L'])
                log_op = np.log10(fit_data_op['op_at_bc'])
                try:
                    coeffs = np.polyfit(log_L, log_op, 1)
                    beta_nu_fit = -coeffs[0]  # Slope is -(beta/nu)
                    log_C_fit = coeffs[1]  # Intercept is log10(C)

                    # Plot the fitted line across the range of L values used for fitting
                    fit_line_L = np.logspace(log_L.min(), log_L.max(), 50)
                    # Use fitted beta_nu
                    fit_line_op = 10**(log_C_fit + (-beta_nu_fit)
                                       * np.log10(fit_line_L))
                    ax_log_op.plot(fit_line_L, fit_line_op, 'r--', linewidth=1.5,
                                   label=f'Fit: $\\beta/\\nu \\approx {beta_nu_fit:.3f}$')
                    ax_log_op.legend(loc='best', frameon=False)
                except np.linalg.LinAlgError:
                    beta_nu_fit = np.nan
                    print("Warning: Could not perform linear fit for OP log-log plot.")
            else:
                beta_nu_fit = np.nan
        else:
            beta_nu_fit = np.nan

        ax_log_op.set_xlabel('System Size $L$')
        ax_log_op.set_ylabel(f'Order Parameter at $b_c$, {op_label}$(b_c)$')
        ax_log_op.set_xscale('log')
        ax_log_op.set_yscale('log')
        ax_log_op.set_title(
            f'Fig 3: Order Parameter Scaling at $b_c$\n{fixed_params_str}')
        ax_log_op.grid(True, which='both', linestyle=':', alpha=0.6)
        save_plot(fig_log_op, f"{filename_base_with_params}_fig3_op_scaling")
        plt.close(fig_log_op)
    else:
        print("Skipping Fig 3 (OP Scaling): No valid data collected.")
        beta_nu_fit = np.nan

    gamma_nu_fit = np.nan
    if not df_loglog_sus.empty:
        # --- Fig 4: Susceptibility Scaling ---
        fig_log_sus, ax_log_sus = plt.subplots(figsize=(6, 4.5))
        ax_log_sus.plot(df_loglog_sus['L'], df_loglog_sus['sus_peak'],
                        marker='s', linestyle='None', color='green')

        # Fit line: log10(chi) = log10(C) + (gamma/nu) * log10(L) -> chi = C * L^(gamma/nu)
        if len(df_loglog_sus) > 1:
            # Filter out non-positive sus_peak if taking log
            fit_data_sus = df_loglog_sus[df_loglog_sus['sus_peak'] > 0].copy()
            if len(fit_data_sus) > 1:
                log_L = np.log10(fit_data_sus['L'])
                log_sus = np.log10(fit_data_sus['sus_peak'])
                try:
                    coeffs = np.polyfit(log_L, log_sus, 1)
                    gamma_nu_fit = coeffs[0]  # Slope is gamma/nu
                    log_C_fit_sus = coeffs[1]  # Intercept is log10(C')

                    # Plot the fitted line across the range of L values used for fitting
                    fit_line_L = np.logspace(log_L.min(), log_L.max(), 50)
                    # Use fitted gamma_nu
                    fit_line_sus = 10**(log_C_fit_sus +
                                        gamma_nu_fit * np.log10(fit_line_L))
                    ax_log_sus.plot(fit_line_L, fit_line_sus, 'r--', linewidth=1.5,
                                    label=f'Fit: $\\gamma/\\nu \\approx {gamma_nu_fit:.3f}$')
                    ax_log_sus.legend(loc='best', frameon=False)
                except np.linalg.LinAlgError:
                    gamma_nu_fit = np.nan
                    print(
                        "Warning: Could not perform linear fit for Susceptibility log-log plot.")
            else:
                gamma_nu_fit = np.nan
        else:
            gamma_nu_fit = np.nan

        ax_log_sus.set_xlabel('System Size $L$')
        ax_log_sus.set_ylabel(f'Susceptibility Peak Value, max({sus_label})')
        ax_log_sus.set_xscale('log')
        ax_log_sus.set_yscale('log')
        ax_log_sus.set_title(
            f'Fig 4: Susceptibility Peak Scaling\n{fixed_params_str}')
        ax_log_sus.grid(True, which='both', linestyle=':', alpha=0.6)
        save_plot(fig_log_sus, f"{filename_base_with_params}_fig4_sus_scaling")
        plt.close(fig_log_sus)

        # Optional: Plot bc(L) scaling (Fig X)
        if len(df_loglog_sus) > 1 and pd.notna(nu) and nu != 0:
            fig_bcL, ax_bcL = plt.subplots(figsize=(6, 4))
            x_bc_scale = df_loglog_sus['L']**(-1/nu)
            ax_bcL.plot(
                x_bc_scale, df_loglog_sus['sus_peak_b'], 'd-', color='purple')
            ax_bcL.set_xlabel(r'$L^{-1/\nu}$')
            ax_bcL.set_ylabel(r'$b_c(L)$')
            ax_bcL.set_title(
                f'$b_c(L)$ Scaling (using $\\nu={nu:.3f}$)\n{fixed_params_str}')
            # ax_bcL.legend() # No legend needed for single line
            ax_bcL.grid(True, which='both', linestyle=':', alpha=0.6)
            save_plot(fig_bcL, f"{filename_base_with_params}_figX_bc_scaling")
            plt.close(fig_bcL)
        else:
            print("Skipping bc(L) scaling plot: Not enough data or invalid nu.")

    else:
        print("Skipping Fig 4 (Susceptibility Scaling) and bc(L) Scaling: No valid data collected.")
        gamma_nu_fit = np.nan

    print(
        f"FSS Log-Log Fit Estimates: beta/nu ~ {beta_nu_fit:.3f}, gamma/nu ~ {gamma_nu_fit:.3f}")
    # print("Compare these fits to your input beta/nu and gamma/nu estimates.")
    # Optional: Plot peak location bc(L) vs L -> estimate nu
    # fig_bcL, ax_bcL = plt.subplots(figsize=(6, 4))
    # ax_bcL.plot(df_loglog['L']**(-1/nu), df_loglog['sus_peak_b'], 'd-') # Example plot bc(L) vs L^(-1/nu)
    # save_plot(fig_bcL, f"{filename_base_prefix}_figX_bc_scaling")
    # plt.close(fig_bcL)


# --- Figure 6: Cluster Size Distribution P(s) ---
# Pass grouping_params to this function to correctly interpret the grouping_tuple key
def plot_fig6_cluster_dist(processed_ps_data, grouping_params_list, filename_base):
    """ Plots P(s) vs s on log-log axes for selected parameters. """
    print("Plotting Cluster Size Distribution P(s)...")
    if not processed_ps_data:
        print("Error plotting Fig 6: Processed P(s) data is missing or empty.")
        return
    if not grouping_params_list:
        print("Error plotting Fig 6: grouping_params_list is empty.")
        return

    # Select parameters to plot (e.g., near critical point, largest L?)
    # Let's plot multiple L for a specific b near the estimated critical point
    # and for the default mutation rates
    target_p_mut_culture = PARAMS.get('p_mut_culture', None)
    target_p_mut_strategy = PARAMS.get('p_mut_strategy', None)
    target_b = PARAMS.get(
        'fss_bc_estimate', PARAMS['cluster_analysis_b_values'][0])
    target_type = 'B'  # Plot Type B clusters (adjust if needed)

    # Build the target grouping tuple pattern
    # Values should match the order in grouping_params_list
    # We only know target values for 'b', 'p_mut_culture', 'p_mut_strategy'. 'L' will vary.
    # Need to find tuples that match fixed target parameters.
    target_fixed_params = {
        'b': target_b,
        'p_mut_culture': target_p_mut_culture,
        'p_mut_strategy': target_p_mut_strategy
        # Add other fixed parameters if they are in grouping_params_list
    }
    target_fixed_params = {k: v for k, v in target_fixed_params.items(
    ) if k in grouping_params_list and v is not None}

    # Find available grouping tuples that match the fixed target parameters
    available_grouping_tuples = list(
        set(k[0] for k in processed_ps_data.keys()))

    # Filter tuples based on fixed parameters
    plotting_grouping_tuples = []
    for gt in available_grouping_tuples:
        is_match = True
        for param_name, target_value in target_fixed_params.items():
            param_index = grouping_params_list.index(param_name)
            actual_value = gt[param_index]
            # Use isclose for float comparison
            # Check if both can be numeric
            if pd.api.types.is_numeric_dtype(np.array([actual_value, target_value])):
                if not np.isclose(actual_value, target_value, atol=1e-9, rtol=1e-9):
                    is_match = False
                    break
            elif actual_value != target_value:  # For non-numeric
                is_match = False
                break
        if is_match:
            plotting_grouping_tuples.append(gt)

    if not plotting_grouping_tuples:
        print(
            f"Error: No P(s) data found matching fixed parameters: {target_fixed_params}")
        return

    # Now refine target_b to be the closest actual b value among the filtered tuples
    available_bs_in_filtered = sorted(list(set(gt[grouping_params_list.index(
        'b')] for gt in plotting_grouping_tuples if 'b' in grouping_params_list)))

    if available_bs_in_filtered:
        target_b_actual = min(available_bs_in_filtered,
                              key=lambda x: abs(x-target_b))
        print(
            f"Plotting P(s) for b closest to estimate: {target_b_actual:.3f}")
        # Update target_fixed_params with the actual b value for filtering below
        target_fixed_params['b'] = target_b_actual
        # Re-filter tuples based on the actual b value
        plotting_grouping_tuples = [
            gt for gt in plotting_grouping_tuples
            # Use isclose
            if 'b' in grouping_params_list and np.isclose(gt[grouping_params_list.index('b')], target_b_actual, atol=1e-9, rtol=1e-9)
        ]
        if not plotting_grouping_tuples:
            print(
                f"Error: No P(s) data found for actual b={target_b_actual:.3f} after filtering.")
            return

    else:
        print("Error: No b values found in filtered P(s) data.")
        return

    # Get unique L values present in the filtered plotting tuples
    Ls = sorted(list(set(gt[grouping_params_list.index('L')]
                for gt in plotting_grouping_tuples if 'L' in grouping_params_list)))

    fig, ax = plt.subplots(figsize=(6, 4.5))
    base_markersize = mpl.rcParams['lines.markersize']
    plotted_data = False

    # --- Iterate through L and find the corresponding grouping_tuple ---
    for i, L in enumerate(Ls):
        # Find the specific grouping tuple for this L and all other fixed parameters
        relevant_tuple = None
        for gt in plotting_grouping_tuples:
            # Check if this tuple matches the current L and all other target fixed params
            is_match = True
            if 'L' in grouping_params_list and gt[grouping_params_list.index('L')] != L:
                is_match = False
            else:
                # Check other fixed parameters
                for param_name, target_value in target_fixed_params.items():
                    if param_name != 'b' and param_name != 'L' and param_name in grouping_params_list:
                        param_index = grouping_params_list.index(param_name)
                        actual_value = gt[param_index]
                        if pd.api.types.is_numeric_dtype(np.array([actual_value, target_value])):
                            if not np.isclose(actual_value, target_value, atol=1e-9, rtol=1e-9):
                                is_match = False
                                break
                        elif actual_value != target_value:
                            is_match = False
                            break
            if is_match:
                relevant_tuple = gt
                break  # Found the tuple for this L

        if relevant_tuple is None:
            # print(f"Warning: No specific tuple found for L={L} with fixed params {target_fixed_params}.")
            continue  # Skip this L if no matching tuple found

        key = (relevant_tuple, target_type)

        if key in processed_ps_data:
            s_values, ps_values = processed_ps_data[key]
            # Ensure data is valid for log-log plot
            valid_indices = (s_values > 0) & (ps_values > 0) & pd.notna(
                s_values) & pd.notna(ps_values)
            if np.any(valid_indices):
                s_plot = s_values[valid_indices]
                ps_plot = ps_values[valid_indices]
                style_kwargs = get_style_kwargs(i, len(Ls), base_markersize)
                # Use markers only for P(s)
                ax.plot(s_plot, ps_plot,
                        marker=style_kwargs['marker'],
                        color=style_kwargs['color'],
                        linestyle='',  # No line
                        markersize=style_kwargs['markersize'],
                        label=f'$L={L}$')
                plotted_data = True

    if not plotted_data:
        print(
            f"Error: No valid P(s) data found to plot for fixed parameters {target_fixed_params}, Type={target_type}.")
        plt.close(fig)
        return

    # --- Optional: Power Law Fit (Example on largest L data) ---
    # Find the tuple for the largest L among the ones actually plotted
    if Ls:
        largest_L_plotted = Ls[-1]
        relevant_tuple_largest_L = None
        for gt in plotting_grouping_tuples:
            if 'L' in grouping_params_list and gt[grouping_params_list.index('L')] == largest_L_plotted:
                relevant_tuple_largest_L = gt
                break

        if relevant_tuple_largest_L:
            key_large = (relevant_tuple_largest_L, target_type)
            if key_large in processed_ps_data:
                s_large, ps_large = processed_ps_data[key_large]
                valid_large = (s_large > 0) & (ps_large > 0) & pd.notna(
                    s_large) & pd.notna(ps_large)
                if np.any(valid_large):
                    s_fit_all = s_large[valid_large]
                    ps_fit_all = ps_large[valid_large]
                    # Select a fitting range (heuristic, needs tuning!)
                    # Avoid very small and very large clusters (finite size effects)
                    # Use 1% of total size as upper limit, or larger fixed number?
                    # L*L represents max size. Use a fraction, e.g., 0.05*L*L?
                    # Let's try a range like s > 3 and s < 100 (example, needs tuning)
                    fit_mask = (s_fit_all > 3) & (
                        s_fit_all < 100)  # Tune this range!
                    # Consider scaling the upper limit with L: s_fit_all < (largest_L_plotted**2 / 50) # Example scaled limit
                    if np.sum(fit_mask) > 2:  # Need at least 3 points for fit
                        s_to_fit = s_fit_all[fit_mask]
                        ps_to_fit = ps_fit_all[fit_mask]
                        try:
                            log_s = np.log10(s_to_fit)
                            log_ps = np.log10(ps_to_fit)
                            # Fit: log10(P) = log10(C) - tau * log10(s)
                            coeffs, cov = np.polyfit(
                                log_s, log_ps, 1, cov=True)
                            tau_fit = -coeffs[0]
                            log_C_fit = coeffs[1]
                            # Estimate error (diagonal of covariance matrix)
                            tau_err = np.sqrt(
                                np.diag(cov)[0]) if cov is not None and cov.ndim == 2 else np.nan

                            # Plot the fitted line only in the fitted range
                            s_line = np.logspace(
                                np.log10(s_to_fit.min()), np.log10(s_to_fit.max()), 50)
                            ps_line = (10**log_C_fit) * (s_line ** (-tau_fit))
                            ax.plot(s_line, ps_line, 'r--', linewidth=1.5,
                                    # Include error if desired
                                    label=f'Fit $(L={largest_L_plotted}): \\tau \\approx {tau_fit:.2f} \\pm {tau_err:.2f}$')
                            print(
                                f"Fitted P(s) exponent tau ~ {tau_fit:.3f} (err: {tau_err:.3f}) for L={largest_L_plotted}, b={target_b_actual:.3f}, Type={target_type}, Fixed: {target_fixed_params}")
                        except Exception as e:
                            print(
                                f"Could not perform/plot power-law fit for P(s): {e}")
                    else:
                        print(
                            f"Not enough points ({np.sum(fit_mask)}) in fitting range for power-law fit (L={largest_L_plotted}, b={target_b_actual:.3f}). Need at least 3.")
            # else: print(f"Warning: Data key {key_large} not found for fit on largest L ({largest_L_plotted}).")
        # else: print(f"Warning: Could not find tuple for largest L ({largest_L_plotted}) for fitting.")

    ax.set_xlabel('Cluster Size ($s$)')
    ax.set_ylabel('Probability Density ($P(s)$)')
    ax.set_xscale('log')
    ax.set_yscale('log')
    # Add fixed parameters to Fig 6 title
    fixed_params_str_title = get_fixed_params_string(target_fixed_params, params_to_display=[
                                                     k for k in grouping_params_list if k != 'L' and k != 'b'])
    ax.set_title(
        f'Cluster Size Distribution ($b={target_b_actual:.2f}$, Type {target_type}){fixed_params_str_title}')

    ax.legend(title='System Size $L$', loc='best', frameon=False)
    ax.grid(True, which='both', linestyle=':',
            alpha=0.4)  # Use grid for log-log

    # Update filename to include fixed parameters and actual b value
    filename = f"{filename_base}_b{target_b_actual:.2f}".replace('.', 'p')
    filename += f"_Type{target_type}"
    for param_name, value in target_fixed_params.items():
        if param_name != 'b':  # b is already included
            # Use .3g for general float fmt
            filename += f"_{param_name}{value:.3g}".replace('.', 'p')
    save_plot(fig, filename)
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
        print(
            f"Error plotting Fig 7: Missing one or more required columns: {required_keys}")
        print(f"Available columns: {data.columns.tolist()}")
        return

    # --- Filter data for fixed parameters ---
    fixed_params_to_plot = {
        'p_mut_culture': PARAMS.get('p_mut_culture', None),
        'p_mut_strategy': PARAMS.get('p_mut_strategy', None)
        # Add other fixed parameters you want to filter by
    }
    fixed_params_to_plot = {k: v for k,
                            v in fixed_params_to_plot.items() if v is not None}
    filtered_data = filter_data_by_fixed_params(data, fixed_params_to_plot)
    if filtered_data.empty:
        print(
            f"Warning: No data found for plotting {filename_base} with fixed parameters: {fixed_params_to_plot}")
        return

    # Get the string representation of fixed parameters for title/filename
    fixed_params_str = get_fixed_params_string(fixed_params_to_plot)
    filename_base_with_params = filename_base + "_" + fixed_params_str.replace(
        '(', '').replace(')', '').replace(', ', '_').replace('=', '').replace('.', 'p')

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True)
    base_markersize = mpl.rcParams['lines.markersize']

    # --- Plot 1: Boundary Fraction ---
    ax1 = axes[0]
    Ls = sorted(filtered_data['L'].unique())
    for i, L in enumerate(Ls):
        L_data = filtered_data[filtered_data['L'] == L].sort_values('b')
        if L_data.empty:
            continue
        style_kwargs = get_style_kwargs(i, len(Ls), base_markersize)
        label = f'$L={L}$'
        yerr = L_data[sem_boundary_frac].replace(
            [np.inf, -np.inf], np.nan).fillna(0)
        ax1.errorbar(L_data['b'], L_data[boundary_frac_key], yerr=yerr if np.any(yerr) else None,
                     label=label,  capsize=3, elinewidth=1, **style_kwargs)

    ax1.set_xlabel('Temptation ($b$)')
    ax1.set_ylabel('Boundary Fraction $\\langle f_{bound} \\rangle$')
    ax1.set_title(f'Fraction of Boundary Agents\n{fixed_params_str}')
    ax1.legend(title='System Size $L$', loc='best', frameon=False)
    ax1.set_ylim(bottom=-0.05)
    ax1.grid(False)

    # --- Plot 2: Boundary vs Bulk Cooperation ---
    ax2 = axes[1]
    # Plot for a representative L (e.g., largest)
    if not Ls:  # No L values found or processed
        print("Warning: No L values processed for plotting Fig 7.")
        plt.close(fig)
        return

    L_plot = Ls[-1]  # Use the largest L
    # Filter data specifically for the chosen L and fixed parameters
    L_data_plot = filtered_data[filtered_data['L'] == L_plot].sort_values('b')

    if not L_data_plot.empty:
        # Boundary Coop
        yerr_bnd = L_data_plot[sem_boundary_coop].replace(
            [np.inf, -np.inf], np.nan).fillna(0)
        ax2.errorbar(L_data_plot['b'], L_data_plot[boundary_coop_key],
                     yerr=yerr_bnd if np.any(yerr_bnd) else None,
                     label=f'Boundary Coop.',
                     fmt='-', capsize=3, elinewidth=1, color='red', marker='o', markersize=base_markersize)
        # Bulk Coop
        yerr_blk = L_data_plot[sem_bulk_coop].replace(
            [np.inf, -np.inf], np.nan).fillna(0)
        ax2.errorbar(L_data_plot['b'], L_data_plot[bulk_coop_key],
                     yerr=yerr_blk if np.any(yerr_blk) else None,
                     label=f'Bulk Coop.',
                     fmt='-', capsize=3, elinewidth=1, color='blue', marker='s', markersize=base_markersize)
    else:
        print(
            f"Warning: No data found for L={L_plot} for boundary/bulk coop plot (Fig 7b) after filtering.")

    ax2.set_xlabel('Temptation ($b$)')
    ax2.set_ylabel('Avg. Cooperation Rate $\\langle f_C \\rangle$')
    ax2.set_title(
        f'Boundary vs. Bulk Cooperation ($L={L_plot}$)\n{fixed_params_str}')
    ax2.legend(loc='best', frameon=False)
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(False)

    plt.tight_layout()
    save_plot(fig, filename_base_with_params)
    plt.close(fig)


# --- Figure 8: Phase Diagram ---

# --- Figure 8: Phase Diagram ---

def plot_fig8_phase_diagram(data, filename_base):
    """ Plots a 2D phase diagram (heatmap). """
    print("Plotting Phase Diagram...")
    if data.empty:
        print("Error plotting Fig 8: Phase diagram data is empty.")
        return

    p1_name = PARAMS['phasediagram_param1_name']
    p2_name = PARAMS['phasediagram_param2_name']
    # e.g., 'avg_CooperationRate'
    target_reporter_avg = PARAMS['phasediagram_target_reporter']

    if not all(k in data.columns for k in [p1_name, p2_name, target_reporter_avg]):
        print(
            f"Error plotting Fig 8: Missing required columns. Need: {p1_name}, {p2_name}, {target_reporter_avg}")
        print(f"Available cols: {data.columns.tolist()}")
        return

    # Identify other fixed parameters for this phase diagram scan
    sim_config_fields = {f.name for f in fields(SimConfig)}
    # Columns in data that are SimConfig fields but NOT p1_name, p2_name, or run identifiers
    fixed_params_cols = [
        col for col in data.columns
        if col in sim_config_fields and col not in [p1_name, p2_name, 'seed', 'run_id', 'param_set_id', 'label', 'steps', 'steady_state_window']
    ]
    # Ensure L is included if it's not swept but is fixed
    if 'L' in data.columns and 'L' not in [p1_name, p2_name] and 'L' not in fixed_params_cols:
        fixed_params_cols.append('L')

    # Group by the sweep parameters and fixed parameters to average over runs
    grouping_cols = [p1_name, p2_name] + sorted(fixed_params_cols)
    print(f"Grouping phase diagram data by: {grouping_cols}")

    try:
        # Group by all relevant columns (sweep + fixed) before calculating mean
        grouped_pd = data.groupby(grouping_cols)[
            target_reporter_avg].mean().reset_index()

        # Select one set of fixed parameters to plot the heatmap
        # Get the first unique combination of fixed parameters
        if fixed_params_cols:
            first_fixed_params_values = grouped_pd[fixed_params_cols].iloc[0].to_dict(
            )
            print(
                f"Plotting phase diagram for fixed parameters: {first_fixed_params_values}")
            # Filter the grouped data to only include this set of fixed parameters
            filtered_grouped_pd = filter_data_by_fixed_params(
                grouped_pd, first_fixed_params_values)
        else:
            print("No additional fixed parameters found for phase diagram.")
            filtered_grouped_pd = grouped_pd
            first_fixed_params_values = {}  # Empty dict

        if filtered_grouped_pd.empty:
            print(
                f"Error plotting Fig 8: Filtered grouped data is empty for fixed params {first_fixed_params_values}.")
            return

        heatmap_data = filtered_grouped_pd.pivot(
            index=p2_name, columns=p1_name, values=target_reporter_avg)

    except Exception as e:
        print(
            f"Error processing phase diagram data or pivoting for heatmap: {e}")
        import traceback
        traceback.print_exc()
        return

    if heatmap_data.empty:
        print(
            f"Error plotting Fig 8: Heatmap data is empty after pivot for fixed params {first_fixed_params_values}.")
        return

    fig, ax = plt.subplots(figsize=(7, 5.5))
    sns.heatmap(heatmap_data, ax=ax, cmap="viridis",  # Use perceptually uniform colormap
                annot=False,  # Annotate values only if grid is small
                # fmt=".2f", # Format for annotation
                # LaTeX label
                cbar_kws={'label': target_reporter_avg.replace('avg_', '$\\langle$') + '$\\rangle$'})

    # Improve axis labels
    ax.set_xlabel(f'{p1_name}' if p1_name != 'b' else 'Temptation ($b$)')
    ax.set_ylabel(f'{p2_name}' if p2_name !=
                  'K_C' else 'Cultural Noise ($K_C$)')

    # Add fixed parameters to Fig 8 title
    fixed_params_str_title = get_fixed_params_string(first_fixed_params_values)
    ax.set_title(f'Phase Diagram {fixed_params_str_title}')

    # Ensure correct orientation (heatmap index often becomes y-axis)
    # If index values are decreasing, seaborn plots them from bottom to top by default
    # To have smallest value at the bottom, y-axis should not be inverted
    # Let's keep default heatmap behavior unless needed

    # Manually set y-axis ticks and labels
    y_labels = heatmap_data.index.tolist()
    if y_labels:
        y_ticks = np.arange(len(y_labels)) + 0.5  # heatmap ticks are centered
        ax.set_yticks(y_ticks)
        # Format labels - use 4 significant figures for general parameters, more for specific ones if needed
        ax.set_yticklabels([f'{y:.4g}' for y in y_labels])
    # Rotate y-axis labels if needed (e.g., if they overlap)
    # plt.yticks(rotation=0)

    # Manually set x-axis ticks and labels
    x_labels = heatmap_data.columns.tolist()
    if x_labels:
        x_ticks = np.arange(len(x_labels)) + 0.5
        ax.set_xticks(x_ticks)
        # Format labels - use 3 decimal places for 'b', 4 significant figures for others
        ax.set_xticklabels(
            [f'{x:.3f}' if p1_name == 'b' else f'{x:.4g}' for x in x_labels])
    # Rotate x-axis labels if needed
    # plt.xticks(rotation=45, ha='right') # Example rotation

    plt.tight_layout()

    # Update filename to include fixed parameters used for this specific heatmap
    filename = filename_base
    # Add fixed parameters to filename, except L which might be implicitly included in phasediagram_L name
    # Let's include all fixed params for clarity in filename
    for param_name, value in first_fixed_params_values.items():
        # Use .4g for consistency
        filename += f"_{param_name}{value:.4g}".replace('.', 'p')

    save_plot(fig, filename)
    plt.close(fig)


# --- Snapshot Plotting ---

def run_and_save_snapshot(b_value, L_snap, filename_tag="snap"):
    """ Runs a single simulation and saves the final grid state. """
    # Get fixed parameters from PARAMS for this snapshot run
    snapshot_fixed_params = {
        'L': L_snap,
        'b': b_value,
        'p_mut_culture': PARAMS.get('p_mut_culture', None),
        'p_mut_strategy': PARAMS.get('p_mut_strategy', None),
        # Add other fixed parameters like K, K_C, p_update_C etc. from PARAMS
        # Ensure these match what CulturalGame constructor expects and what's in PARAMS
        'K': PARAMS.get('K'),
        'K_C': PARAMS.get('K_C'),
        'p_update_C': PARAMS.get('p_update_C'),
        'initial_coop_ratio': PARAMS.get('initial_coop_ratio'),
        'C_dist': PARAMS.get('C_dist'),
        'mu': PARAMS.get('mu'),
        'sigma': PARAMS.get('sigma'),
    }
    # Remove None values
    snapshot_fixed_params = {
        k: v for k, v in snapshot_fixed_params.items() if v is not None}

    # Generate filename based on these fixed parameters
    filename = f"snapshot_{filename_tag}_L{L_snap}_b{b_value:.2f}".replace(
        '.', 'p')
    # Add other fixed parameters to filename
    for param_name, value in snapshot_fixed_params.items():
        if param_name not in ['L', 'b']:  # Already included L and b
            # Use .3g for consistency
            filename += f"_{param_name}{value:.3g}".replace('.', 'p')

    snapshot_filename = os.path.join(SNAPSHOT_DATA_DIR, f"{filename}.pkl")

    # Check if snapshot exists
    if not FORCE_RERUN_SNAPSHOTS and os.path.exists(snapshot_filename):
        print(f"Skipping run, snapshot exists: {snapshot_filename}")
        return snapshot_filename

    print(f"Running simulation for snapshot (L={L_snap}, b={b_value:.2f})...")
    # Build parameters dictionary for CulturalGame constructor
    # Use snapshot_fixed_params as the base, add run-specific params (seed, steps)
    model_params = snapshot_fixed_params.copy()
    # Use time for uniqueness if not seeding runs
    model_params['seed'] = int(
        time.time() * 1000 + b_value * 100 + L_snap) % (2**32 - 1)
    # Use the total simulation steps for snapshots
    steps_to_run = PARAMS['steps']

    try:
        # Initialize the model with the collected parameters
        model = CulturalGame(**model_params)

        # Run the model silently
        for _ in range(steps_to_run):
            model.step()

    except Exception as e:
        print(
            f"Error running snapshot simulation (L={L_snap}, b={b_value}): {e}")
        import traceback
        traceback.print_exc()
        return None

    # --- Grid state extraction logic ---
    # This part relies on _get_agent_type and agent attributes (strategy, C)
    # It assumes agents have a 'pos' attribute and can be iterated via model.schedule.agents
    # It also assumes CulturalAgent has 'strategy' and 'C'
    # x, y, [strategy, C, type_id]
    grid_state = np.zeros((model.grid.width, model.grid.height, 3))
    agent_types_map = {}  # Map type_id to description
    TYPE_A_ID = 1
    TYPE_B_ID = 2
    threshold = 0.5  # Assuming fixed threshold for snapshot visualization - ideally, this should be saved in params

    if TYPE_A_ID not in agent_types_map:
        agent_types_map[TYPE_A_ID] = f"Type A (C<{threshold})"
    if TYPE_B_ID not in agent_types_map:
        agent_types_map[TYPE_B_ID] = f"Type B (C>={threshold})"

    for agent in model.schedule.agents:
        x, y = agent.pos
        if x is None or y is None:
            continue
        strategy = agent.strategy  # 0 or 1
        culture = agent.C         # 0 to 1
        # Pass threshold explicitly
        agent_type = _get_agent_type(agent, threshold=threshold)
        if agent_type == 'A':
            agent_type_id = TYPE_A_ID
        elif agent_type == 'B':
            agent_type_id = TYPE_B_ID
        else:
            agent_type_id = 0  # Or some other indicator

        grid_state[x, y, 0] = strategy
        grid_state[x, y, 1] = culture
        grid_state[x, y, 2] = agent_type_id

    # Also save the actual steps run and the threshold used
    # model_params already has correct keys
    snapshot_params_saved = model_params.copy()
    snapshot_params_saved['steps_run'] = steps_to_run
    # Save the threshold used for typing
    snapshot_params_saved['threshold'] = threshold

    snapshot_data = {'grid': grid_state,
                     'params': snapshot_params_saved, 'types': agent_types_map}
    # --- Saving logic ---
    try:
        with open(snapshot_filename, 'wb') as f:
            pickle.dump(snapshot_data, f)
        print(f"Snapshot data saved to {snapshot_filename}")
        return snapshot_filename
    except Exception as e:
        print(f"Error saving snapshot data to {snapshot_filename}: {e}")
        return None


def plot_snapshot(snapshot_data_file):
    """ Plots Spatial Snapshot with Physica A style. """
    if not snapshot_data_file or not os.path.exists(snapshot_data_file):
        print(f"Snapshot file not found or invalid: {snapshot_data_file}")
        return

    print(f"Plotting Snapshot from {os.path.basename(snapshot_data_file)}...")
    try:
        with open(snapshot_data_file, 'rb') as f:
            snapshot_data = pickle.load(f)
        grid = snapshot_data['grid']
        # Contains saved parameters including mutation rates
        params = snapshot_data['params']
        type_desc = snapshot_data['types']  # {1: 'Type A...', 2: 'Type B...'}
        L = params['L']
        b_val = params['b']
        # Get mutation rates and other fixed params from loaded params
        pm_culture = params.get('p_mut_culture')
        pm_strategy = params.get('p_mut_strategy')
        # Get threshold used, default if not saved
        threshold = params.get('threshold', 0.5)
        # Get other relevant parameters for title if needed
        k_val = params.get('K')
        kc_val = params.get('K_C')

    except Exception as e:
        print(f"Error loading snapshot data from {snapshot_data_file}: {e}")
        return

    img = np.zeros((L, L, 3))  # RGB image
    TYPE_A_ID = 1
    TYPE_B_ID = 2
    STRAT_D = 0  # Defector
    STRAT_C = 1  # Cooperator

    # Define colors (adjust for clarity/preference)
    color_map = {
        # Dark Blue (A, Defect)
        (TYPE_A_ID, STRAT_D): np.array([0.0, 0.0, 0.6]),
        # Light Blue (A, Coop)
        (TYPE_A_ID, STRAT_C): np.array([0.6, 0.8, 1.0]),
        # Dark Red (B, Defect)
        (TYPE_B_ID, STRAT_D): np.array([0.6, 0.0, 0.0]),
        # Pink/Light Red (B, Coop)
        (TYPE_B_ID, STRAT_C): np.array([1.0, 0.6, 0.6]),
        'default': np.array([0.5, 0.5, 0.5])  # Grey for unknown type_id
    }
    # Update legend labels to use the saved threshold
    legend_labels = {
        (TYPE_A_ID, STRAT_D): f"Type A (C<{threshold:.2f}) / Defect",
        (TYPE_A_ID, STRAT_C): f"Type A (C<{threshold:.2f}) / Cooperate",
        (TYPE_B_ID, STRAT_D): f"Type B (C>={threshold:.2f}) / Defect",
        (TYPE_B_ID, STRAT_C): f"Type B (C>={threshold:.2f}) / Cooperate",
    }

    for x in range(L):
        for y in range(L):
            strat = int(grid[x, y, 0])  # Strategy
            # Culture (float) - not directly used for color, only for type_id
            culture = grid[x, y, 1]
            agent_type_id = int(grid[x, y, 2])  # Pre-calculated type ID

            key = (agent_type_id, strat)
            # imshow expects (height, width) = (y, x)
            img[y, x, :] = color_map.get(key, color_map['default'])

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(img, origin='lower', interpolation='nearest')

    # Create custom legend
    valid_legend_keys = [key for key in color_map.keys(
    ) if key != 'default' and key in legend_labels]
    # Sort keys for consistent legend order
    sorted_legend_keys = sorted(valid_legend_keys)
    legend_elements = [Patch(facecolor=color_map[key], edgecolor='k', linewidth=0.5, label=legend_labels[key])
                       for key in sorted_legend_keys]

    ax.legend(handles=legend_elements, bbox_to_anchor=(1.02, 1),
              loc='upper left', title="Agent State", fontsize='small')

    # Add fixed parameters to snapshot plot title
    title_str = f'Spatial Snapshot ($L={L}, b={b_val:.2f}'
    if pm_culture is not None:
        # Use .3g for consistency
        title_str += f", $p_{{mut,culture}}={pm_culture:.3g}$"
    if pm_strategy is not None:
        title_str += f", $p_{{mut,strategy}}={pm_strategy:.3g}$"  # Use .3g
    if k_val is not None:
        title_str += f", $K={k_val:.2g}$"  # Use .2g for K
    if kc_val is not None:
        title_str += f", $K_C={kc_val:.2g}$"  # Use .2g
    title_str += ')'
    ax.set_title(title_str)
    ax.set_xticks([])
    ax.set_yticks([])

    plt.tight_layout(rect=[0, 0, 0.75, 1])  # Adjust right margin for legend
    # Use the filename that was saved (which includes fixed parameters)
    saved_filename_base = os.path.splitext(
        os.path.basename(snapshot_data_file))[0]
    save_plot(fig, saved_filename_base)  # Save using the saved filename
    plt.close(fig)


# ==============================================================================
# Main Execution Workflow
# ==============================================================================
if __name__ == "__main__":
    print("--- Starting Physica A Analysis Workflow ---")
    overall_start_time = time.time()

    # --- Control Flags ---
    # Set True to force regeneration of data even if files exist
    FORCE_RERUN_MAIN = False
    FORCE_RERUN_PHASEDIAGRAM = False
    FORCE_RERUN_SNAPSHOTS = False

    # --- 1. Run/Load Main L vs b Scan ---
    # This scan assumes other parameters (like mutation rates, K_C) are fixed at PARAMS values.
    print("\n--- Running/Loading Main Scan ---")
    main_raw_data = run_simulation_batch(
        generate_main_scan_configs, MAIN_DATA_FILENAME, force_rerun=FORCE_RERUN_MAIN)

    # --- 2. Process Main Scan Data (Includes Cluster Aggregation) ---
    # This step groups the data by all parameters except run-specific ones.
    # The grouping_params are stored as an attribute in the processed_df.
    print("\n--- Processing Main Scan Data ---")
    main_processed_data, processed_ps_data = process_main_data(main_raw_data)

    # --- 3. Run/Load Phase Diagram Scan ---
    # This scan assumes L and other non-swept parameters are fixed at PARAMS values.
    print("\n--- Running/Loading Phase Diagram Scan ---")
    phasediagram_raw_data = run_simulation_batch(
        generate_phasediagram_configs, PHASEDIAGRAM_DATA_FILENAME, force_rerun=FORCE_RERUN_PHASEDIAGRAM)

    # --- 4. Generate/Check Snapshot Data ---
    # Snapshots are generated for specific L, b, and fixed parameters (from PARAMS).
    print("\n--- Generating/Checking Snapshots ---")
    snapshot_files = []
    snap_L = PARAMS['snapshot_L']
    for b_snap in PARAMS['snapshot_b_values']:
        # Use a filename tag, e.g., based on C_dist or other key features
        fname = run_and_save_snapshot(
            b_snap, snap_L, filename_tag=PARAMS.get('C_dist', 'snap'))
        if fname:
            snapshot_files.append(fname)

    # --- 5. Generate Plots ---
    print("\n--- Generating Plots ---")
    if not main_processed_data.empty:
        # Retrieve grouping parameters from the dataframe attributes
        main_grouping_params = main_processed_data.attrs.get(
            'grouping_params', ['L', 'b'])  # Default fallback

        # --- Fig 1: Order Parameters vs b (Multi-L) ---
        # Plotting functions automatically filter based on default PARAMS values for mutation rates
        plot_fig1_order_param(main_processed_data, 'avg_CooperationRate', 'sem_CooperationRate',
                              'Avg. Cooperation Rate $\\langle f_C \\rangle$', "fig1a_coop_rate_vs_b")
        plot_fig1_order_param(main_processed_data, 'avg_SegregationIndex', 'sem_SegregationIndex',
                              'Avg. Segregation Index $\\langle S \\rangle$', "fig1b_segregation_vs_b")

        # --- Fig 2: Susceptibilities vs b (Multi-L) ---
        # Plotting functions automatically filter
        peaks_fc = plot_fig2_susceptibility(main_processed_data, 'chi_CooperationRate',
                                            'Susceptibility $\\chi_{f_C}$', "fig2a_susc_coop_rate")
        peaks_s = plot_fig2_susceptibility(main_processed_data, 'chi_SegregationIndex',
                                           r'Susceptibility $\chi_S$', "fig2b_susc_segregation")

        # --- Figs 3, 4, 5: FSS Analysis ---
        print("\n--- Performing FSS Analysis ---")
        # Use estimated exponents from PARAMS. Refine b_c based on peaks if desired.
        # You should manually check peaks and update fss_bc_estimate in PARAMS for accurate FSS.
        # Start with param, provide default
        bc_refined_fc = PARAMS.get('fss_bc_estimate', 2.5)
        # Optional: Refine bc estimate using peak location from largest L's susceptibility
        # if not peaks_fc.empty and pd.notna(peaks_fc.get(PARAMS['L_values'][-1])):
        #     bc_refined_fc = peaks_fc.get(PARAMS['L_values'][-1])
        #     print(f"Refined b_c estimate using peak of largest L susceptibility: {bc_refined_fc:.4f}")
        # else:
        #     print(f"Warning: Could not refine b_c from susceptibility peaks. Using estimate from PARAMS: {bc_refined_fc:.4f}")

        # Call FSS plot function, it will filter by default mutation rates
        plot_fss_analysis(main_processed_data,
                          op_col_avg='avg_CooperationRate',
                          sus_col='chi_CooperationRate',
                          b_param_col='b',
                          L_param_col='L',
                          b_c=bc_refined_fc,
                          beta=PARAMS.get('fss_beta_estimate', 0.125),
                          gamma=PARAMS.get('fss_gamma_estimate', 1.75),
                          nu=PARAMS.get('fss_nu_estimate', 1.0),
                          filename_base_prefix="fss_cooperation",
                          op_label=r"$\langle f_C \rangle$",
                          sus_label=r"$\chi_{f_C}$",
                          b_label=r"$b$",
                          # fixed_params_to_plot will default to mutation rates
                          )

        # Optional: FSS for Segregation Index (if it shows criticality)
        # plot_fss_analysis(main_processed_data, 'avg_SegregationIndex', 'chi_SegregationIndex', ...)

        # --- Fig 7: Boundary Effects ---
        # Plotting function automatically filters
        plot_fig7_boundary_effects(
            main_processed_data, "fig7_boundary_effects")

    else:
        print(
            "Skipping plots based on main scan data (Figs 1, 2, FSS, 7) as data is empty.")

    # --- Fig 6: Cluster Distribution ---
    # Pass the grouping_params list used in process_main_data to interpret keys correctly
    if processed_ps_data and 'grouping_params' in main_processed_data.attrs:
        plot_fig6_cluster_dist(
            processed_ps_data, main_processed_data.attrs['grouping_params'], "fig6_cluster_dist")
    else:
        print("Skipping Fig 6 (Cluster Distribution) due to missing/empty processed P(s) data or grouping parameters.")

    # --- Fig 8: Phase Diagram ---
    if not phasediagram_raw_data.empty:
        # Plotting function identifies fixed parameters automatically from the data
        plot_fig8_phase_diagram(phasediagram_raw_data, "fig8_phase_diagram")
    else:
        print("Skipping Fig 8 (Phase Diagram) due to missing/empty data.")

    # --- Plot Snapshots ---
    print("\n--- Plotting Snapshots ---")
    if not snapshot_files:
        print("No snapshot files found or generated to plot.")
    else:
        for snap_file in snapshot_files:
            plot_snapshot(snap_file)

    overall_end_time = time.time()
    print(
        f"\n--- Workflow finished in {(overall_end_time - overall_start_time):.2f} seconds ---")
    print(f"Data saved in: '{DATA_SAVE_DIR}'")
    print(f"Plots saved in: '{PLOT_SAVE_DIR}'")
