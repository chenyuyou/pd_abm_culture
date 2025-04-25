# plot_susceptibility_focused.py
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
from matplotlib import cm
from dataclasses import fields
from typing import Dict, Any

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

# --- Import Necessary Plotting Helpers from plot_figures.py ---
# (We copy/paste or import these helper functions)
# Assume these functions are available (either copied or imported)
from plot_figures import (
    get_style_kwargs,
    save_plot,
    filter_data_by_fixed_params,
    get_fixed_params_string
)

# ==============================================================================
# Simulation Parameters for FOCUSED Susceptibility Calculation
# ==============================================================================
SUS_PARAMS = {
    # --- Core Model Parameters (MUST MATCH plot_figures.py PARAMS for fixed values) ---
    'initial_coop_ratio': 0.5,
    'K': 0.1,
    'K_C': 0.1,
    'p_update_C': 0.1,
    'p_mut_culture': 0.01,      # Probability of random cultural mutation per step
    'p_mut_strategy': 0.001,    # Probability of random strategy mutation per step
    'C_dist': 'bimodal',
    'mu': 0.5,
    'sigma': 0.1,

    # --- Simulation Control (KEY DIFFERENCE) ---
    'sus_steps': 15000,        # <<< INCREASED STEPS SIGNIFICANTLY FOR SUSCEPTIBILITY >>>
    'steady_state_window': 1500, # Window for averaging within each run (adjust if needed, e.g., last 10-20% of sus_steps)
    'runs_per_parameter_set': 100, # Keep the high number of runs for statistics

    # --- Scan Parameters (Match plot_figures.py) ---
    'L_values': [20, 30, 40, 50],
    'b_values': np.unique(np.concatenate((
        np.linspace(1.3, 1.9, 4),
        np.linspace(2.0, 3.0, 21), # Keep high density near expected peak
        np.linspace(3.2, 4.0, 5),
        np.linspace(4.5, 7.0, 6)
    ))),
    'susceptibility_target_names': ['CooperationRate', 'SegregationIndex'] # Reporters for Chi calc
}

# Ensure steady_state_window is reasonable compared to sus_steps
SUS_PARAMS['steady_state_window'] = max(100, int(SUS_PARAMS['sus_steps'] * 0.1)) # E.g., use last 10% or min 100 steps

SUS_DATA_SAVE_DIR = "simulation_data_susceptibility"
SUS_PLOT_SAVE_DIR = "plots_susceptibility"
os.makedirs(SUS_DATA_SAVE_DIR, exist_ok=True)
os.makedirs(SUS_PLOT_SAVE_DIR, exist_ok=True)

# Use specific data filename for susceptibility raw run data
SUS_DATA_FILENAME = os.path.join(SUS_DATA_SAVE_DIR, "susceptibility_scan_raw_data.pkl")

# ==============================================================================
# Physica A Style Settings (Copied from plot_figures.py)
# ==============================================================================
mpl.rcParams.update({
    'font.size': 10, 'axes.labelsize': 12, 'axes.titlesize': 14,
    'xtick.labelsize': 10, 'ytick.labelsize': 10, 'legend.fontsize': 9,
    'legend.title_fontsize': 10, 'figure.dpi': 300, 'savefig.dpi': 300,
    'savefig.format': 'png', 'savefig.bbox': 'tight',
    'lines.linewidth': 1.5, 'lines.markersize': 4,
    'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
    'mathtext.fontset': 'cm', 'axes.grid': False,
})
MARKERS = ['o', 's', '^', 'D', 'v', 'p', '*', 'X']
LINESTYLES = ['-', '--', ':', '-.']

# ==============================================================================
# Focused Config Generation and Simulation Execution
# ==============================================================================

def generate_sus_scan_configs():
    """Generates SimConfig objects specifically for the susceptibility scan."""
    configs = []
    base_params = SUS_PARAMS.copy()
    sweep_params_sus = ['L_values', 'b_values']
    run_specific_params = ['runs_per_parameter_set', 'sus_steps', 'seed', 'param_set_id', 'run_id', 'steady_state_window']
    # Remove keys not needed for SimConfig or that are swept/run-specific
    sim_config_fields = {f.name for f in fields(SimConfig)}

    # Filter base_params to only include valid SimConfig fields, excluding sweep/run-specific
    valid_base_params = {}
    for k, v in base_params.items():
        if k in sim_config_fields and k not in sweep_params_sus and k not in run_specific_params:
            valid_base_params[k] = v

    # Add steps from SUS_PARAMS specifically
    valid_base_params['steps'] = SUS_PARAMS['sus_steps'] # Use the longer steps count

    param_list = []
    for L in SUS_PARAMS['L_values']:
        for b in SUS_PARAMS['b_values']:
            current_params = valid_base_params.copy()
            current_params['L'] = L
            current_params['b'] = b
            # Generate a unique ID including fixed parameters
            id_parts = [f"Sus_L{L}", f"b{b:.3f}".replace('.', 'p')]
            if 'p_mut_culture' in SUS_PARAMS: id_parts.append(f"pmC{SUS_PARAMS['p_mut_culture']:.3f}".replace('.', 'p'))
            if 'p_mut_strategy' in SUS_PARAMS: id_parts.append(f"pmS{SUS_PARAMS['p_mut_strategy']:.3f}".replace('.', 'p'))
            current_params['param_set_id'] = "_".join(id_parts)
            param_list.append(current_params)

    # Create SimConfig objects for each parameter set, repeated for each run
    for i in range(SUS_PARAMS['runs_per_parameter_set']):
         run_seed_base = i * 10000 + 5000 # Use a different seed offset
         for idx, params in enumerate(param_list):
              run_params = params.copy()
              run_params['seed'] = run_seed_base + idx
              run_params['run_id'] = i
              try:
                  configs.append(SimConfig(**run_params))
              except TypeError as e:
                   print(f"Error creating SimConfig. Params: {run_params}\nError: {e}")
                   # Check if all required fields in SimConfig are present in run_params
                   # print(f"SimConfig fields: {sim_config_fields}")
                   # print(f"Provided keys: {run_params.keys()}")

    print(f"Generated {len(configs)} SimConfig objects for susceptibility scan (Steps: {SUS_PARAMS['sus_steps']}).")
    return configs


def run_sus_simulation_batch(config_generator, data_filename, force_rerun=False):
    """
    Runs a batch of simulations specifically for susceptibility, using parallel.py or sequentially.
    Reuses the logic from plot_figures.py's run_simulation_batch but uses SUS_PARAMS.
    """
    if not force_rerun and os.path.exists(data_filename):
        print(f"Loading existing susceptibility raw data from {data_filename}...")
        try:
            with open(data_filename, 'rb') as f:
                results_df = pickle.load(f)
            print(f"Loaded {len(results_df)} susceptibility run results.")
            return results_df
        except Exception as e:
            print(f"Error loading data from {data_filename}: {e}. Rerunning...")
            force_rerun = True # Force rerun if loading fails

    print(f"Running simulations for {os.path.basename(data_filename)} (Steps: {SUS_PARAMS['sus_steps']})...")
    sim_configs = config_generator()
    if not sim_configs:
         print("Warning: No configurations generated for susceptibility scan.")
         return pd.DataFrame()

    # --- Parallel Execution ---
    if RUN_WITH_PARALLEL:
        # batch_run_parallel expects SimConfig objects and steady_state_window
        results_df = batch_run_parallel(
            sim_configs,
            steady_state_window=SUS_PARAMS['steady_state_window']
            # num_workers defaults in batch_run_parallel
        )
    # --- Sequential Execution (Fallback) ---
    else:
        print("Running sequentially for susceptibility (This will be slow)...")
        all_results = []
        # NON_AVERAGE_REPORTERS should match definition in run_single_simulation if used
        NON_AVERAGE_REPORTERS = ["ClusterSizeDistribution"] # Keep consistent if used
        with tqdm(total=len(sim_configs), desc=f"Sus Sims ({os.path.basename(data_filename)})") as pbar:
             for config in sim_configs:
                 try:
                      # Pass config fields directly to CulturalGame
                      model = CulturalGame(**config.to_dict()) # Assumes to_dict() provides correct args
                      # Use the steps defined in the config (which came from sus_steps)
                      for _ in range(config.steps):
                          model.step()

                      model_df = model.datacollector.get_model_vars_dataframe()
                      result_dict = config.to_dict() # Start with parameters
                      n_rows = len(model_df)

                      if n_rows > 0:
                          start_idx = max(0, n_rows - SUS_PARAMS['steady_state_window'])
                          window_df = model_df.iloc[start_idx:]

                          # Calculate avg/std or get last value for each reporter FOR THIS RUN
                          for col in model.datacollector.model_reporters.keys():
                              if col in model_df.columns:
                                  if col in NON_AVERAGE_REPORTERS:
                                      result_dict[f"{col}"] = model_df[col].iloc[-1]
                                  elif not window_df.empty:
                                      # IMPORTANT: Store the per-run average needed for Chi calculation later
                                      result_dict[f"avg_{col}"] = window_df[col].mean()
                                      # Optional: Store std within run if needed, but not for Chi
                                      # result_dict[f"std_within_run_{col}"] = window_df[col].std()
                                  else:
                                      result_dict[f"avg_{col}"] = np.nan
                                      if col in NON_AVERAGE_REPORTERS: result_dict[f"{col}"] = None
                              else:
                                  result_dict[f"avg_{col}"] = np.nan
                                  if col in NON_AVERAGE_REPORTERS: result_dict[f"{col}"] = None
                      else: # No data collected
                          for col in model.datacollector.model_reporters.keys():
                              result_dict[f"avg_{col}"] = np.nan
                              if col in NON_AVERAGE_REPORTERS: result_dict[f"{col}"] = None

                      all_results.append(result_dict)

                 except Exception as e:
                      print(f"\nError running config {config.param_set_id} (run {config.run_id}): {e}")
                      import traceback
                      traceback.print_exc()
                 finally:
                      pbar.update(1)

        results_df = pd.DataFrame(all_results)

    # Save the raw run results
    if not results_df.empty:
        try:
            with open(data_filename, 'wb') as f:
                pickle.dump(results_df, f)
            print(f"Susceptibility raw run data saved to {data_filename}")
        except Exception as e:
            print(f"Error saving data to {data_filename}: {e}")
    else:
         print(f"Warning: No susceptibility results generated for {data_filename}.")

    return results_df


# ==============================================================================
# Focused Data Processing for Susceptibility
# ==============================================================================

def process_susceptibility_data(raw_df):
    """
    Calculates ONLY susceptibility (Chi) from raw run data.
    Assumes raw_df contains columns like 'L', 'b', 'avg_CooperationRate', etc. for EACH run.
    Groups by all simulation parameters except run-specific ones (seed, run_id).
    """
    if raw_df.empty:
        print("Warning: process_susceptibility_data received empty DataFrame.")
        return pd.DataFrame()

    processed_data = []

    # Determine grouping parameters (same logic as process_main_data)
    sim_config_fields = {f.name for f in fields(SimConfig)}
    parameter_cols = [col for col in raw_df.columns if col in sim_config_fields]
    run_identifiers = ['seed', 'run_id', 'param_set_id']
    # Identify reporter results (avg_, std_, or specific non-avg like ClusterSizeDistribution)
    reporter_result_cols = [col for col in raw_df.columns if col.startswith('avg_') or col.startswith('std_') or col == 'ClusterSizeDistribution' or col == 'runtime_seconds'] # Add others if they exist

    grouping_params = [col for col in parameter_cols if col not in run_identifiers and col not in reporter_result_cols]
    # Ensure L and b are included for grouping
    if 'L' not in grouping_params and 'L' in raw_df.columns: grouping_params.append('L')
    if 'b' not in grouping_params and 'b' in raw_df.columns: grouping_params.append('b')
    grouping_params = sorted(list(set(grouping_params)))

    print(f"Processing susceptibility data, grouping by: {grouping_params}")

    grouped = raw_df.groupby(grouping_params)
    print(f"Processing {len(grouped)} unique parameter sets for susceptibility...")

    for name, group in tqdm(grouped, desc="Calculating Chi"):
        processed_point = dict(zip(grouping_params, name)) # Store grouping params
        L = processed_point.get('L')
        if pd.notna(L) and L is not None: N = L * L
        else: N = np.nan

        # Calculate Susceptibility (Chi) = N * Var(<Reporter>)
        for reporter_name in SUS_PARAMS['susceptibility_target_names']:
             avg_col = f"avg_{reporter_name}" # Column containing the time-averaged value for each run
             if avg_col in group.columns and pd.notna(N) and N > 0:
                 # Variance of the time-averaged values ACROSS runs
                 variance_across_runs = group[avg_col].var()
                 chi_val = N * variance_across_runs if pd.notna(variance_across_runs) else np.nan
                 processed_point[f"chi_{reporter_name}"] = chi_val
             else:
                 processed_point[f"chi_{reporter_name}"] = np.nan
                 if avg_col not in group.columns:
                     print(f"Warning: Column '{avg_col}' not found for Chi calculation at {name}.")

        processed_data.append(processed_point)

    processed_df = pd.DataFrame(processed_data)
    print(f"Finished processing susceptibility. Found {len(processed_df)} processed points.")
    return processed_df


# ==============================================================================
# Plotting Function (Adapted from plot_figures.py)
# ==============================================================================

def plot_fig2_susceptibility_focused(data, chi_key, ylabel, filename_base):
    """
    Plots susceptibility vs b for different L values.
    Uses the data processed specifically for susceptibility.
    (Essentially the same plotting logic as plot_fig2_susceptibility).
    """
    print(f"Plotting Focused Susceptibility: {chi_key}...")
    if data.empty or chi_key not in data.columns:
        print(f"Error plotting {filename_base}: Data missing or '{chi_key}' column not found.")
        return pd.Series(dtype=float)

    # --- Filter data for fixed parameters (using SUS_PARAMS for consistency) ---
    fixed_params_to_plot = {
        'p_mut_culture': SUS_PARAMS.get('p_mut_culture', None),
        'p_mut_strategy': SUS_PARAMS.get('p_mut_strategy', None),
        # Add other fixed parameters from SUS_PARAMS if needed
        'K': SUS_PARAMS.get('K'),
        'K_C': SUS_PARAMS.get('K_C'),
    }
    fixed_params_to_plot = {k: v for k, v in fixed_params_to_plot.items() if v is not None and k in data.columns}

    filtered_data = filter_data_by_fixed_params(data, fixed_params_to_plot)
    if filtered_data.empty:
         print(f"Warning: No data found for plotting {filename_base} with fixed parameters: {fixed_params_to_plot}")
         print(f"Available fixed params in data: { {k: data[k].unique() for k in fixed_params_to_plot.keys()} }")
         return pd.Series(dtype=float)

    fixed_params_str = get_fixed_params_string(fixed_params_to_plot)
    # Add steps info to filename
    filename_base_with_params = f"{filename_base}_steps{SUS_PARAMS['sus_steps']}"
    filename_base_with_params += "_" + fixed_params_str.replace('(', '').replace(')', '').replace(', ', '_').replace('=', '').replace('.', 'p')

    fig, ax = plt.subplots(figsize=(6, 4))
    Ls = sorted(filtered_data['L'].unique())
    base_markersize = mpl.rcParams['lines.markersize']
    peak_locs = {}

    for i, L in enumerate(Ls):
        L_data = filtered_data[filtered_data['L'] == L].sort_values('b')
        L_data = L_data.replace([np.inf, -np.inf], np.nan).dropna(subset=[chi_key])
        if L_data.empty: continue

        style_kwargs = get_style_kwargs(i, len(Ls), base_markersize)
        label = f'$L={L}$'
        # Plot points and lines for susceptibility
        ax.plot(L_data['b'], L_data[chi_key], label=label, marker=style_kwargs['marker'], linestyle=style_kwargs['linestyle'], color=style_kwargs['color'], markersize=style_kwargs['markersize'])

        # Find peak location
        if not L_data.empty and chi_key in L_data.columns and not L_data[chi_key].dropna().empty:
             try:
                 peak_idx = L_data[chi_key].idxmax()
                 if pd.notna(peak_idx) and peak_idx in L_data.index:
                     peak_locs[L] = L_data.loc[peak_idx, 'b']
                 else:
                     peak_locs[L] = np.nan
             except ValueError: # Handles case where all values are NaN
                 peak_locs[L] = np.nan
        else:
             peak_locs[L] = np.nan

    ax.set_xlabel('Temptation ($b$)')
    ax.set_ylabel(ylabel)
    ax.set_title(f'Susceptibility vs. Temptation $b$\n{fixed_params_str} (Steps={SUS_PARAMS["sus_steps"]})')
    ax.legend(title='System Size $L$', loc='best', frameon=False)
    # ax.set_yscale('log') # Optional: Consider log scale
    ax.grid(False) # Keep Physica A style

    # Set sensible y-limits, e.g., start from 0
    ax.set_ylim(bottom=-0.005) # Slightly below 0 for visibility

    save_plot(fig, filename_base_with_params, plot_dir=SUS_PLOT_SAVE_DIR) # Save in specific dir
    plt.close(fig)
    print(f"Susceptibility Peak Locations ({chi_key}): {peak_locs}")
    valid_peaks = {L: peak_locs.get(L) for L in Ls if pd.notna(peak_locs.get(L))}
    return pd.Series(valid_peaks)


# ==============================================================================
# Main Execution Workflow
# ==============================================================================
if __name__ == "__main__":
    print("--- Starting Focused Susceptibility Analysis Workflow ---")
    overall_start_time = time.time()

    # --- Control Flag ---
    FORCE_RERUN_SUS = False # Set True to force regeneration of susceptibility data

    # --- 1. Run/Load Focused Susceptibility Scan ---
    print("\n--- Running/Loading Focused Susceptibility Scan (Longer Steps) ---")
    sus_raw_data = run_sus_simulation_batch(
        generate_sus_scan_configs,
        SUS_DATA_FILENAME,
        force_rerun=FORCE_RERUN_SUS
    )

    # --- 2. Process Susceptibility Data ---
    print("\n--- Processing Focused Susceptibility Data ---")
    sus_processed_data = process_susceptibility_data(sus_raw_data)

    # --- 3. Generate Plots ---
    print("\n--- Generating Focused Susceptibility Plots (Fig 2) ---")
    if not sus_processed_data.empty:
        # Plot chi_fc
        peaks_fc = plot_fig2_susceptibility_focused(sus_processed_data, 'chi_CooperationRate',
                                           'Susceptibility $\\chi_{f_C}$', "fig2a_susc_coop_rate_focused")
        # Plot chi_S
        peaks_s = plot_fig2_susceptibility_focused(sus_processed_data, 'chi_SegregationIndex',
                                          r'Susceptibility $\chi_S$', "fig2b_susc_segregation_focused")
    else:
        print("Skipping susceptibility plots as processed data is empty.")

    overall_end_time = time.time()
    print(f"\n--- Focused Susceptibility Workflow finished in {(overall_end_time - overall_start_time):.2f} seconds ---")
    print(f"Susceptibility Data saved in: '{SUS_DATA_SAVE_DIR}'")
    print(f"Susceptibility Plots saved in: '{SUS_PLOT_SAVE_DIR}'")

