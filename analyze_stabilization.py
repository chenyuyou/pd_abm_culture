# analyze_stabilization.py
import os
import time
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np
from tqdm import tqdm
import sys  # Import sys to modify path
import os  # Import os to get current path
import inspect  # Import inspect to check function signatures

# --- 确保可以导入核心模块和Mesa组件 ---
try:
    # Adjust sys.path to find core and utils directories
    # Assuming analyze_stabilization.py is in 'analysis' or similar,
    # and core/ and utils/ are one level up. Adjust if your structure is different.
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up one level to find 'core' and 'utils'
    parent_dir = os.path.join(current_dir, '..')
    # Add parent_dir to sys.path if it's not already there
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
        print(f"Added '{parent_dir}' to sys.path.")

    # Import Mesa components
    from mesa.datacollection import DataCollector

    # Import core modules
    from core.model import CulturalGame
    from utils.config import SimConfig
    # 导入必要的reporter函数，DataCollector需要它们
    # Ensure all reporters you want to monitor are imported and return scalar values
    from utils.reporters import (
        get_cooperation_rate,
        get_segregation_index,
        get_average_culture,
        get_std_culture,
        get_cooperation_rate_A,
        get_cooperation_rate_B,
        get_boundary_fraction,
        get_boundary_coop_rate,
        get_bulk_coop_rate
        # get_cluster_size_distribution is NOT suitable for plotting as a simple time series
    )
    MODULES_LOADED = True
    print("Core modules and Mesa components loaded successfully.")
except ImportError as e:
    print(f"Error importing core modules or Mesa components: {e}")
    print("Please ensure Mesa is installed (`pip install mesa`),")
    print("agent.py, model.py, config.py, reporters.py are accessible")
    print("and that the script's path allows finding 'core' and 'utils' directories.")
    MODULES_LOADED = False
    # Define dummy classes/functions if import fails to prevent NameErrors later

    class CulturalGame:
        def __init__(self, **kwargs):  # Dummy init to accept args
            print("Dummy CulturalGame initialized (Modules not loaded).")
            self.datacollector = DataCollector()  # Use dummy DataCollector

        def step(self): pass  # Dummy step method

    # Define dummy SimConfig class
    class SimConfig:
        __dataclass_fields__ = {}  # Dummy fields

        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
            print(
                f"Dummy SimConfig initialized with: {kwargs} (Modules not loaded).")

        def to_dict(self): return self.__dict__  # Basic dict conversion

    # Define a dummy DataCollector to prevent errors if Mesa import fails
    class DataCollector:
        def __init__(self, model_reporters=None):
            print("Dummy DataCollector initialized (Modules not loaded).")
            self._model_vars = {}
            self._agent_vars = {}
            self.model_reporters = model_reporters if model_reporters is not None else {}
            self.agent_reporters = {}  # Not used in this script, but for completeness
            self.steps = 0

        def collect(self, model):
            self.steps += 1
            for name, reporter in self.model_reporters.items():
                try:
                    value = reporter(model)
                    if name not in self._model_vars:
                        self._model_vars[name] = []
                    self._model_vars[name].append(value)
                except Exception as e:
                    print(f"Error collecting dummy reporter '{name}': {e}")
                    if name not in self._model_vars:
                        self._model_vars[name] = []
                    self._model_vars[name].append(
                        np.nan)  # Append NaN on error

        def get_model_vars_dataframe(self):
            if not self._model_vars:
                return pd.DataFrame()
            # Pad shorter lists with NaNs if needed (though collect should add per step)
            max_len = max((len(v)
                          for v in self._model_vars.values()), default=0)
            data = {k: v + [np.nan] * (max_len - len(v))
                    for k, v in self._model_vars.items()}
            df = pd.DataFrame(data)
            df.index.name = 'Step'
            return df

        # Dummy agent data collection method
        def get_agent_vars_dataframe(self): return pd.DataFrame()

    # Define dummy reporter functions returning NaN

    def get_cooperation_rate(model): return np.nan
    def get_segregation_index(model): return np.nan
    def get_average_culture(model): return np.nan
    def get_std_culture(model): return np.nan
    def get_cooperation_rate_A(model): return np.nan
    def get_cooperation_rate_B(model): return np.nan
    def get_boundary_fraction(model): return np.nan
    def get_boundary_coop_rate(model): return np.nan
    def get_bulk_coop_rate(model): return np.nan


# ==============================================================================
# 探索性模拟参数设置
# ==============================================================================
# --- 要测试的关键参数组合 ---
# 目标：找到最坏情况（最长）的弛豫时间
# 建议：
# 1. 选择最大的 L。
# 2. 选择 b 值：
#    - 远离相变点 (低 b 和高 b)。
#    - 接近相变点 (根据 Fig 2 感受态峰值附近的 b 值)。
#    - 至少包含感受态峰值最显著的 b 值。
# 3. 可以（可选地）包含一些不同 L 的临界点附近的 b 值，以观察 L 对弛豫时间的影响。
PARAMS_TO_TEST = [
    # --- 选择最大系统尺寸 L ---
    {'L': 50, 'b': 1.1},   # 最大 L, 远离相变点 (低 b)
    {'L': 50, 'b': 1.8},   # 最大 L, 接近相变点 (假设 b_c 约 1.8，请根据 Fig 2 更新)
    {'L': 50, 'b': 2.2},   # 最大 L, 接近相变点
    {'L': 50, 'b': 2.5},   # 最大 L, 接近相变点
    {'L': 50, 'b': 3.0},   # 最大 L, 接近相变点
    {'L': 50, 'b': 4.0},   # 最大 L, 远离相变点 (高 b)
    # --- （可选）测试较小 L 的临界点行为 ---
    # {'L': 20, 'b': 2.5},   # 较小 L, 临界点附近 (如果临界点随 L 变化)
    # {'L': 30, 'b': 2.5},   # 较小 L, 临界点附近
]

# --- 长时间模拟的总步数 ---
# **必须**远超预期的稳定时间。初步尝试 5000-10000 步，如果趋势在结束时仍不稳定，需要显著增加。
LONG_SIMULATION_STEPS = 5000       ## 推荐10000

# --- 用于平滑的移动平均窗口大小 ---
# 窗口大小应小于总步数，通常选择总步数的 1% - 5% 左右。
MOVING_AVG_WINDOW = 100         ## 推荐500

# --- 基础模型参数 (保持与主模拟一致) ---
# **!!! 重要 !!!** 确保这些参数与 plot_figures.py 中用于主扫描的固定参数一致。
# 特别是突变率 p_mut_culture 和 p_mut_strategy
BASE_PARAMS = {
    'initial_coop_ratio': 0.5,
    'K': 0.1,
    'K_C': 0.1,
    'p_update_C': 0.1,
    'p_mut_culture': 0.01,      # 新参数名，请与 config.py 和 model.py 核对
    'p_mut_strategy': 0.001,    # 新参数名，请与 config.py 和 model.py 核对
    'C_dist': 'bimodal',
    'mu': 0.5,
    'sigma': 0.1,
    # 'seed': None # 对于趋势观察，可以不固定seed，或每个参数点用不同seed
}

# --- 要监控和绘图的观测量 Key ---
# 这些 key 必须与 utils/reporters.py 中函数名（不含 get_）或 model_reporters 字典中的 key 一致
OBSERVABLES_TO_PLOT = [
    "CooperationRate",
    "SegregationIndex",
    "AverageCulture",
    "StdCulture",
    "CoopRate_A",  # Assuming get_cooperation_rate_A reporter exists
    "CoopRate_B",  # Assuming get_cooperation_rate_B reporter exists
    "BoundaryFraction",  # Assuming get_boundary_fraction reporter exists
    "BoundaryCoopRate",  # Assuming get_boundary_coop_rate reporter exists
    "BulkCoopRate"  # Assuming get_bulk_coop_rate reporter exists
    # "ClusterSizeDistribution" is NOT suitable for this type of plotting
]


# --- 输出目录 ---
STABILIZATION_PLOT_DIR = "plots_stabilization"
os.makedirs(STABILIZATION_PLOT_DIR, exist_ok=True)

# --- Matplotlib 样式 (与主脚本保持一致) ---
mpl.rcParams.update({
    'font.size': 10, 'axes.labelsize': 12, 'axes.titlesize': 14,
    'xtick.labelsize': 10, 'ytick.labelsize': 10, 'legend.fontsize': 9,
    'figure.dpi': 150,  # 屏幕查看，无需300dpi
    'savefig.format': 'png', 'savefig.bbox': 'tight',
    'lines.linewidth': 1.5, 'lines.markersize': 4,
    'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
    'mathtext.fontset': 'cm', 'axes.grid': True, 'grid.alpha': 0.5, 'grid.linestyle': '--'
})

# ==============================================================================
# 辅助函数
# ==============================================================================


def run_single_long_simulation(config: SimConfig, observables_to_collect: list) -> pd.DataFrame:
    """
    运行单次长时间模拟并收集每一步的数据。

    Args:
        config: SimConfig object for the simulation parameters.
        observables_to_collect: List of string keys for reporters to collect.

    Returns:
        DataFrame with time series data for the specified observables.
    """
    start_time = time.time()
    # Build a unique ID for this run based on key parameters
    # Use a more robust string formatting for the ID
    param_id_parts = [f"L{config.L}", f"b{config.b:.2f}".replace('.', 'p')]
    # Add relevant base parameters to ID if they deviate significantly or are key
    if hasattr(config, 'p_mut_culture'):
        param_id_parts.append(
            f"pmC{config.p_mut_culture:.4g}".replace('.', 'p'))
    if hasattr(config, 'p_mut_strategy'):
        param_id_parts.append(
            f"pmS{config.p_mut_strategy:.4g}".replace('.', 'p'))
    if hasattr(config, 'K_C'):
        param_id_parts.append(f"KC{config.K_C:.4g}".replace('.', 'p'))
    run_id_str = "_".join(param_id_parts)

    print(f"\nProcessing parameters: {run_id_str}")
    print(f"  Running simulation for {config.steps} steps...")

    # --- 关键：设置DataCollector收集模型级别的*时间序列*数据 ---
    # Only include reporters requested and available in utils.reporters
    available_reporters = {
        "CooperationRate": get_cooperation_rate,
        "SegregationIndex": get_segregation_index,
        "AverageCulture": get_average_culture,
        "StdCulture": get_std_culture,
        "CoopRate_A": get_cooperation_rate_A,
        "CoopRate_B": get_cooperation_rate_B,
        "BoundaryFraction": get_boundary_fraction,
        "BoundaryCoopRate": get_boundary_coop_rate,
        "BulkCoopRate": get_bulk_coop_rate
    }
    model_reporters = {key: func for key, func in available_reporters.items(
    ) if key in observables_to_collect}

    if not model_reporters:
        print("  Warning: No valid reporters to collect. Skipping simulation.")
        return pd.DataFrame()

    try:
        # Create a dictionary containing only the parameters accepted by CulturalGame.__init__
        # Using inspect is robust against changes in __init__ signature
        sig = inspect.signature(CulturalGame.__init__)
        # Exclude 'self' and any variable keyword arguments (**kwargs) if present
        model_init_param_names = [p.name for p in sig.parameters.values(
        ) if p.name != 'self' and p.kind != inspect.Parameter.VAR_KEYWORD]

        # Get parameters from config object and filter based on __init__ signature
        config_dict = config.to_dict() if hasattr(
            config, 'to_dict') else config.__dict__  # Use to_dict if available
        model_init_params = {
            k: config_dict[k] for k in model_init_param_names if k in config_dict}

        # Initialize the model with the collected parameters
        model = CulturalGame(**model_init_params)

        # Initialize a new DataCollector instance for this model run
        # Overwriting the default one created by the model's __init__
        model.datacollector = DataCollector(model_reporters=model_reporters)

        # --- 运行模拟，每步收集数据 ---
        # The collect method is usually called implicitly by Mesa's run_model,
        # but here we run step manually and call collect at each step.
        for i in tqdm(range(config.steps), desc=f"  Sim {run_id_str}", leave=False):
            model.step()
            # Explicitly collect data after each step
            model.datacollector.collect(model)

        results_df = model.datacollector.get_model_vars_dataframe()
        end_time = time.time()
        print(
            f"  Finished simulation. Time: {end_time - start_time:.2f}s. Got {len(results_df)} steps.")
        return results_df

    except Exception as e:
        print(f"  Error during simulation {run_id_str}: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()  # 返回空DataFrame表示失败


def plot_stabilization_trends(df: pd.DataFrame, L: int, b: float, param_set_id: str, observable_key: str, moving_avg_window: int, save_dir: str):
    """
    绘制单个观测量的时间序列和移动平均线。
    """
    if df.empty or observable_key not in df.columns:
        # print(f"  Skipping plot for {observable_key} ({param_set_id}): Data missing or column not found.")
        return

    # --- 计算移动平均 ---
    # Use min_periods=1 to avoid NaNs at the beginning
    moving_avg = df[observable_key].rolling(
        window=moving_avg_window, center=True, min_periods=1).mean()

    # --- 绘图 ---
    fig, ax = plt.subplots(figsize=(8, 4))

    # 绘制原始数据 (半透明)
    ax.plot(df.index, df[observable_key], label='Raw Data',
            color='grey', alpha=0.5, linewidth=1)

    # 绘制移动平均线
    ax.plot(df.index, moving_avg,
            label=f'Moving Avg (w={moving_avg_window})', color='red', linewidth=1.5)

    # --- 设置图表元素 ---
    ax.set_xlabel('Time Step (t)')
    # Format ylabel for readability (e.g., "CooperationRate" -> "Cooperation Rate")
    # Remove "get_" prefix if present and add spaces before capital letters (except first)
    ylabel_formatted = observable_key
    if ylabel_formatted.startswith('get_'):
        ylabel_formatted = ylabel_formatted[4:]
    ylabel_formatted = ''.join(
        [' ' + char if char.isupper() else char for char in ylabel_formatted]).strip().title()

    ax.set_ylabel(ylabel_formatted)
    ax.set_title(
        f'Stabilization Analysis: {ylabel_formatted} ($L={L}, b={b:.2f}$)')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.6, linestyle=':')
    # 可选：设置 Y 轴范围
    if "Rate" in observable_key or "Index" in observable_key or "Fraction" in observable_key:
        # Rates, indices, fractions are usually between 0 and 1
        ax.set_ylim(-0.05, 1.05)
    if observable_key == "StdCulture":
        ax.set_ylim(bottom=0)  # Std dev is non-negative
    if observable_key == "AverageCulture":
        ax.set_ylim(-0.05, 1.05)  # Culture is [0, 1]

    # --- 保存图像 ---
    plot_filename_base = f"stabilization_{observable_key}_{param_set_id}".replace(
        '.', 'p').replace('-', '_')  # Replace problematic chars
    plot_filename = os.path.join(save_dir, f"{plot_filename_base}.png")
    try:
        fig.savefig(plot_filename)
    except Exception as e:
        print(f"  Error saving plot {plot_filename}: {e}")
    plt.close(fig)
    # print(f"  Plot saved: {plot_filename}")


# ==============================================================================
# 主执行流程
# ==============================================================================
if __name__ == "__main__":
    if not MODULES_LOADED:
        print("Core modules failed to load. Exiting.")
        sys.exit(1)  # Exit with an error code

    print("--- Starting Stabilization Analysis ---")
    print(f"Output directory: {STABILIZATION_PLOT_DIR}")
    print(f"Testing {len(PARAMS_TO_TEST)} parameter combinations.")
    print(f"Simulation steps per run: {LONG_SIMULATION_STEPS}")
    print(f"Moving average window: {MOVING_AVG_WINDOW}")
    print(
        f"\nBase parameters used (ensure these match your main simulation fixed parameters):\n{BASE_PARAMS}")
    print(f"\nObservables to plot:\n{OBSERVABLES_TO_PLOT}")

    overall_start = time.time()

    # Generate a unique base seed for this run
    seed_base = int(time.time() * 1000) % (2**32 - 1)

    for i, params in enumerate(PARAMS_TO_TEST):
        # --- 构建当前运行的 SimConfig ---
        current_config_dict = BASE_PARAMS.copy()
        # Add L and b etc. specific test parameters
        current_config_dict.update(params)
        current_config_dict['steps'] = LONG_SIMULATION_STEPS
        # Assign a unique seed for this specific (L, b, etc.) combination based on base seed
        current_config_dict['seed'] = seed_base + i
        # Generate a param_set_id based on the specific parameters being tested for filename
        param_id_parts = [
            f"L{current_config_dict.get('L', 'N/A')}", f"b{current_config_dict.get('b', 'N/A'):.2f}".replace('.', 'p')]
        # Include relevant base parameters in ID for clarity if they are not default Mesa values
        if 'p_mut_culture' in current_config_dict:
            param_id_parts.append(
                f"pmC{current_config_dict['p_mut_culture']:.4g}".replace('.', 'p'))
        if 'p_mut_strategy' in current_config_dict:
            param_id_parts.append(
                f"pmS{current_config_dict['p_mut_strategy']:.4g}".replace('.', 'p'))
        if 'K_C' in current_config_dict:
            param_id_parts.append(
                f"KC{current_config_dict['K_C']:.4g}".replace('.', 'p'))
        # Add other key parameters if necessary for differentiation
        # e.g., if K or p_update_C are varied in BASE_PARAMS
        if 'K' in current_config_dict and current_config_dict['K'] != BASE_PARAMS['K']:
            param_id_parts.append(
                f"K{current_config_dict['K']:.4g}".replace('.', 'p'))
        if 'p_update_C' in current_config_dict and current_config_dict['p_update_C'] != BASE_PARAMS['p_update_C']:
            param_id_parts.append(
                f"pUC{current_config_dict['p_update_C']:.4g}".replace('.', 'p'))

        param_set_id = "_".join(param_id_parts)
        # Store in config for run_single_long_simulation
        current_config_dict['param_set_id'] = param_set_id
        current_config_dict['run_id'] = 0  # Use run_id=0 for this analysis

        # Validate and create SimConfig
        valid_config_params = {}
        # Safely get SimConfig fields, assuming it's a dataclass
        sim_config_fields = set()
        if hasattr(SimConfig, '__dataclass_fields__'):
            sim_config_fields = set(SimConfig.__dataclass_fields__.keys())
        # Fallback for non-dataclass or if __dataclass_fields__ is missing
        elif hasattr(SimConfig, '__init__'):
            sig = inspect.signature(SimConfig.__init__)
            sim_config_fields = {p.name for p in sig.parameters.values(
            ) if p.name != 'self' and p.kind != inspect.Parameter.VAR_KEYWORD}

        for k, v in current_config_dict.items():
            if k in sim_config_fields:
                valid_config_params[k] = v
            else:
                # print(f"  Warning: Parameter '{k}' from test config is not a field in SimConfig and will be ignored.")
                pass  # Suppress warning for brevity unless debugging

        try:
            # Attempt to create SimConfig with validated parameters
            config = SimConfig(**valid_config_params)
            # Ensure all expected parameters for the model are present in the config object
            # This is a sanity check after creating SimConfig
            required_model_params = {p.name for p in inspect.signature(CulturalGame.__init__).parameters.values(
            ) if p.name != 'self' and p.kind != inspect.Parameter.VAR_KEYWORD}
            if not all(hasattr(config, p) for p in required_model_params):
                missing = [
                    p for p in required_model_params if not hasattr(config, p)]
                raise ValueError(
                    f"SimConfig object is missing parameters required by CulturalGame.__init__: {missing}")

            # Set the generated param_set_id on the config object for easy access
            setattr(config, 'param_set_id', param_set_id)

        except (TypeError, ValueError) as e:
            print(f"  Error creating SimConfig for parameters {params}: {e}")
            print(f"  Attempted parameters: {valid_config_params}")
            continue  # Skip this parameter combination

        # --- 运行长时模拟 ---
        simulation_df = run_single_long_simulation(config, OBSERVABLES_TO_PLOT)

        # --- 如果模拟成功，绘制图表 ---
        if not simulation_df.empty:
            for observable_key in OBSERVABLES_TO_PLOT:
                # Only plot if the column exists in the collected data
                if observable_key in simulation_df.columns:
                    # Pass param_set_id to plotting function for filename
                    plot_stabilization_trends(simulation_df, config.L, config.b, config.param_set_id,
                                              observable_key, MOVING_AVG_WINDOW, STABILIZATION_PLOT_DIR)
                # else:
                #      print(f"  Warning: Observable '{observable_key}' not found in collected data for {config.param_set_id}.")

        # Add a small delay to ensure unique seeds if using time-based seeding (mostly for base_seed)
        time.sleep(0.05)

    overall_end = time.time()
    print("\n--- Stabilization Analysis Finished ---")
    print(f"Total time: {overall_end - overall_start:.2f} seconds.")
    print(f"Plots saved in '{STABILIZATION_PLOT_DIR}'.")
    print("\n**NEXT STEP:** Manually inspect the generated plots.")
    print("For each observable and parameter combination, find the time step where the RED moving average line becomes FLAT.")
    print("The LATEST time point across ALL plots is your estimated 'transient_steps'.")
    print("Add a safety margin to this latest time to determine the 'steps' parameter for your main simulations.")
    print("Make sure the 'steady_state_window' in your main simulation is within this flat region.")
