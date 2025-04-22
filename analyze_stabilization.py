# analyze_stabilization.py
import os
import time
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np
from tqdm import tqdm

# --- 确保可以导入核心模块 ---
try:
    from core.model import CulturalGame
    from utils.config import SimConfig
    # 导入必要的reporter函数，DataCollector需要它们
    from utils.reporters import get_cooperation_rate, get_segregation_index
    MODULES_LOADED = True
except ImportError as e:
    print(f"Error importing core modules: {e}")
    print("Please ensure agent.py, model.py, config.py, reporters.py are accessible.")
    MODULES_LOADED = False
    # Define dummy functions if import fails, so script structure works
    class CulturalGame: pass
    class SimConfig: pass
    def get_cooperation_rate(model): return np.nan
    def get_segregation_index(model): return np.nan


# ==============================================================================
# 探索性模拟参数设置
# ==============================================================================
# --- 要测试的关键参数组合 ---
# 目标：找到最坏情况（最长）的弛豫时间
PARAMS_TO_TEST = [
    # --- 选择最大系统尺寸 L ---
    {'L': 50, 'b': 1.1},   # 最大 L, 远离相变点 (低 b)
    {'L': 50, 'b': 1.7},   # 最大 L, 接近相变点 (假设 b_c 约 1.8)
    {'L': 50, 'b': 1.8},   # 最大 L, 接近相变点
    {'L': 50, 'b': 1.9},   # 最大 L, 接近相变点
    {'L': 50, 'b': 2.4},   # 最大 L, 远离相变点 (高 b)
    # --- （可选）测试较小 L 的临界点行为 ---
    # {'L': 20, 'b': 1.8},   # 较小 L, 接近相变点
]

# --- 长时间模拟的总步数 ---
# **必须**远超预期的稳定时间
LONG_SIMULATION_STEPS = 5000 # 初始值，根据输出图调整，可能需要10000或更多

# --- 用于平滑的移动平均窗口大小 ---
MOVING_AVG_WINDOW = 100 # 可以根据需要调整

# --- 基础模型参数 (保持与主模拟一致) ---
BASE_PARAMS = {
    'initial_coop_ratio': 0.5,
    'K': 0.1,
    'K_C': 0.1,
    'p_update_C': 0.1,
    'p_mut': 0.001,
    'C_dist': 'bimodal',
    'mu': 0.5,
    'sigma': 0.1,
    # 'seed': None # 对于趋势观察，可以不固定seed，或每个参数点用不同seed
}

# --- 输出目录 ---
STABILIZATION_PLOT_DIR = "plots_stabilization"
os.makedirs(STABILIZATION_PLOT_DIR, exist_ok=True)

# --- Matplotlib 样式 (与主脚本保持一致) ---
mpl.rcParams.update({
    'font.size': 10, 'axes.labelsize': 12, 'axes.titlesize': 14,
    'xtick.labelsize': 10, 'ytick.labelsize': 10, 'legend.fontsize': 9,
    'figure.dpi': 150, # 屏幕查看，无需300dpi
    'savefig.format': 'png', 'savefig.bbox': 'tight',
    'lines.linewidth': 1.5, 'lines.markersize': 4,
    'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
    'mathtext.fontset': 'cm', 'axes.grid': True, 'grid.alpha': 0.5, 'grid.linestyle': '--'
})

# ==============================================================================
# 辅助函数
# ==============================================================================

def run_single_long_simulation(config: SimConfig) -> pd.DataFrame:
    """
    运行单次长时间模拟并收集每一步的数据。
    """
    start_time = time.time()
    print(f"  Running L={config.L}, b={config.b} for {config.steps} steps...")

    # --- 关键：设置DataCollector收集模型级别的*时间序列*数据 ---
    model_reporters = {
        "CooperationRate": get_cooperation_rate,
        "SegregationIndex": get_segregation_index,
    }

    try:
        # --- *** 修改部分开始 *** ---
        # 创建一个只包含 CulturalGame.__init__ 所需参数的字典
        model_init_params = {
            'L': config.L,
            'initial_coop_ratio': config.initial_coop_ratio,
            'b': config.b,
            'K': config.K,
            'C_dist': config.C_dist,
            'mu': config.mu,
            'sigma': config.sigma,
            'seed': config.seed,
            'K_C': config.K_C,
            'p_update_C': config.p_update_C,
            'p_mut': config.p_mut
            # 确保这里包含了 CulturalGame.__init__ 定义的所有参数
            # 不要包含 'steps', 'param_set_id', 'run_id' 等非初始化参数
        }
        # 使用筛选后的参数初始化模型
        model = CulturalGame(**model_init_params)
        # --- *** 修改部分结束 *** ---

        # 覆盖 datacollector 以确保收集每一步数据 (这部分保持不变)
        model.datacollector = model.datacollector.__class__(model_reporters=model_reporters)

        # --- 运行模拟，每步收集数据 (这部分保持不变) ---
        for i in tqdm(range(config.steps), desc=f"  Sim L={config.L}, b={config.b}", leave=False):
            model.step()

        results_df = model.datacollector.get_model_vars_dataframe()
        end_time = time.time()
        print(f"  Finished L={config.L}, b={config.b}. Time: {end_time - start_time:.2f}s. Got {len(results_df)} steps.")
        return results_df

    except Exception as e:
        print(f"  Error during simulation L={config.L}, b={config.b}: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame() # 返回空DataFrame表示失败



def plot_stabilization_trends(df: pd.DataFrame, L: int, b: float, observable_key: str, moving_avg_window: int, save_dir: str):
    """
    绘制单个观测量的时间序列和移动平均线。
    """
    if df.empty or observable_key not in df.columns:
        print(f"  Skipping plot for {observable_key} (L={L}, b={b}): Data missing.")
        return

    # --- 计算移动平均 ---
    moving_avg = df[observable_key].rolling(window=moving_avg_window, center=True, min_periods=1).mean()

    # --- 绘图 ---
    fig, ax = plt.subplots(figsize=(8, 4))

    # 绘制原始数据 (半透明)
    ax.plot(df.index, df[observable_key], label='Raw Data', color='grey', alpha=0.5, linewidth=1)

    # 绘制移动平均线
    ax.plot(df.index, moving_avg, label=f'Moving Avg (w={moving_avg_window})', color='red', linewidth=1.5)

    # --- 设置图表元素 ---
    ax.set_xlabel('Time Step (t)')
    ax.set_ylabel(observable_key)
    ax.set_title(f'Stabilization Analysis: {observable_key} (L={L}, b={b:.2f})')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.6, linestyle=':')
    # 可选：设置 Y 轴范围，例如合作率在 [0, 1]
    if "Rate" in observable_key or "Index" in observable_key:
        ax.set_ylim(-0.05, 1.05)

    # --- 保存图像 ---
    plot_filename = os.path.join(save_dir, f"stabilization_L{L}_b{b:.2f}_{observable_key}.png")
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
        exit()

    print("--- Starting Stabilization Analysis ---")
    print(f"Output directory: {STABILIZATION_PLOT_DIR}")
    print(f"Testing {len(PARAMS_TO_TEST)} parameter combinations.")
    print(f"Simulation steps per run: {LONG_SIMULATION_STEPS}")
    print(f"Moving average window: {MOVING_AVG_WINDOW}")

    overall_start = time.time()

    for params in PARAMS_TO_TEST:
        print(f"\nProcessing parameters: L={params['L']}, b={params['b']}")

        # --- 构建当前运行的 SimConfig ---
        current_config_dict = BASE_PARAMS.copy()
        current_config_dict.update(params) # 加入 L 和 b
        current_config_dict['steps'] = LONG_SIMULATION_STEPS
        current_config_dict['seed'] = int(time.time() * 1000 + params['L'] * 10 + params['b'] * 100) % (2**32 - 1) # 简单生成伪随机种子
        current_config_dict['param_set_id'] = f"stabilization_L{params['L']}_b{params['b']:.2f}"
        current_config_dict['run_id'] = 0

        # 过滤掉 SimConfig 不接受的参数 (如果BASE_PARAMS包含非模型参数)
        valid_keys = SimConfig.__dataclass_fields__.keys()
        filtered_config_dict = {k: v for k, v in current_config_dict.items() if k in valid_keys}

        try:
            config = SimConfig(**filtered_config_dict)
        except TypeError as e:
            print(f"Error creating SimConfig for L={params['L']}, b={params['b']}: {e}")
            print(f"Provided dict: {filtered_config_dict}")
            continue # 跳过这个参数组合

        # --- 运行长时模拟 ---
        simulation_df = run_single_long_simulation(config)

        # --- 如果模拟成功，绘制图表 ---
        if not simulation_df.empty:
            plot_stabilization_trends(simulation_df, config.L, config.b,
                                      "CooperationRate", MOVING_AVG_WINDOW, STABILIZATION_PLOT_DIR)
            plot_stabilization_trends(simulation_df, config.L, config.b,
                                      "SegregationIndex", MOVING_AVG_WINDOW, STABILIZATION_PLOT_DIR)
            # 在这里可以为其他需要监控的观测量添加绘图调用
            # plot_stabilization_trends(simulation_df, config.L, config.b, "AverageCulture", MOVING_AVG_WINDOW, STABILIZATION_PLOT_DIR)

    overall_end = time.time()
    print("\n--- Stabilization Analysis Finished ---")
    print(f"Total time: {overall_end - overall_start:.2f} seconds.")
    print(f"Plots saved in '{STABILIZATION_PLOT_DIR}'.")
    print("\n**NEXT STEP:** Manually inspect the generated plots.")
    print("Look for the LATEST time point across ALL plots where the RED moving average line becomes FLAT.")
    print("Add a safety margin to this latest time to determine the 'transient_steps' for your main simulations.")

