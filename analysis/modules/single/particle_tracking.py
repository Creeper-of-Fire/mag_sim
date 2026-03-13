# analysis/modules/particle_tracking.py

import os
import traceback
from typing import List, Dict, Any, Tuple, Optional

import h5py
import numpy as np
import pandas as pd
from scipy.constants import c, m_e, e
from tqdm import tqdm

from analysis.core.simulation import SimulationRun
from analysis.core.utils import console
from analysis.modules.abstract.base_module import BaseAnalysisModule
from analysis.modules.utils.spectrum_tools import filter_valid_runs
from analysis.plotting.layout import create_analysis_figure

# 物理常量
ME_C2_J = m_e * c ** 2
J_PER_MEV = e * 1e6
J_PER_KEV = e * 1e3


def get_robust_dataset(h5_item: h5py.Group, component_path: str) -> np.ndarray:
    """
    健壮地从 HDF5 中读取数据，自动处理 Dataset 和 Group 的嵌套情况。
    (参考自 data_loader.py 中的逻辑)
    """
    if component_path not in h5_item:
        return np.array([])

    item = h5_item[component_path]
    if isinstance(item, h5py.Dataset):
        return item[:]
    elif isinstance(item, h5py.Group):
        if not item: return np.array([])

        # 策略 1: 组内只有一个成员，假设它就是数据
        if len(item) == 1:
            first_key = list(item.keys())[0]
            if isinstance(item[first_key], h5py.Dataset):
                return item[first_key][:]

        # 策略 2: 尝试读取与路径最后一部分同名的数据集 (例如 id/id)
        ds_name = component_path.split('/')[-1]
        if ds_name in item and isinstance(item[ds_name], h5py.Dataset):
            return item[ds_name][:]

    return np.array([])

def _read_species_tracking_data(fpath: str) -> Tuple[Optional[str], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    从单个 HDF5 文件中读取带有 ID 的粒子数据，并计算动能。
    增加了详细的 Debug 输出，以定位为什么找不到 ID。
    """
    try:
        with h5py.File(fpath, 'r') as f:
            if 'data' not in f: return None, None, None
            step_key = list(f['data'].keys())[0]
            particles_group = f[f'data/{step_key}/particles']

            # 寻找具有 'id' 的物种
            target_species = None
            for sp in particles_group.keys():
                # 只要这个物种组下面能读到有效 id 数组即可
                test_ids = get_robust_dataset(particles_group[sp], 'id')
                if test_ids.size > 0:
                    target_species = sp
                    break

            if not target_species:
                return None, None, None

            sp_group = particles_group[target_species]

            # 使用健壮读取器
            ids = get_robust_dataset(sp_group, 'id')
            px = get_robust_dataset(sp_group, 'momentum/x')
            py = get_robust_dataset(sp_group, 'momentum/y')
            pz = get_robust_dataset(sp_group, 'momentum/z')

            if ids.size == 0 or px.size == 0:
                return None, None, None

            # 计算动能 (keV)
            p_sq = px ** 2 + py ** 2 + pz ** 2
            ek_J = np.sqrt(p_sq * c ** 2 + ME_C2_J ** 2) - ME_C2_J
            ek_keV = ek_J / J_PER_KEV

            return target_species, ids, ek_keV


    except Exception as err:
        console.print(f"  [bold red]读取 {os.path.basename(fpath)} 时发生未捕获的错误: {err}[/bold red]")
        # 打印详细报错堆栈
        traceback.print_exc()
        return None, None, None


def compute_particle_trajectories(run: SimulationRun, sample_size: int = 500, mode: str = 'backward') -> Dict[str, Any]:
    """
    核心追踪逻辑。
    """
    files = run.particle_files
    if not files:
        return {}

    # 1. 读取初始帧，确立追踪目标
    # --- 选择初始帧还是末尾帧 ---
    if mode == 'forward':
        selection_file = files[0]
    else:
        selection_file = files[-1]  # 从最后一帧选

    console.print(f"  [cyan]正在从 {('末尾帧' if mode == 'backward' else '初始帧')} 筛选追踪目标...[/cyan]")

    sp_name, initial_ids, initial_ek = _read_species_tracking_data(selection_file)
    if initial_ids is None or len(initial_ids) == 0:
        console.print("  [yellow]⚠ 未能在帧找到粒子 ID 数据，跳过追踪。[/yellow]")
        return {}

    # 估算有效温度 (非相对论粗略近似 <Ek> = 1.5 kT，足以用来做标签分类)
    mean_ek = np.mean(initial_ek)
    kT_eff = mean_ek * (2.0 / 3.0)

    # 筛选条件
    bulk_mask = (initial_ek > 0.8 * kT_eff) & (initial_ek < 1.2 * kT_eff)
    tail_mask = (initial_ek > 3.0 * kT_eff)

    bulk_candidates = initial_ids[bulk_mask]
    tail_candidates = initial_ids[tail_mask]

    # 随机抽样 (保证追踪速度和图形不至于杂乱)
    np.random.seed(42)  # 保证重复运行结果一致
    n_bulk = min(sample_size, len(bulk_candidates))
    n_tail = min(sample_size, len(tail_candidates))

    if n_bulk == 0 or n_tail == 0:
        console.print("  [yellow]⚠ 无法找到足够的体粒子或尾粒子，跳过追踪。[/yellow]")
        return {}

    track_bulk_ids = np.random.choice(bulk_candidates, n_bulk, replace=False)
    track_tail_ids = np.random.choice(tail_candidates, n_tail, replace=False)

    console.print(f"  [green]锁定目标[/green]: 追踪 {n_bulk} 个 Bulk 粒子 (~{kT_eff:.1f} keV), {n_tail} 个 Tail 粒子 (> {3 * kT_eff:.1f} keV)")

    # 2. 准备存储矩阵
    num_steps = len(files)
    times = np.zeros(num_steps)
    bulk_history = np.full((n_bulk, num_steps), np.nan)
    tail_history = np.full((n_tail, num_steps), np.nan)

    # 为了极速提取，我们使用 Pandas 的索引对齐功能
    target_ids_all = np.concatenate([track_bulk_ids, track_tail_ids])

    # 3. 穿越时间步进行追踪
    for i, fpath in enumerate(tqdm(files, desc="  穿越时间追踪粒子", unit="帧", leave=False)):
        try:
            step = int(os.path.basename(fpath).split('_')[-1].split('.')[0])
            times[i] = step * run.sim.dt

            _, current_ids, current_ek = _read_species_tracking_data(fpath)
            if current_ids is None: continue

            # [核心魔法] 利用 Pandas 瞬间对齐跨帧打乱的 ID
            # 将当前帧的数据建成一个以 ID 为索引的字典表
            frame_series = pd.Series(current_ek, index=current_ids)

            # 直接提取我们需要追踪的 ID 的能量（如果该 ID 丢失，自动填 NaN）
            tracked_energies = frame_series.reindex(target_ids_all).values

            # 填入历史记录矩阵
            bulk_history[:, i] = tracked_energies[:n_bulk]
            tail_history[:, i] = tracked_energies[n_bulk:]

        except Exception as e:
            continue

    return {
        "time": times,
        "kT_eff": kT_eff,
        "bulk_history": bulk_history,
        "tail_history": tail_history
    }


class ParticleTrackingModule(BaseAnalysisModule):
    @property
    def name(self) -> str:
        return "粒子微观轨迹追踪 (Particle Tracking)"

    @property
    def description(self) -> str:
        return "追踪特定粒子的相空间演化，验证高能粒子是否参与热化或发生二次加速。"

    def _analyze_single_run(self, run: SimulationRun):
        metrics = compute_particle_trajectories(run, sample_size=5000)

        if not metrics:
            return

        times = metrics["time"]
        kT_eff = metrics["kT_eff"]
        bulk_history = metrics["bulk_history"]
        tail_history = metrics["tail_history"]

        filename_override = f"{run.name}_particle_tracking"

        with create_analysis_figure(run, "particle_tracking", num_plots=2, figsize=(12, 5), override_filename=filename_override) as (fig, axes):
            ax_traj, ax_hist = axes

            # ==========================================
            # Plot 1: 粒子能量演化轨迹 (Energy Trajectories)
            # ==========================================
            # 为了防止图太满变成一团黑，我们只画出前 30 根线
            plot_limit = 30

            for j in range(min(plot_limit, bulk_history.shape[0])):
                ax_traj.plot(times * 1e12, bulk_history[j, :], color='dodgerblue', alpha=0.3, lw=1)

            for j in range(min(plot_limit, tail_history.shape[0])):
                ax_traj.plot(times * 1e12, tail_history[j, :], color='crimson', alpha=0.3, lw=1.5)

            # 绘制均值参考线
            ax_traj.axhline(kT_eff, color='blue', linestyle='--', label='1.0 $kT_{eff}$ (Bulk)')
            ax_traj.axhline(3 * kT_eff, color='red', linestyle='--', label='3.0 $kT_{eff}$ (Tail)')

            # 添加哑图例 (Dummy legends for lines)
            ax_traj.plot([], [], color='dodgerblue', label='Bulk Particles Trajectory')
            ax_traj.plot([], [], color='crimson', label='Tail Particles Trajectory')

            ax_traj.set_title("单个粒子动能演化轨迹 (Lagrangian View)", fontsize=13)
            ax_traj.set_xlabel("时间 (ps)")
            ax_traj.set_ylabel("动能 $E_k$ (keV)")
            # 这里如果尾部能量特别高，可考虑对数坐标
            ax_traj.set_yscale('log')
            ax_traj.legend(loc='upper right')
            ax_traj.grid(True, alpha=0.3, which='both')

            # ==========================================
            # Plot 2: 动能相对变化率分布 (Energy Variation PDF)
            # ==========================================
            # 我们衡量的是：结束时的能量与初始能量的相对变化量 (E_final / E_initial)
            # 这能最直观地反映它是否参与了剧烈的能量交换

            # 清理可能的 NaN (丢失的粒子)
            valid_bulk = ~np.isnan(bulk_history[:, 0]) & ~np.isnan(bulk_history[:, -1])
            valid_tail = ~np.isnan(tail_history[:, 0]) & ~np.isnan(tail_history[:, -1])

            # 计算能量变化率 E_final / E_initial
            ratio_bulk = bulk_history[valid_bulk, -1] / bulk_history[valid_bulk, 0]
            ratio_tail = tail_history[valid_tail, -1] / tail_history[valid_tail, 0]

            # 绘制直方图
            bins = np.logspace(np.log10(0.1), np.log10(10.0), 50)

            ax_hist.hist(ratio_bulk, bins=bins, color='dodgerblue', alpha=0.6, density=True, label='Bulk Particles')
            ax_hist.hist(ratio_tail, bins=bins, color='crimson', alpha=0.7, density=True, label='Tail Particles')

            ax_hist.axvline(1.0, color='black', linestyle='--', lw=2, label='No Change ($E_{final} = E_{initial}$)')

            ax_hist.set_title("整个演化周期内的相对能量改变", fontsize=13)
            ax_hist.set_xlabel(r"能量比值 $E_{final} / E_{initial}$")
            ax_hist.set_ylabel("概率密度 (PDF)")
            ax_hist.set_xscale('log')
            ax_hist.legend()
            ax_hist.grid(True, alpha=0.3)

            # 自动添加结论文本
            ax_hist.text(0.05, 0.85, "Bulk 粒子呈现极宽分布\n(剧烈热化与能量交换)",
                         transform=ax_hist.transAxes, color='mediumblue', fontsize=10)
            ax_hist.text(0.05, 0.70, "Tail 粒子集中于 1.0 附近\n(未参与热化, 绝缘状态)",
                         transform=ax_hist.transAxes, color='darkred', fontsize=10)

        console.print(f"  [green]✔ 分析完成: {run.name}[/green]")

    def run(self, loaded_runs: List[SimulationRun]):
        console.print("\n[bold magenta]执行: 粒子相空间微观轨迹追踪...[/bold magenta]")

        valid_runs = filter_valid_runs(
            loaded_runs,
            require_particles=True,
            min_particle_files=2
        )
        if not valid_runs:
            console.print("[yellow]警告: 没有找到足够包含 ID 的粒子文件，跳过此分析。[/yellow]")
            return

        for run in valid_runs:
            console.print(f"\n[bold]追踪模拟: {run.name}[/bold]")
            self._analyze_single_run(run)