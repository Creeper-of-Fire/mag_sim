# analysis/modules/spectrum_evolution.py

import gc
import os
from typing import List, Dict, Optional

import numpy as np
from scipy.constants import c, m_e, e
from tqdm import tqdm

from analysis.core.simulation import SimulationRun, SpectrumData
from analysis.core.utils import console
from analysis.modules.abstract.base_module import BaseAnalysisModule
from analysis.modules.utils import physics_mj
from analysis.modules.utils.spectrum_tools import filter_valid_runs
from analysis.plotting.layout import create_analysis_figure

# --- 物理常量 ---
ME_C2_J = m_e * c ** 2
J_PER_MEV = e * 1e6
J_TO_KEV = 1.0 / (e * 1e3)


def _analyze_single_spectrum(spec: SpectrumData) -> Dict[str, float]:
    """对单个时刻的能谱进行分析，返回关键指标。"""
    if spec is None or spec.weights.size == 0:
        return {'T_keV': 0.0, 'excess_ratio': 0.0, 'max_E_MeV': 0.0}

    total_energy_MeV = np.sum(spec.energies_MeV * spec.weights)
    total_weight = np.sum(spec.weights)
    avg_energy_MeV = total_energy_MeV / total_weight
    T_keV = physics_mj.solve_mj_temperature_kev(avg_energy_MeV)

    pos_energies = spec.energies_MeV[spec.energies_MeV > 0]
    max_E_MeV = pos_energies.max() if pos_energies.size > 0 else 0.0

    # --- 计算非热能量 ---
    min_e = max(1e-4, pos_energies.min()) if pos_energies.size > 0 else 1e-4
    max_e = max(10.0, max_E_MeV * 1.5)
    bins = np.logspace(np.log10(min_e), np.log10(max_e), 200)
    centers = np.sqrt(bins[:-1] * bins[1:])
    widths = np.diff(bins)

    counts_sim, _ = np.histogram(spec.energies_MeV, bins=bins, weights=spec.weights)
    pdf_vals = physics_mj.calculate_mj_pdf(centers, T_keV)
    counts_th = pdf_vals * widths * total_weight

    positive_diff = np.maximum(0.0, counts_sim - counts_th)
    excess_energy_MeV = np.sum(positive_diff * centers)

    excess_ratio = excess_energy_MeV / total_energy_MeV if total_energy_MeV > 0 else 0.0

    return {
        'T_keV': T_keV,
        'excess_ratio': excess_ratio,
        'max_E_MeV': max_E_MeV
    }


def compute_spectrum_evolution_metrics(run: SimulationRun) -> Dict[str, np.ndarray]:
    """
    遍历所有粒子文件，计算每一帧的等效温度、非热占比和最大能量。
    此函数的返回值将被自动序列化缓存。
    """
    times = []
    temps_keV = []
    excess_ratios = []
    max_energies = []

    # 获取文件列表
    files_to_process = run.particle_files
    if not files_to_process:
        return {}

    console.print(f"  [cyan]计算能谱演化指标 (共 {len(files_to_process)} 帧)...[/cyan]")

    for fpath in tqdm(files_to_process, desc="  处理能谱", unit="file", leave=False):
        # 1. 使用 run 提供的单文件读取能力

        spec: Optional[SpectrumData] = run.get_spectrum_from_path(fpath)

        if spec is None or spec.weights.size == 0:
            continue

        # 获取时间
        step = int(os.path.basename(fpath).split('_')[-1].split('.')[0])
        time = step * run.sim.dt

        # 2. 使用 _analyze_single_spectrum 进行分析
        metrics = _analyze_single_spectrum(spec)

        times.append(time)
        temps_keV.append(metrics['T_keV'])
        excess_ratios.append(metrics['excess_ratio'])
        max_energies.append(metrics['max_E_MeV'])

        # 显式释放内存，防止大循环内存泄漏
        del spec
        gc.collect()

    return {
        "time": np.array(times),
        "T_keV": np.array(temps_keV),
        "excess_ratio": np.array(excess_ratios),
        "max_E_MeV": np.array(max_energies)
    }


class SpectrumEvolutionModule(BaseAnalysisModule):
    @property
    def name(self) -> str:
        return "能谱时间演化分析"

    @property
    def description(self) -> str:
        return "分析每个时间步的能谱，绘制等效温度、非热能量占比和最大能量随时间的演化。"

    def run(self, loaded_runs: List[SimulationRun]):
        console.print("\n[bold magenta]执行: 能谱时间演化分析...[/bold magenta]")
        valid_runs = filter_valid_runs(loaded_runs, require_particles=True, min_particle_files=2)
        if not valid_runs:
            console.print("[yellow]警告: 没有找到有效的粒子诊断文件，跳过此分析。[/yellow]")
            return

        for run in valid_runs:
            self._process_and_plot_run(run)

    def _process_and_plot_run(self, run: SimulationRun):
        console.print(f"\n[bold]分析模拟: {run.name}[/bold]")

        data = compute_spectrum_evolution_metrics(run)

        if not data or len(data["time"]) == 0:
            console.print("[red]  错误: 未能生成能谱演化数据。[/red]")
            return

        times = data["time"]
        temps_keV = data["T_keV"]
        excess_ratios = data["excess_ratio"]
        max_energies = data["max_E_MeV"]

        # --- 绘图 ---
        filename_override = f"{run.name}_analysis_spectrum_evolution"
        with create_analysis_figure(run, "spectrum_evolution", num_plots=3,
                                    figsize=(10, 12), override_filename=filename_override) as (fig, axes):
            ax_temp, ax_excess, ax_max_e = axes

            # 图1: 等效温度 (整体加热)
            ax_temp.plot(times, temps_keV, 'o-', color='darkorange', label='等效温度')
            ax_temp.set_title("整体加热趋势 (Equivalent Temperature)")
            ax_temp.set_ylabel("等效温度 T_fit (keV)")
            ax_temp.grid(True, alpha=0.3)

            # 图2: 非热能量占比 (尾部加速)
            ax_excess.plot(times, np.array(excess_ratios) * 100, 'o-', color='crimson', label='非热能量占比')
            ax_excess.set_title("非热加速效率 (Non-thermal Energy Fraction)")
            ax_excess.set_ylabel(r"非热能量占比 (%)")
            ax_excess.set_ylim(bottom=0)
            ax_excess.grid(True, alpha=0.3)

            # 图3: 最大粒子能量
            ax_max_e.plot(times, max_energies, 'o-', color='royalblue', label='最大能量')
            ax_max_e.set_title("最大粒子能量演化")
            ax_max_e.set_xlabel("时间 (s)")
            ax_max_e.set_ylabel("最大动能 (MeV)")
            ax_max_e.set_yscale('log')
            ax_max_e.grid(True, which='both', alpha=0.3)
