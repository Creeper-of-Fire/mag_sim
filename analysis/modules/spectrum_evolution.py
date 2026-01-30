# analysis/modules/spectrum_evolution.py

import os
import gc
from typing import List, Set, Dict, Any, Optional

import h5py
import numpy as np
from scipy.constants import k as kB, c, m_e, e
from scipy.optimize import root_scalar
from scipy.special import kn as bessel_k
from tqdm import tqdm

from .base_module import BaseAnalysisModule
from ..core.simulation import SimulationRun, SpectrumData
from ..core.utils import console
from ..plotting.layout import create_analysis_figure
from ..core.data_loader import _get_h5_dataset  # 复用底层加载逻辑

# --- 物理常量 ---
ME_C2_J = m_e * c ** 2
J_PER_MEV = e * 1e6
J_TO_KEV = 1.0 / (e * 1e3)


class SpectrumEvolutionModule(BaseAnalysisModule):
    @property
    def name(self) -> str:
        return "能谱时间演化分析"

    @property
    def description(self) -> str:
        return "分析每个时间步的能谱，绘制等效温度、非热能量占比和最大能量随时间的演化。"

    @property
    def required_data(self) -> Set[str]:
        # 只需要粒子文件列表，我们会自己逐个加载
        return {'particle_files'}

    # =========================================================================
    # 1. 核心计算逻辑 (逐帧调用)
    # =========================================================================

    def _load_spectrum_from_file(self, h5_filepath: str) -> Optional[SpectrumData]:
        """从单个 HDF5 文件加载能谱，这是模块的私有加载器。"""
        all_energies_MeV, all_weights = [], []
        try:
            with h5py.File(h5_filepath, 'r') as f:
                step_key = list(f['data'].keys())[0]
                particles_group = f[f'data/{step_key}/particles']
                for species_name in particles_group.keys():
                    if 'photon' in species_name: continue
                    species_group = particles_group[species_name]
                    px = _get_h5_dataset(species_group, 'momentum/x')
                    py = _get_h5_dataset(species_group, 'momentum/y')
                    pz = _get_h5_dataset(species_group, 'momentum/z')
                    weights = _get_h5_dataset(species_group, 'weighting')
                    if weights.size == 0: continue
                    p_sq = px ** 2 + py ** 2 + pz ** 2
                    kinetic_energy_J = np.sqrt(p_sq * c ** 2 + ME_C2_J ** 2) - ME_C2_J
                    all_energies_MeV.append(kinetic_energy_J / J_PER_MEV)
                    all_weights.append(weights)
            if not all_energies_MeV: return None
            return SpectrumData(np.concatenate(all_energies_MeV), np.concatenate(all_weights))
        except Exception:
            return None

    def _solve_temperature_kev(self, avg_ek_mev: float) -> float:
        """根据平均动能反推 M-J 温度 (keV)"""
        if avg_ek_mev <= 0: return 0.0
        target_avg_ek_j = avg_ek_mev * J_PER_MEV

        def mj_avg_energy(T_K):
            if T_K <= 0: return -1.0
            theta = (kB * T_K) / ME_C2_J
            if theta < 1e-9: return 1.5 * kB * T_K
            return ME_C2_J * (3 * theta + bessel_k(1, 1.0 / theta) / bessel_k(2, 1.0 / theta) - 1.0)

        T_guess = (2.0 / 3.0) * target_avg_ek_j / kB
        try:
            sol = root_scalar(lambda t: mj_avg_energy(t) - target_avg_ek_j,
                              x0=T_guess, bracket=[T_guess * 0.1, T_guess * 10.0], method='brentq')
            return (sol.root * kB) * J_TO_KEV
        except:
            return 0.0

    def _calculate_mj_pdf(self, E_MeV: np.ndarray, T_keV: float) -> np.ndarray:
        """计算 M-J 概率密度 f(E) (per MeV)"""
        if T_keV <= 0: return np.zeros_like(E_MeV)
        T_J = T_keV * 1e3 * e
        theta = T_J / ME_C2_J
        norm = 1.0 / (ME_C2_J * theta * bessel_k(2, 1.0 / theta))
        E_J = E_MeV * J_PER_MEV
        gamma = 1.0 + E_J / ME_C2_J
        pc_J = np.sqrt(E_J * (E_J + 2 * ME_C2_J))
        pdf = norm * (pc_J / ME_C2_J) * gamma * np.exp(-gamma / theta) * J_PER_MEV
        return pdf

    def _analyze_single_spectrum(self, spec: SpectrumData) -> Dict[str, float]:
        """对单个时刻的能谱进行分析，返回关键指标。"""
        if spec is None or spec.weights.size == 0:
            return {'T_keV': 0.0, 'excess_ratio': 0.0, 'max_E_MeV': 0.0}

        total_energy_MeV = np.sum(spec.energies_MeV * spec.weights)
        total_weight = np.sum(spec.weights)
        avg_energy_MeV = total_energy_MeV / total_weight
        T_keV = self._solve_temperature_kev(avg_energy_MeV)

        pos_energies = spec.energies_MeV[spec.energies_MeV > 0]
        max_E_MeV = pos_energies.max() if pos_energies.size > 0 else 0.0

        # --- 计算非热能量 ---
        min_e = max(1e-4, pos_energies.min()) if pos_energies.size > 0 else 1e-4
        max_e = max(10.0, max_E_MeV * 1.5)
        bins = np.logspace(np.log10(min_e), np.log10(max_e), 200)
        centers = np.sqrt(bins[:-1] * bins[1:])
        widths = np.diff(bins)

        counts_sim, _ = np.histogram(spec.energies_MeV, bins=bins, weights=spec.weights)
        pdf_vals = self._calculate_mj_pdf(centers, T_keV)
        counts_th = pdf_vals * widths * total_weight

        positive_diff = np.maximum(0.0, counts_sim - counts_th)
        excess_energy_MeV = np.sum(positive_diff * centers)

        excess_ratio = excess_energy_MeV / total_energy_MeV if total_energy_MeV > 0 else 0.0

        return {
            'T_keV': T_keV,
            'excess_ratio': excess_ratio,
            'max_E_MeV': max_E_MeV
        }

    # =========================================================================
    # 2. 模块主流程
    # =========================================================================

    def run(self, loaded_runs: List[SimulationRun]):
        console.print("\n[bold magenta]执行: 能谱时间演化分析...[/bold magenta]")
        valid_runs = [r for r in loaded_runs if r.particle_files]
        if not valid_runs:
            console.print("[yellow]警告: 没有找到有效的粒子诊断文件，跳过此分析。[/yellow]")
            return

        for run in valid_runs:
            self._process_and_plot_run(run)

    def _process_and_plot_run(self, run: SimulationRun):
        console.print(f"\n[bold]分析模拟: {run.name}[/bold]")

        times = []
        temps_keV = []
        excess_ratios = []
        max_energies = []

        files_to_process = run.particle_files

        # 逐帧处理
        for fpath in tqdm(files_to_process, desc="  处理能谱帧", unit="file", leave=False):
            step = int(os.path.basename(fpath).split('_')[-1].split('.')[0])
            time = step * run.sim.dt

            # 1. 加载数据
            spectrum = self._load_spectrum_from_file(fpath)

            # 2. 分析
            if spectrum:
                metrics = self._analyze_single_spectrum(spectrum)
                times.append(time)
                temps_keV.append(metrics['T_keV'])
                excess_ratios.append(metrics['excess_ratio'])
                max_energies.append(metrics['max_E_MeV'])

            # 3. 关键：手动释放内存
            del spectrum
            gc.collect()

        if not times:
            console.print("[red]  错误: 未能处理任何能谱文件。[/red]")
            return

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