# analysis/modules/spectrum_evolution_heatmap.py
import gc
import os
from typing import List, Set, Tuple, Optional

import h5py
import numpy as np
from matplotlib.colors import LogNorm
from tqdm import tqdm

from analysis.modules.abstract.base_module import BaseAnalysisModule
from analysis.core.simulation import SimulationRun
from analysis.core.utils import console
from analysis.plotting.layout import create_analysis_figure
from scipy.constants import c, m_e, e

# --- 内部辅助函数 (从 data_loader.py 借鉴并简化) ---

def _get_h5_dataset_local(h5_item: h5py.Group, component_path: str) -> np.ndarray:
    """本地化的 HDF5 数据集读取器。"""
    item = h5_item[component_path]
    if isinstance(item, h5py.Dataset):
        return item[:]
    elif isinstance(item, h5py.Group):
        if not item: return np.array([])
        if len(item) == 1:
            first_key = list(item.keys())[0]
            if isinstance(item[first_key], h5py.Dataset):
                return item[first_key][:]
        dataset_name = component_path.split('/')[-1]
        if dataset_name in item and isinstance(item[dataset_name], h5py.Dataset):
            return item[dataset_name][:]
    raise KeyError(f"在组 '{item.name}' 中无法找到预期的数据集。")


def _get_step_from_filename_local(filename: str) -> Optional[int]:
    """本地化的从文件名提取步数的函数。"""
    try:
        base_name = os.path.basename(filename)
        step_str = base_name.split('_')[-1].split('.')[0]
        return int(step_str)
    except (IndexError, ValueError):
        return None


def _load_spectrum_minimal(h5_filepath: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    极简版的能谱加载器，只返回能量和权重，不做任何封装。
    """
    all_energies_MeV, all_weights = [], []
    m_e_c2_J = m_e * c ** 2
    J_PER_MEV = e * 1e6
    try:
        with h5py.File(h5_filepath, 'r') as f:
            step_key = list(f['data'].keys())[0]
            particles_group = f[f'data/{step_key}/particles']
            for species_name in particles_group.keys():
                if 'photon' in species_name: continue
                species_group = particles_group[species_name]
                try:
                    px = _get_h5_dataset_local(species_group, 'momentum/x')
                    py = _get_h5_dataset_local(species_group, 'momentum/y')
                    pz = _get_h5_dataset_local(species_group, 'momentum/z')
                    weights = _get_h5_dataset_local(species_group, 'weighting')
                except KeyError:
                    continue # 如果某个物种缺少动量或权重数据，则跳过

                if weights.size == 0: continue
                p_sq = px ** 2 + py ** 2 + pz ** 2
                kinetic_energy_J = np.sqrt(p_sq * c ** 2 + m_e_c2_J ** 2) - m_e_c2_J
                all_energies_MeV.append(kinetic_energy_J / J_PER_MEV)
                all_weights.append(weights)

        if not all_energies_MeV:
            return None, None

        return np.concatenate(all_energies_MeV), np.concatenate(all_weights)
    except Exception:
        # 在批量处理时，静默处理单个文件的失败
        return None, None


class SpectrumEvolutionHeatmapModule(BaseAnalysisModule):
    @property
    def name(self) -> str:
        return "能谱演化热力图 (Waterfall Plot)"

    @property
    def description(self) -> str:
        return "绘制'时间-能量-粒子数'热力图，展示能谱随时间的连续演化过程。"

    def _create_common_bins(self, run: SimulationRun, sampled_files: List[str]) -> np.ndarray:
        """
        通过扫描部分文件，为所有时间步创建一个统一的能量分箱。
        """
        console.print("  [cyan]预扫描文件以确定全局能量范围...[/cyan]")
        all_energies = []
        for fpath in tqdm(sampled_files, desc="  预扫描", unit="file", leave=False):
            energies, _ = _load_spectrum_minimal(fpath)
            if energies is not None and energies.size > 0:
                all_energies.append(energies)

        if not all_energies:
            raise ValueError("在预扫描中未能从任何文件中加载到有效的粒子数据。")

        combined = np.concatenate(all_energies)
        positive = combined[combined > 0]
        if positive.size < 10:
            raise ValueError("有效粒子数据过少，无法创建能量分箱。")

        # 使用对数分箱，覆盖从最小值到最大值的范围
        global_min = max(positive.min() * 0.9, 1e-4)
        global_max = positive.max() * 1.1
        console.print(f"  [green]全局能量范围: [{global_min:.2e}, {global_max:.2e}] MeV[/green]")

        # 返回分箱的边界
        return np.logspace(np.log10(global_min), np.log10(global_max), 200)

    def _analyze_single_run(self, run: SimulationRun):
        files = run.particle_files
        if not files or len(files) < 2:
            console.print(f"  [yellow]粒子文件不足 ({len(files)}个)，无法生成演化图。[/yellow]")
            return

        # 为了性能和绘图清晰度，对文件进行采样，最多100帧
        max_frames = 100
        if len(files) > max_frames:
            indices = np.linspace(0, len(files) - 1, max_frames, dtype=int)
            sampled_files = [files[i] for i in indices]
        else:
            sampled_files = files

        try:
            # 1. 创建统一分箱
            common_bins = self._create_common_bins(run, sampled_files)
            bin_widths = np.diff(common_bins)
        except ValueError as e:
            console.print(f"  [red]错误: {e}[/red]")
            return

        # 2. 循环处理每个文件，填充数据矩阵
        times = []
        dNdE_list = []
        console.print("  [cyan]逐帧生成能谱数据...[/cyan]")
        for fpath in tqdm(sampled_files, desc="  帧处理", unit="file", leave=False):
            step = _get_step_from_filename_local(fpath)
            if step is None: continue

            energies, weights = _load_spectrum_minimal(fpath)

            if energies is None or weights is None or energies.size == 0:
                # 保持矩阵形状一致，如果某帧数据为空，则填充0
                dNdE = np.zeros_like(bin_widths)
            else:
                counts, _ = np.histogram(energies, bins=common_bins, weights=weights)
                dNdE = counts / bin_widths

            dNdE_list.append(dNdE)
            times.append(step * run.sim.dt)

            # --- 激进的内存管理 ---
            del energies, weights, dNdE
            gc.collect()

        # 3. 准备绘图数据
        # 将列表转换为2D NumPy数组
        # 转置 (.T) 使其维度为 (能量箱, 时间步) 以便 pcolormesh 绘图
        dNdE_matrix = np.array(dNdE_list).T
        times_array = np.array(times)

        # 准备 pcolormesh 的坐标网格
        # X 对应时间，Y 对应能量中心点
        energy_centers = np.sqrt(common_bins[:-1] * common_bins[1:])
        X, Y = np.meshgrid(times_array, energy_centers)

        # 4. 绘图
        filename_override = f"{run.name}_analysis_spectrum_heatmap"
        with create_analysis_figure(run, "spectrum_evolution_heatmap", num_plots=1, figsize=(10, 7), override_filename=filename_override) as (fig, ax):
            fig.suptitle(f"能谱演化热力图: {run.name}", fontsize=16)

            # 使用对数颜色映射，并处理0值
            # vmin 设为最大值的百万分之一，防止颜色条范围过大
            vmax = dNdE_matrix.max()
            vmin = max(vmax * 1e-6, 1e-9) # 避免 vmax 为 0
            norm = LogNorm(vmin=vmin, vmax=vmax)

            # 绘制伪彩色图
            color_setting = ax.pcolormesh(X, Y, dNdE_matrix, norm=norm, cmap='inferno', shading='auto')

            # 添加颜色条
            cbar = fig.colorbar(color_setting, ax=ax)
            cbar.set_label('粒子数密度 dN/dE [/MeV]')

            # 设置坐标轴
            ax.set_yscale('log')
            ax.set_xlabel('时间 (s)')
            ax.set_ylabel('动能 (MeV)')
            ax.set_title("时间-能量-粒子数密度", fontsize=12)

    def run(self, loaded_runs: List[SimulationRun]):
        console.print("\n[bold magenta]执行: 能谱演化热力图 (Waterfall Plot)...[/bold magenta]")

        for run in loaded_runs:
            console.print(f"\n[bold]分析模拟: [cyan]{run.name}[/cyan][/bold]")
            self._analyze_single_run(run)