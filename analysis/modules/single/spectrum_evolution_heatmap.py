# analysis/modules/spectrum_evolution_heatmap.py
from typing import List

import numpy as np
from matplotlib.colors import LogNorm
from tqdm import tqdm

from analysis.core.simulation import SimulationRun
from analysis.core.utils import console
from analysis.modules.abstract.base_module import BaseAnalysisModule
from analysis.plotting.layout import create_analysis_figure


class SpectrumEvolutionHeatmapModule(BaseAnalysisModule):
    @property
    def name(self) -> str:
        return "能谱演化热力图 (Waterfall Plot)"

    @property
    def description(self) -> str:
        return "绘制'时间-能量-粒子数'热力图，展示能谱随时间的连续演化过程。"

    def _analyze_single_run(self, run: SimulationRun):
        if not run.particle_files or len(run.particle_files) < 2:
            console.print(f"  [yellow]粒子文件不足 ({len(run.particle_files)}个)，无法生成演化图。[/yellow]")
            return

        times, energy_centers, dNdE_matrix = run.get_spectrum_evolution_matrix(n_bins=200, log_scale=True)

        # 3. 准备绘图数据
        # 将列表转换为2D NumPy数组
        # 转置 (.T) 使其维度为 (能量箱, 时间步) 以便 pcolormesh 绘图
        dNdE_matrix_T = dNdE_matrix.T

        # 准备 pcolormesh 的坐标网格
        # X 对应时间，Y 对应能量中心点
        # 我们需要的是 bin edges，而不是 centers，以便 pcolormesh 正确绘制
        # 让我们从 centers 推导出 edges
        log_energy = np.log10(energy_centers)
        log_diff = np.diff(log_energy)
        # 假设对数间距是恒定的
        half_log_step = log_diff[0] / 2
        log_edges = np.append(log_energy - half_log_step, log_energy[-1] + half_log_step)
        energy_edges = 10 ** log_edges

        # 创建带有 edges 的时间网格
        time_diff = np.diff(times)
        half_time_step = time_diff[0] / 2 if time_diff.size > 0 else (times[0] or 1e-9)
        time_edges = np.append(times - half_time_step, times[-1] + half_time_step)

        X, Y = np.meshgrid(time_edges, energy_edges)

        # 4. 绘图
        filename_override = f"{run.name}_analysis_spectrum_heatmap"
        with create_analysis_figure(run, "spectrum_evolution_heatmap", num_plots=1, figsize=(10, 7), override_filename=filename_override) as (fig, ax):
            fig.suptitle(f"能谱演化热力图: {run.name}", fontsize=16)

            # 使用对数颜色映射，并处理0值
            # vmin 设为最大值的百万分之一，防止颜色条范围过大
            vmax = dNdE_matrix_T.max()
            vmin = max(vmax * 1e-6, 1e-9)  # 避免 vmax 为 0
            norm = LogNorm(vmin=vmin, vmax=vmax)

            # 绘制伪彩色图
            color_setting = ax.pcolormesh(X, Y, dNdE_matrix_T, norm=norm, cmap='inferno', shading='auto')

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
