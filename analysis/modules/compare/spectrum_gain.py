# analysis/modules/spectrum_gain.py

from typing import List

import numpy as np

from analysis.core.simulation import SimulationRun
from analysis.core.utils import console
from analysis.modules.abstract.base_module import BaseComparisonModule
from analysis.modules.utils.comparison_utils import create_common_energy_bins
from analysis.modules.utils.spectrum_tools import filter_valid_runs
from analysis.plotting.layout import create_analysis_figure, AnalysisLayout
from analysis.plotting.styles import get_style


class SpectrumGainModule(BaseComparisonModule):
    """
    能谱增益/比率分析模块。
    计算并绘制 Ratio = Final_Spectrum / Initial_Spectrum。
    用于直观展示粒子加速的倍数（Amplification Factor）。
    """

    @property
    def name(self) -> str:
        return "能谱增益分析 (Ratio vs Energy)"

    @property
    def description(self) -> str:
        return "计算最终能谱与初始能谱的比率 (f_final / f_initial)，展示各能量段的粒子数放大倍数。"

    def run(self, loaded_runs: List[SimulationRun]):
        console.print("\n[bold magenta]执行: 能谱增益(比率)分析...[/bold magenta]")

        valid_runs = filter_valid_runs(loaded_runs, require_particles=True, min_particle_files=2)
        if len(valid_runs) < 1:
            console.print("[yellow]警告: 需要至少一个包含初始和最终能谱的模拟。[/yellow]")
            return

        # 1. 创建统一分箱
        try:
            bins, centers, widths = create_common_energy_bins(valid_runs, num_bins=150)
        except ValueError as e:
            console.print(f"[red]错误: {e}[/red]")
            return

        # 2. 绘图
        with AnalysisLayout(valid_runs, "comparison_spectrum_gain") as layout:
            ax = layout.request_axes()
            # 绘制基准线 y=1 (无变化)
            ax.axhline(1, color='black', linestyle='-', linewidth=1, alpha=0.5, label='无变化 (Ratio=1)')

            for run in valid_runs:
                # 计算初始直方图
                counts_i, _ = np.histogram(run.initial_spectrum.energies_MeV, bins=bins, weights=run.initial_spectrum.weights)
                dNdE_i = counts_i / widths

                # 计算最终直方图
                counts_f, _ = np.histogram(run.final_spectrum.energies_MeV, bins=bins, weights=run.final_spectrum.weights)
                dNdE_f = counts_f / widths

                # 计算比率 Ratio = Final / Initial
                # 处理分母为0的情况：
                # 1. 如果 Initial > 0, Ratio = Final / Initial
                # 2. 如果 Initial = 0 且 Final = 0, Ratio = 1 (无变化/无数据)
                # 3. 如果 Initial = 0 且 Final > 0, Ratio = inf (数学上)，但画图时可以设为 NaN 或忽略

                with np.errstate(divide='ignore', invalid='ignore'):
                    ratio = dNdE_f / dNdE_i

                # 掩码：只绘制初始谱有粒子的地方，且比率有效的地方
                # 通常我们只关心初始热库覆盖的范围内，或者稍微延伸一点的地方
                mask = (dNdE_i > 0) & (counts_i > 5) & (~np.isnan(ratio))

                if np.any(mask):
                    ax.plot(centers[mask], ratio[mask], linewidth=2, label=run.name)

            # --- 设置坐标轴 ---
            ax.set_title("能谱演化增益比率 (Amplification Factor)", fontsize=16)
            ax.set_xlabel("动能 (MeV)", fontsize=12)
            ax.set_ylabel(r"增益比率 $R(E) = f_{final} / f_{initial}$", fontsize=12)

            ax.set_xscale('log')
            ax.set_yscale('log')

            ax.grid(True, which="major", ls="-", alpha=0.3)
            ax.grid(True, which="minor", ls=":", alpha=0.1)

            ax.legend(loc='best', fontsize=10)

            # 在图上添加注释区域
            # y > 1 : Acceleration / Heating
            # y < 1 : Cooling / Depletion
            ylim = ax.get_ylim()
            # 只有当Y轴跨度包含1时才标记
            if ylim[0] < 1 and ylim[1] > 1:
                ax.text(0.02, 0.95, "加速/加热区域 (Ratio > 1)", transform=ax.transAxes,
                        color='green', alpha=0.7, fontweight='bold', ha='left', va='top')
                ax.text(0.02, 0.05, "耗散/冷却区域 (Ratio < 1)", transform=ax.transAxes,
                        color='red', alpha=0.7, fontweight='bold', ha='left', va='bottom')
