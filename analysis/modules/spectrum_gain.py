# analysis/modules/spectrum_gain.py

from typing import List, Set, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

from .base_module import BaseComparisonModule
from ..core.simulation import SimulationRun
from ..core.utils import console
from ..plotting.layout import create_analysis_figure
from ..plotting.styles import get_style


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

    @property
    def required_data(self) -> Set[str]:
        return {'initial_spectrum', 'final_spectrum'}

    def _create_common_bins(self, runs: List[SimulationRun], num_bins: int = 150):
        """为所有模拟创建统一的能量分箱，确保比率计算的基准一致。"""
        all_energies = []
        for run in runs:
            if run.initial_spectrum and run.initial_spectrum.energies_MeV.size > 0:
                all_energies.append(run.initial_spectrum.energies_MeV)
            if run.final_spectrum and run.final_spectrum.energies_MeV.size > 0:
                all_energies.append(run.final_spectrum.energies_MeV)

        if not all_energies:
            raise ValueError("没有有效的能谱数据。")

        combined = np.concatenate(all_energies)
        positive = combined[combined > 0]

        # 避免 min 为 0 或过小
        min_e = max(positive.min() * 0.9, 1e-4)
        max_e = positive.max() * 1.1

        # 使用对数分箱
        bins = np.logspace(np.log10(min_e), np.log10(max_e), num_bins)
        centers = np.sqrt(bins[:-1] * bins[1:])
        widths = np.diff(bins)
        return bins, centers, widths

    def run(self, loaded_runs: List[SimulationRun]):
        console.print("\n[bold magenta]执行: 能谱增益(比率)分析...[/bold magenta]")

        valid_runs = [r for r in loaded_runs if r.initial_spectrum and r.final_spectrum]
        if len(valid_runs) < 1:
            console.print("[yellow]警告: 需要至少一个包含初始和最终能谱的模拟。[/yellow]")
            return

        # 1. 创建统一分箱
        try:
            bins, centers, widths = self._create_common_bins(valid_runs)
        except ValueError as e:
            console.print(f"[red]错误: {e}[/red]")
            return

        # 2. 绘图
        # 使用 create_analysis_figure 管理画布和参数表
        with create_analysis_figure(
                valid_runs,
                "comparison_spectrum_gain",
                num_plots=1,
                figsize=(10, 6)
        ) as (fig, ax):

            style = get_style()

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