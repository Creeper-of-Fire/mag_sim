from typing import List

import numpy as np
from scipy.interpolate import interp1d

from analysis.core.parameter_selector import ParameterSelector
from analysis.core.simulation import SimulationRun, SpectrumData
from analysis.core.utils import console
from analysis.modules.abstract.base_module import BaseComparisonModule
from analysis.plotting.comparison_layout import ComparisonContext, ComparisonLayout
from analysis.plotting.layout import create_analysis_figure
from analysis.plotting.styles import get_style


class QuantileDiagnosticModule(BaseComparisonModule):
    @property
    def name(self) -> str:
        return "诊断：能量分位数演化 (加速 vs 加热)"

    @property
    def description(self) -> str:
        return "通过对比能量的最高端(99.9%)与中位数(50%)的比率变化，判断是发生了随机加速(比率上升)还是整体加热(比率不变)。"

    def _get_quantile(self, spec: SpectrumData, q: float) -> float:
        """
        计算能谱的第 q 分位数能量 (例如 q=0.999)。
        由于是直方图数据，需要通过累积分布函数(CDF)插值。
        """
        if spec is None or spec.weights.sum() == 0:
            return 0.0

        # 1. 按能量排序 (通常已经是排好序的，但为了保险)
        sort_idx = np.argsort(spec.energies_MeV)
        energies = spec.energies_MeV[sort_idx]
        weights = spec.weights[sort_idx]

        # 2. 计算 CDF
        cumsum = np.cumsum(weights)
        cdf = cumsum / cumsum[-1]

        # 3. 插值寻找 q 对应的能量
        # 考虑到 q 可能非常接近 1，或者 energies 范围很大，使用 log 空间插值更好
        # 但为了稳健，如果 q 在 CDF 范围内，直接线性插值即可
        if q > cdf[-1]: return energies[-1]
        if q < cdf[0]: return energies[0]

        interp_func = interp1d(cdf, energies, kind='linear', bounds_error=False, fill_value="extrapolate")
        return float(interp_func(q))

    def run(self, loaded_runs: List[SimulationRun]):
        style = get_style()
        console.print("\n[bold magenta]执行: 分位数诊断 (判断是否存在隐形尾巴)...[/bold magenta]")

        ctx = ComparisonContext(loaded_runs, "quantile_diag")
        runs, x_scaled = ctx.unpack
        x_raw, _, x_label = ctx.x

        # 准备数据容器
        ratios = []  # E_99.9 / E_50
        e_maxs = []  # E_max
        e_heade = []  # E_99.9%
        e_medians = []  # E_50%

        for run in runs:
            spec = run.final_spectrum
            if spec is None:
                ratios.append(0)
                e_maxs.append(0)
                e_heade.append(0)
                e_medians.append(0)
                continue

            # 获取分位数
            e_max = spec.energies_MeV.max()
            e_999 = self._get_quantile(spec, 0.999)  # 头部千分之一
            e_500 = self._get_quantile(spec, 0.500)  # 中位数

            # 计算比率 (类似基尼系数的概念，衡量能量不平等的程度)
            if e_500 > 0:
                ratio = e_999 / e_500
            else:
                ratio = 0.0

            ratios.append(ratio)
            e_maxs.append(e_max)
            e_heade.append(e_999)
            e_medians.append(e_500)

            console.print(f"  [{run.name}] Ratio(99.9%/50%)={ratio:.2f} | E_med={e_500 * 1000:.1f} keV, E_top={e_999 * 1000:.1f} keV")

        with ComparisonLayout(ctx) as layout:
            ax1 = layout.request_axes()
            ax2 = layout.request_axes()

            # --- 图1: 能量不平等度 (Acceleration Indicator) ---
            # 如果这条线是平的或者下降的，说明没有非热加速。
            # 如果这条线向上翘，说明虽然肉眼看不出，但尾部确实相对于头部在变长。
            ax1.plot(x_scaled, ratios, marker='o', color=style.color_comparison_primary, lw=2, label='$E_{99.9\%} / E_{median}$')

            ax1.set_ylabel("能谱硬度比\n($E_{99.9\%} / E_{50\%}$)")
            ax1.grid(True, linestyle='--', alpha=0.5)
            ax1.legend()

            # --- 图2: 绝对能量演化 (Heating Indicator) ---
            # 用半对数坐标看数量级
            ax2.semilogy(x_scaled, np.array(e_maxs) * 1000, marker='x', linestyle='--', color='gray', alpha=0.6, label='$E_{max}$')
            ax2.semilogy(x_scaled, np.array(e_heade) * 1000, marker='s', color=style.color_comparison_primary, label='$E_{99.9\%}$ (Head)')
            ax2.semilogy(x_scaled, np.array(e_medians) * 1000, marker='^', color=style.color_comparison_secondary, label='$E_{50\%}$ (Body)')

            ax2.set_ylabel("能量 (keV)")
            ax2.grid(True, linestyle='--', alpha=0.5, which='both')
            ax2.legend()

            console.print("\n[bold yellow]诊断指南:[/bold yellow]")
            console.print("1. 看图1 (Ratio):")
            console.print("   - 如果曲线 [green]上升[/green]: 确认存在非热加速机制，只是信号弱。")
            console.print("   - 如果曲线 [red]平坦或下降[/red]: 确认只有体加热(Bulk Heating)或压缩，无有效加速。")
            console.print("2. 看图2 (Energy):")
            console.print("   - 如果 $E_{50\%}$ (Body) 随磁场显著上升，说明磁能主要转化为了整体热能。")
