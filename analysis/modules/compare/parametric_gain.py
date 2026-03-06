# analysis/modules/parametric_gain.py

from typing import List, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np

from analysis.core.parameter_selector import ParameterSelector
from analysis.core.simulation import SimulationRun
from analysis.core.utils import console
from analysis.modules.abstract.base_module import BaseComparisonModule
from analysis.modules.utils.comparison_utils import create_common_energy_bins
from analysis.modules.utils.spectrum_tools import filter_valid_runs
from analysis.plotting.layout import create_analysis_figure

# 增加一个统计阈值，避免因为初始粒子数太少导致比率爆炸
MIN_INITIAL_COUNTS = 10


class ParametricGainModule(BaseComparisonModule):
    @property
    def name(self) -> str:
        return "参数扫描：能谱增益效率"

    @property
    def description(self) -> str:
        return "自动识别变化的输入参数(X轴)，绘制峰值增益比率(f_final/f_initial)和对应能量的变化趋势。"

    # =========================================================================
    # 1. 物理计算核心 (增益比率计算)
    # =========================================================================

    def _calculate_gain_metrics(self, run: SimulationRun, bins: np.ndarray, centers: np.ndarray, widths: np.ndarray) -> Dict[str, float]:
        """
        核心算法：计算单个run的峰值增益和对应能量。
        """
        spec_i = run.initial_spectrum
        spec_f = run.final_spectrum

        if not all([spec_i, spec_f, spec_i.weights.size > 0, spec_f.weights.size > 0]):
            return {'peak_gain': 0.0, 'peak_gain_energy_mev': 0.0}

        # 1. 计算初始和最终的 dN/dE
        counts_i, _ = np.histogram(spec_i.energies_MeV, bins=bins, weights=spec_i.weights)
        counts_f, _ = np.histogram(spec_f.energies_MeV, bins=bins, weights=spec_f.weights)
        dNdE_i = counts_i / widths
        dNdE_f = counts_f / widths

        # 2. 计算比率
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = dNdE_f / dNdE_i

        # 3. 建立一个统计学上可靠的掩码
        #   - 初始粒子数必须大于阈值，防止 1/1 这种偶然情况
        #   - 初始 dN/dE 必须大于0
        #   - 比率必须是有效的数值
        mask = (counts_i >= MIN_INITIAL_COUNTS) & (dNdE_i > 0) & np.isfinite(ratio)

        if not np.any(mask):
            return {'peak_gain': 1.0, 'peak_gain_energy_mev': 0.0}

        # 4. 在可靠区域内寻找峰值
        valid_ratios = ratio[mask]
        valid_centers = centers[mask]

        max_idx = np.argmax(valid_ratios)
        peak_gain = valid_ratios[max_idx]
        peak_energy = valid_centers[max_idx]

        return {
            'peak_gain': peak_gain,
            'peak_gain_energy_mev': peak_energy
        }

    # =========================================================================
    # 2. 运行与绘图
    # =========================================================================

    def run(self, loaded_runs: List[SimulationRun]):
        console.print("\n[bold magenta]执行: 参数扫描能谱增益分析...[/bold magenta]")

        valid_runs = filter_valid_runs(loaded_runs, require_particles=True, min_particle_files=2)
        if len(valid_runs) < 2:
            console.print("[red]错误: 需要至少 2 个包含初始和最终能谱的模拟。[/red]")
            return

        # 1. 建立全局统一分箱
        try:
            bins, centers, widths = create_common_energy_bins(valid_runs, num_bins=150)
        except ValueError as e:
            console.print(f"[red]创建分箱失败: {e}[/red]")
            return

        # 2. 使用 Selector 准备数据
        selector = ParameterSelector(valid_runs)
        x_label, x_vals, sorted_runs = selector.select()

        # 3. 生成文件名
        final_filename = selector.generate_filename(x_label, sorted_runs, prefix="scan_gain")

        # 4. 计算指标
        y_peak_gain = []
        y_peak_energy = []

        console.print(f"  正在计算每个run的峰值增益...")
        for i, run in enumerate(sorted_runs):
            m = self._calculate_gain_metrics(run, bins, centers, widths)
            y_peak_gain.append(m['peak_gain'])
            y_peak_energy.append(m['peak_gain_energy_mev'])

            console.print(f"    - {run.name}: {x_label}={x_vals[i]} -> "
                          f"Peak Gain={m['peak_gain']:.2f} @ {m['peak_gain_energy_mev']:.3f} MeV")

        try:
            x_num = [float(v) for v in x_vals]
            is_num = True
        except (ValueError, TypeError):
            x_num = range(len(x_vals))
            is_num = False

        with create_analysis_figure(sorted_runs, "scan_gain", num_plots=2, figsize=(9, 8), override_filename=final_filename) as (fig, (ax1, ax2)):

            # --- 图1: 峰值增益 ---
            ax1.plot(x_num, y_peak_gain, 'o-', color='crimson', lw=2, markersize=6)
            ax1.set_ylabel(r"峰值增益比率 $\max(f_{final}/f_{initial})$")
            ax1.set_title(f"峰值增益 vs {x_label}", fontsize=14)
            ax1.grid(True, linestyle='--', alpha=0.5)
            ax1.set_yscale('log')  # 增益通常是对数尺度更直观

            # --- 图2: 发生峰值增益的能量 ---
            ax2.plot(x_num, y_peak_energy, 's-', color='darkorange', lw=2, markersize=6)
            ax2.set_ylabel("峰值增益处能量 (MeV)")
            ax2.set_title(f"最有效加速能量点 vs {x_label}", fontsize=14)
            ax2.set_xlabel(x_label if is_num else "Simulation Case", fontsize=12)
            ax2.grid(True, linestyle='--', alpha=0.5)
            ax2.set_yscale('log')

            if not is_num:
                # 如果X轴是字符串，设置标签
                plt.setp(ax1.get_xticklabels(), visible=False)  # 隐藏上图的x轴标签
                ax2.set_xticks(x_num)
                ax2.set_xticklabels(x_vals, rotation=45, ha='right')

            fig.tight_layout(rect=[0, 0, 1, 0.96])  # 为fig.suptitle留出空间
