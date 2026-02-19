# analysis/modules/parametric_tail_debug.py

from typing import List, Dict

import matplotlib.pyplot as plt
import numpy as np

from analysis.core.parameter_selector import ParameterSelector
from analysis.core.simulation import SimulationRun, SpectrumData
from analysis.core.utils import console
from analysis.modules.abstract.base_module import BaseComparisonModule
from analysis.modules.utils import physics_mj
from analysis.plotting.layout import create_analysis_figure
from analysis.plotting.styles import get_style


class ParametricTailDebugModule(BaseComparisonModule):
    @property
    def name(self) -> str:
        return "DEBUG：非热算法底噪分析"

    @property
    def description(self) -> str:
        return "对比初始时刻(t=0)与最终时刻的'非热能量'计算值，量化算法由统计涨落引起的误差底噪。"

    # =========================================================================
    # 物理计算核心
    # =========================================================================

    def _analyze_spectrum_excess(self, spec: SpectrumData) -> Dict[str, float]:
        """
        基于阈值能量的直接差值积分。
        """
        if spec is None or spec.weights.size == 0:
            return {'T_keV': 0.0, 'excess_ratio': 0.0, 'total_excess_MeV': 0.0}

        # 1. 基础统计与温度拟合
        total_energy_MeV = np.sum(spec.energies_MeV * spec.weights)
        total_weight = np.sum(spec.weights)

        if total_weight == 0:
            return {'T_keV': 0.0, 'excess_ratio': 0.0, 'total_excess_MeV': 0.0}

        avg_energy_MeV = total_energy_MeV / total_weight
        T_keV = physics_mj.solve_mj_temperature_kev(avg_energy_MeV)

        threshold_energy_MeV = (3.0 * T_keV) / 1000.0

        # 2. 建立分箱
        min_e = max(1e-4, spec.energies_MeV.min())
        max_e = max(10.0, spec.energies_MeV.max() * 1.5)
        bins = np.logspace(np.log10(min_e), np.log10(max_e), 200)
        centers = np.sqrt(bins[:-1] * bins[1:])
        widths = np.diff(bins)

        # 3. 模拟数据直方图
        counts_sim, _ = np.histogram(spec.energies_MeV, bins=bins, weights=spec.weights)

        # 4. 理论数据直方图
        pdf_vals = physics_mj.calculate_mj_pdf(centers, T_keV)
        counts_th = pdf_vals * widths * total_weight

        # 5. 直接做差，且只关注高能区
        # 不再使用 np.maximum(0.0, ...)
        diff_counts = counts_sim - counts_th

        # 仅在 E > threshold 区域积分
        mask = centers > threshold_energy_MeV
        excess_energy_MeV = np.sum(diff_counts[mask] * centers[mask])

        excess_ratio = excess_energy_MeV / total_energy_MeV

        return {
            'T_keV': T_keV,
            'excess_ratio': excess_ratio,
            'total_excess_MeV': excess_energy_MeV,
            'total_energy_MeV': total_energy_MeV,
            'threshold_MeV': threshold_energy_MeV
        }

    # =========================================================================
    # 2. 运行与绘图
    # =========================================================================

    def run(self, loaded_runs: List[SimulationRun]):
        style = get_style()  # 获取当前激活的样式
        console.print("\n[bold magenta]执行: 算法底噪分析 (T=0 vs T=End)...[/bold magenta]")

        # 过滤掉没有初始谱的数据
        valid_runs = []
        for r in loaded_runs:
            if r.final_spectrum and r.initial_spectrum:
                valid_runs.append(r)
            else:
                console.print(f"[yellow]警告: 模拟 {r.name} 缺少 initial_spectrum 或 final_spectrum，已跳过。[/yellow]")

        if len(valid_runs) < 1:
            console.print("[red]错误: 没有足够的数据进行对比。请确保模拟运行时保存了 Step 0 数据。[/red]")
            return

        # 1. 使用 Selector
        selector = ParameterSelector(valid_runs)
        x_label, x_vals, sorted_runs = selector.select()

        # 2. 生成文件名
        final_filename = selector.generate_filename(x_label, sorted_runs, prefix="debug_tail")

        # 数据容器
        y_ratio_init = []
        y_ratio_final = []
        y_temp_init = []
        y_temp_final = []

        console.print(f"  正在计算 Initial (底噪) 与 Final (信号) ...")

        for i, run in enumerate(sorted_runs):
            m_init = self._analyze_spectrum_excess(run.initial_spectrum)
            m_final = self._analyze_spectrum_excess(run.final_spectrum)

            y_ratio_init.append(m_init['excess_ratio'])
            y_ratio_final.append(m_final['excess_ratio'])
            y_temp_init.append(m_init['T_keV'])
            y_temp_final.append(m_final['T_keV'])

            console.print(f"    [{run.name}] {x_label}={x_vals[i]}")
            console.print(f"      Initial(T=0):  Excess={m_init['excess_ratio'] * 100:6.3f}% (Noise), T={m_init['T_keV']:.2f} keV")
            console.print(f"      Final  (T=end): Excess={m_final['excess_ratio'] * 100:6.3f}% (Signal), T={m_final['T_keV']:.2f} keV")

        # 绘图
        try:
            x_num = [float(v) for v in x_vals]
            is_num = True
        except:
            x_num = range(len(x_vals))
            is_num = False

        with create_analysis_figure(sorted_runs, "debug_tail", num_plots=2, override_filename=final_filename) as (fig, (ax1, ax2)):

            # --- 图1: 信号 vs 底噪 ---
            # 使用 style.color_comparison_primary (信号) 和 style.color_baseline_secondary (底噪)
            ax1.plot(x_num, np.array(y_ratio_final) * 100,
                     marker='o', linestyle=style.ls_primary,
                     color=style.color_comparison_primary, lw=style.lw_base,
                     label='最终能谱')

            # 我们在这里改了算法了，不需要底噪部分了。
            # ax1.plot(x_num, np.array(y_ratio_init) * 100,
            #          marker='o', linestyle=style.ls_secondary,
            #          color=style.color_baseline_secondary, lw=style.lw_base,
            #          label='Initial (Noise Floor)')

            # 填充差值
            ax1.fill_between(x_num, np.array(y_ratio_init) * 100, np.array(y_ratio_final) * 100,
                             color=style.color_comparison_primary, alpha=0.1)

            ax1.set_ylabel("非热成分 (%)")
            ax1.legend()
            ax1.grid(True, linestyle='--', alpha=0.5)

            # --- 图2: 温度对比 ---
            ax2.plot(x_num, y_temp_final,
                     marker='s', linestyle=style.ls_primary,
                     color=style.color_comparison_secondary, lw=style.lw_base,
                     label='最终温度 $T$')

            ax2.plot(x_num, y_temp_init,
                     marker='s', linestyle=style.ls_secondary,
                     color=style.color_baseline_secondary, lw=style.lw_base,
                     label='初始温度 $T$')

            ax2.set_ylabel("$T_{eff}$ (keV)")

            # TODO 这里是临时的，之后改成一个映射表的形式
            if x_label == "target_sigma":
                x_label_name = "磁场能量占比 $\sigma$"
            else:
                x_label_name = x_label

            ax2.set_xlabel(x_label_name if is_num else "模拟案例")
            ax2.legend()
            ax2.grid(True, linestyle='--', alpha=0.5)

            # 处理非数值坐标轴
            if not is_num:
                ax1.set_xticks(x_num)
                ax1.set_xticklabels(x_vals, rotation=45)
                ax2.set_xticks(x_num)
                ax2.set_xticklabels(x_vals, rotation=45)

            # 注意：不调用 set_title，不手动指定 fontsize，这些交给 styles.py 和 LaTeX
            plt.subplots_adjust(hspace=0.3)

            console.print("\n[bold green]分析完成。[/bold green]")
            console.print("如果 'Initial Spectrum' 的灰色虚线很高(例如 > 1%)，说明当前的正向差值算法(Positive Excess)受统计涨落影响严重。")
            console.print("建议：改用 Quantile (分位数) 分析或更高阶的拟合方法。")
