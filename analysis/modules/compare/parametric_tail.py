# analysis/modules/parametric_tail.py

from typing import List, Dict

import matplotlib.pyplot as plt
import numpy as np

from analysis.core.parameter_selector import ParameterSelector
from analysis.core.simulation import SimulationRun
from analysis.core.utils import console
from analysis.modules.abstract.base_module import BaseComparisonModule
from analysis.modules.utils import physics_mj
from analysis.modules.utils.spectrum_tools import filter_valid_runs
from analysis.plotting.layout import create_analysis_figure


class ParametricTailModule(BaseComparisonModule):
    @property
    def name(self) -> str:
        return "参数扫描：高能尾巴与加热效率"

    @property
    def description(self) -> str:
        return "自动识别变化的输入参数(X轴)，绘制高能尾部能量占比和等效温度的变化趋势。"

    # =========================================================================
    # 物理计算核心
    # =========================================================================

    def _calculate_excess_energy(self, run: SimulationRun) -> Dict[str, float]:
        """
        核心算法改进：
        1. 建立对数分箱。
        2. 计算每个箱内的 模拟粒子数 N_sim 和 理论粒子数 N_th。
        3. 计算差值 ΔN = max(0, N_sim - N_th)。
        4. 加权求和：Sum( ΔN * E_center )。
        """
        spec = run.final_spectrum
        if spec is None or spec.weights.size == 0:
            return {'T_keV': 0.0, 'excess_ratio': 0.0, 'total_excess_MeV': 0.0}

        # 1. 基础统计与温度拟合
        total_energy_MeV = np.sum(spec.energies_MeV * spec.weights)
        total_weight = np.sum(spec.weights)
        avg_energy_MeV = total_energy_MeV / total_weight
        T_keV = physics_mj.solve_mj_temperature_kev(avg_energy_MeV)

        # 2. 建立分箱 (覆盖整个范围，从极低到极高)
        min_e = max(1e-4, spec.energies_MeV.min())
        max_e = max(10.0, spec.energies_MeV.max() * 1.5)  # 确保覆盖高能尾
        bins = np.logspace(np.log10(min_e), np.log10(max_e), 200)
        centers = np.sqrt(bins[:-1] * bins[1:])
        widths = np.diff(bins)

        # 3. 模拟数据的直方图
        counts_sim, _ = np.histogram(spec.energies_MeV, bins=bins, weights=spec.weights)

        # 4. 理论数据的直方图
        # N_th(bin) ≈ PDF(center) * width * total_weight
        pdf_vals = physics_mj.calculate_mj_pdf(centers, T_keV)
        counts_th = pdf_vals * widths * total_weight

        # 5. 核心：计算加权正向差值 (Weighted Positive Excess)
        # 只有当 sim > th 时才计入，且乘以能量 E 进行加权
        diff_counts = counts_sim - counts_th

        # 过滤掉负值 (即模拟 < 理论的部分不扣分)
        positive_diff = np.maximum(0.0, diff_counts)

        # 能量加权积分：Sum ( ΔN * E )
        excess_energy_MeV = np.sum(positive_diff * centers)

        # 6. 归一化指标
        # 非热能量占比 = 溢出的能量 / 总能量
        excess_ratio = excess_energy_MeV / total_energy_MeV

        return {
            'T_keV': T_keV,
            'excess_ratio': excess_ratio,
            'total_excess_MeV': excess_energy_MeV,
            'total_energy_MeV': total_energy_MeV
        }

    # =========================================================================
    # 运行与绘图
    # =========================================================================

    def run(self, loaded_runs: List[SimulationRun]):
        console.print("\n[bold magenta]执行: 参数扫描非热能量分析 (正向差值法)...[/bold magenta]")

        # 1. 准备数据
        valid_runs = filter_valid_runs(loaded_runs, require_particles=True, min_particle_files=2)
        if len(valid_runs) < 2:
            console.print("[red]错误: 需要至少 2 个模拟来进行对比。[/red]")
            return

        selector = ParameterSelector(valid_runs)
        x_label, x_vals, sorted_runs = selector.select()

        # 2. 生成文件名
        final_filename = selector.generate_filename(x_label, sorted_runs, prefix="scan_excess")

        # 3. 循环计算物理量
        y_ratio = []
        y_energy = []
        y_temp = []

        console.print(f"  正在逐个能箱计算 (Sim - Theory) * Energy ...")
        for i, run in enumerate(sorted_runs):
            m = self._calculate_excess_energy(run)
            y_ratio.append(m['excess_ratio'])
            y_energy.append(m['total_excess_MeV'])
            y_temp.append(m['T_keV'])

            console.print(f"    - {run.name} ({x_label}={x_vals[i]}): "
                          f"Excess Ratio={m['excess_ratio'] * 100:.2f}%, T={m['T_keV']:.1f} keV")

        # 4. 绘图
        try:
            x_num = [float(v) for v in x_vals]
            is_num = True
        except:
            x_num = range(len(x_vals))
            is_num = False

        with create_analysis_figure(sorted_runs, "scan_excess", num_plots=2, figsize=(9, 8), override_filename=final_filename) as (fig, (ax1, ax2)):

            # --- 图1: 非热能量占比 ---
            # 这是最有物理意义的图：到底有多少比例的能量进入了非热部分
            ax1.plot(x_num, np.array(y_ratio) * 100, 'o-', color='crimson', lw=2, markersize=6)
            ax1.set_ylabel(r"非热能量占比 (%)" + "\n" + r"$\sum (N_{sim}-N_{th})E / E_{total}$")
            ax1.set_title(f"非热加速效率 vs {x_label}", fontsize=14)
            ax1.grid(True, linestyle='--', alpha=0.5)

            # --- 图2 (放在下半部分): 背景温度 ---
            # 用于区分是“整体加热”还是“尾部加速”
            ax2.plot(x_num, y_temp, 's-', color='darkorange', lw=2, markersize=6)
            ax2.set_ylabel("整体等效温度 $T_{fit}$ (keV)")
            ax2.set_title(f"背景加热效果 vs {x_label}", fontsize=14)
            ax2.set_xlabel(x_label if is_num else "Simulation Case", fontsize=12)
            ax2.grid(True, linestyle='--', alpha=0.5)

            if not is_num:
                ax1.set_xticks(x_num)
                ax1.set_xticklabels(x_vals, rotation=45)
                ax2.set_xticks(x_num)
                ax2.set_xticklabels(x_vals, rotation=45)

            plt.subplots_adjust(hspace=0.3)
