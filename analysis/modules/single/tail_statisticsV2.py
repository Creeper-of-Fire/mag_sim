# analysis/modules/parametric_tail_debug.py

import warnings
from typing import List, Dict

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad, IntegrationWarning

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
        return "DEBUG V2：无分箱精确算法底噪分析"

    @property
    def description(self) -> str:
        return "使用解析积分与无分箱精确求和，对比(t=0)与最终时刻的非热能量，量化纯统计涨落。"

    # =========================================================================
    # 物理计算核心 (V2: 完美积分方案)
    # =========================================================================

    def _analyze_spectrum_excess(self, spec: SpectrumData) -> Dict[str, float]:
        """
        基于无分箱精确求和与 Scipy 数值积分的非热成分分析。
        不再使用直方图，杜绝一切离散化带来的系统偏差。
        """
        if spec is None or spec.weights.size == 0:
            return {'T_keV': 0.0, 'excess_ratio': 0.0, 'total_excess_MeV': 0.0}

        # 1. 精确基础统计
        total_energy_MeV = np.sum(spec.energies_MeV * spec.weights)
        total_weight = np.sum(spec.weights)

        if total_weight == 0 or total_energy_MeV <= 0:
            return {'T_keV': 0.0, 'excess_ratio': 0.0, 'total_excess_MeV': 0.0}

        # 2. 温度拟合 (使用整体平均能量)
        # 既然是评估底噪，t=0 时整体就是热的，整体拟合是最无偏的。
        avg_energy_MeV = total_energy_MeV / total_weight
        T_keV = physics_mj.solve_mj_temperature_kev(avg_energy_MeV)

        # 设定阈值 (例如 3kT)
        threshold_energy_MeV = (3.0 * T_keV) / 1000.0

        # --- 3. 计算模拟数据在尾部(E > E_th)的真实总能量 (绝对精确，无分箱) ---
        tail_mask = spec.energies_MeV > threshold_energy_MeV
        if not np.any(tail_mask):
            sim_tail_energy_MeV = 0.0
        else:
            sim_tail_energy_MeV = np.sum(spec.energies_MeV[tail_mask] * spec.weights[tail_mask])

        # 获取模拟中粒子的真实截断上限 (非常关键！)
        max_sim_energy_MeV = np.max(spec.energies_MeV)

        # --- 4. 计算理论分布在尾部的期望总能量 (消除 PIC 截断误差) ---
        def integrand(e):
            return e * physics_mj.calculate_mj_pdf(np.array([e]), T_keV)[0]

        def pdf_func(e):
            return physics_mj.calculate_mj_pdf(np.array([e]), T_keV)[0]

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=IntegrationWarning)

            # 1. 积分上限改为 max_sim_energy_MeV，而不是 np.inf！
            quad_result = quad(integrand, threshold_energy_MeV, max_sim_energy_MeV, limit=200)
            theoretical_tail_energy_per_particle = quad_result[0]

            # 2. 概率补偿：因为模拟粒子在 max_sim_energy_MeV 处截断了，
            #    [0, max_sim_energy_MeV] 区间的概率和实际上 < 1。
            #    我们需要算出现在这个被截断的空间里，总概率是多少，进行重归一化。
            prob_norm = quad(pdf_func, 0, max_sim_energy_MeV, limit=200)[0]

            # 3. 得到完全无偏的理论期望能量
            th_tail_energy_MeV = (theoretical_tail_energy_per_particle / prob_norm) * total_weight

        # --- 5. 纯粹的做差 ---
        excess_energy_MeV = sim_tail_energy_MeV - th_tail_energy_MeV

        # 计算比例
        excess_ratio = excess_energy_MeV / total_energy_MeV

        # =====================================================================
        # 极严苛底层诊断 (Strict Diagnostics)
        # =====================================================================
        total_particles = spec.energies_MeV.size
        tail_particles = np.sum(tail_mask)

        # 使用 float64 极高精度计算残差
        sim_tail_val = np.float64(sim_tail_energy_MeV)
        th_tail_val = np.float64(th_tail_energy_MeV)
        diff_val = sim_tail_val - th_tail_val

        console.print(f"      [dim cyan]├─ 宏粒子总数: {total_particles} (尾部粒子数: {tail_particles})[/dim cyan]")
        console.print(f"      [dim cyan]├─ 绝对最大截断能量: {np.max(spec.energies_MeV):.6f} MeV[/dim cyan]")
        console.print(f"      [dim cyan]├─ 模拟尾部能量: {sim_tail_val:.8e} MeV[/dim cyan]")
        console.print(f"      [dim cyan]├─ 理论尾部能量: {th_tail_val:.8e} MeV[/dim cyan]")
        console.print(f"      [dim cyan]└─ 纯差值 (Sim - Th): {diff_val:.8e} MeV[/dim cyan]")

        return {
            'T_keV': T_keV,
            'excess_ratio': excess_ratio,
            'total_excess_MeV': excess_energy_MeV,
            'total_energy_MeV': total_energy_MeV,
            'threshold_MeV': threshold_energy_MeV
        }

    # =========================================================================
    # 2. 运行与绘图 (保持不变，仅更新了图例等细节)
    # =========================================================================

    def run(self, loaded_runs: List[SimulationRun]):
        style = get_style()
        console.print("\n[bold magenta]执行: 算法底噪分析 V2 (无分箱完美积分)...[/bold magenta]")

        valid_runs = [r for r in loaded_runs if r.final_spectrum and r.initial_spectrum]
        if len(valid_runs) < 1:
            console.print("[red]错误: 没有足够的数据进行对比。[/red]")
            return

        selector = ParameterSelector(valid_runs)
        x_label, x_vals, sorted_runs = selector.select()
        final_filename = selector.generate_filename(x_label, sorted_runs, prefix="debug_tail_v2")

        y_ratio_init, y_ratio_final = [], []
        y_temp_init, y_temp_final = [], []

        console.print("  正在计算 Initial (底噪) 与 Final (信号) ...")

        for i, run in enumerate(sorted_runs):
            m_init = self._analyze_spectrum_excess(run.initial_spectrum)
            m_final = self._analyze_spectrum_excess(run.final_spectrum)

            y_ratio_init.append(m_init['excess_ratio'])
            y_ratio_final.append(m_final['excess_ratio'])
            y_temp_init.append(m_init['T_keV'])
            y_temp_final.append(m_final['T_keV'])

            console.print(f"    [{run.name}] {x_label}={x_vals[i]}")
            # 加上正负号显示，方便观察涨落
            console.print(f"      Initial(T=0):  Excess={m_init['excess_ratio'] * 100:>7.4f}% (Noise), T={m_init['T_keV']:.2f} keV")
            console.print(f"      Final  (T=end): Excess={m_final['excess_ratio'] * 100:>7.4f}% (Signal), T={m_final['T_keV']:.2f} keV")

        try:
            x_num = [float(v) for v in x_vals]
            is_num = True
        except ValueError:
            x_num = range(len(x_vals))
            is_num = False

        with create_analysis_figure(sorted_runs, "debug_tail_v2", num_plots=2, override_filename=final_filename) as (fig, (ax1, ax2)):

            # --- 图1: 信号 vs 底噪 ---
            ax1.plot(x_num, np.array(y_ratio_final) * 100,
                     marker='o', linestyle=style.ls_primary,
                     color=style.color_comparison_primary, lw=style.lw_base,
                     label='最终时刻 (Signal)')

            # 将初始底噪重新画出来，看看是否在 0 上下波动
            ax1.plot(x_num, np.array(y_ratio_init) * 100,
                     marker='x', linestyle=style.ls_secondary,
                     color=style.color_baseline_secondary, lw=style.lw_base,
                     label='初始时刻 (Noise Floor)')

            # 画一条 y=0 的绝对基准线
            ax1.axhline(0, color='gray', linestyle=':', alpha=0.8, lw=1)

            ax1.set_ylabel("非热能量占比 (%)")
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
            x_label_name = "磁场能量占比 $\sigma$" if x_label == "target_sigma" else x_label
            ax2.set_xlabel(x_label_name if is_num else "模拟案例")
            ax2.legend()
            ax2.grid(True, linestyle='--', alpha=0.5)

            if not is_num:
                ax1.set_xticks(x_num)
                ax1.set_xticklabels(x_vals, rotation=45)
                ax2.set_xticks(x_num)
                ax2.set_xticklabels(x_vals, rotation=45)

            plt.subplots_adjust(hspace=0.3)

            console.print("\n[bold green]分析完成。[/bold green]")
            console.print("注意：当前版本使用了 [bold cyan]无分箱精确解 (Binless Exact Method)[/bold cyan]。")
            console.print("理论上，Initial (Noise Floor) 的点应该非常紧密地围绕在 [bold yellow]0% (y=0 虚线)[/bold yellow] 上下均匀波动。")