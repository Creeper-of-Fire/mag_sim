# analysis/modules/single/tail_statisticsV2.py

import warnings
from typing import List, Dict

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad, IntegrationWarning

from analysis.core.cache import cached_op
from analysis.core.parameter_selector import ParameterSelector
from analysis.core.simulation import SimulationRun
from analysis.core.simulationSingle import SimulationRunSingle
from analysis.core.utils import console
from analysis.modules.abstract.base_module import BaseComparisonModule
from analysis.modules.utils import physics_mj
from analysis.modules.utils.spectrum_tools import filter_valid_runs
from analysis.plotting.layout import create_analysis_figure
from analysis.plotting.styles import get_style


@cached_op(file_dep="particle")
def compute_run_tail_metrics(run: 'SimulationRunSingle', is_final: bool) -> Dict[str, float]:
    """
    计算单个 Run 的物理度量，并执行极其严谨的统计学误差传递 (Error Propagation)。
    结果将被小巧的字典缓存。
    """
    step_index = -1 if is_final else 0
    spec = run.get_spectrum(step_index)

    if spec is None or spec.weights.size == 0:
        return {'T_keV': 0.0, 'excess_ratio': 0.0, 'propagated_uncertainty': 0.0}

    # =========================================================================
    # 0. 精确基础统计与有效粒子数 (Effective Sample Size)
    # =========================================================================
    total_energy_MeV = np.sum(spec.energies_MeV * spec.weights)
    total_weight = np.sum(spec.weights)

    if total_weight == 0 or total_energy_MeV <= 0:
        return {'T_keV': 0.0, 'excess_ratio': 0.0, 'propagated_uncertainty': 0.0}

    avg_energy_MeV = total_energy_MeV / total_weight

    # PIC 加权统计中的关键：有效粒子数 N_eff
    V1 = total_weight
    V2 = np.sum(spec.weights ** 2)
    N_eff = (V1 ** 2) / V2 if V2 > 0 else 1.0

    # =========================================================================
    # 1. 步骤一：计算平均能量的标准误 sigma_<E>
    # =========================================================================
    # 加权能量方差 Var(E)
    var_E = np.sum(spec.weights * (spec.energies_MeV - avg_energy_MeV) ** 2) / V1
    # 平均值的标准误
    sigma_avg_E = np.sqrt(var_E / N_eff)

    # =========================================================================
    # 2. 步骤二：误差传递到 MJ 温度 sigma_T
    # =========================================================================
    T_keV = physics_mj.solve_mj_temperature_kev(avg_energy_MeV)

    # 使用数值微积分 (中心差分法) 计算导数 |dT / d<E>|
    delta_E = avg_energy_MeV * 1e-4  # 取 0.01% 作为微小扰动
    T_plus = physics_mj.solve_mj_temperature_kev(avg_energy_MeV + delta_E, guess_T_keV=T_keV)
    T_minus = physics_mj.solve_mj_temperature_kev(avg_energy_MeV - delta_E, guess_T_keV=T_keV)

    dT_dE = (T_plus - T_minus) / (2 * delta_E)
    sigma_T = abs(dT_dE) * sigma_avg_E

    # 设定基准阈值
    threshold_energy_MeV = (3.0 * T_keV) / 1000.0
    max_sim_energy_MeV = np.max(spec.energies_MeV)

    # =========================================================================
    # 3. 步骤三：定义理论尾部能量函数，并计算其误差 sigma_Eth
    # =========================================================================
    def compute_theoretical_tail_energy(T_val: float) -> float:
        """封装积分逻辑，以便于进行数值求导"""
        th_E = (3.0 * T_val) / 1000.0

        def integrand(e): return e * physics_mj.calculate_mj_pdf(np.array([e]), T_val)[0]

        def pdf_func(e):  return physics_mj.calculate_mj_pdf(np.array([e]), T_val)[0]

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=IntegrationWarning)
            quad_result = quad(integrand, th_E, max_sim_energy_MeV, limit=200)[0]
            prob_norm = quad(pdf_func, 0, max_sim_energy_MeV, limit=200)[0]

        if prob_norm <= 0: return 0.0
        return (quad_result / prob_norm) * total_weight

    # 计算基准理论尾部能量
    th_tail_energy_MeV = compute_theoretical_tail_energy(T_keV)

    # 数值求导 |dE_th / dT|
    delta_T = T_keV * 1e-4
    th_tail_plus = compute_theoretical_tail_energy(T_keV + delta_T)
    th_tail_minus = compute_theoretical_tail_energy(T_keV - delta_T)
    dEth_dT = (th_tail_plus - th_tail_minus) / (2 * delta_T)

    sigma_th_tail_energy = abs(dEth_dT) * sigma_T

    # =========================================================================
    # 4. 步骤四：计算模拟尾部的泊松/散粒散布误差 sigma_Esim
    # =========================================================================
    tail_mask = spec.energies_MeV > threshold_energy_MeV
    if not np.any(tail_mask):
        sim_tail_energy_MeV = 0.0
        sigma_sim_tail_energy = 0.0
    else:
        sim_tail_energy_MeV = np.sum(spec.energies_MeV[tail_mask] * spec.weights[tail_mask])
        # 核心：加权蒙特卡洛计数的方差公式 Var = sum( (W_i * E_i)^2 )
        var_sim_tail_energy = np.sum((spec.energies_MeV[tail_mask] * spec.weights[tail_mask]) ** 2)
        sigma_sim_tail_energy = np.sqrt(var_sim_tail_energy)

    # =========================================================================
    # 5. 步骤五：合成最终的理论 Excess 误差
    # =========================================================================
    excess_energy_MeV = sim_tail_energy_MeV - th_tail_energy_MeV
    excess_ratio = excess_energy_MeV / total_energy_MeV

    # 独立误差平方和开根号
    sigma_excess_energy = np.sqrt(sigma_sim_tail_energy ** 2 + sigma_th_tail_energy ** 2)
    # 转换为占比的相对误差
    propagated_uncertainty = sigma_excess_energy / total_energy_MeV

    return {
        'T_keV': T_keV,
        'excess_ratio': excess_ratio,
        'propagated_uncertainty': propagated_uncertainty,  # <--- 这是我们手算出来的终极理论误差！
        'total_excess_MeV': excess_energy_MeV,
        'total_energy_MeV': total_energy_MeV,
        'threshold_MeV': threshold_energy_MeV
    }


class ParametricTailDebugModule(BaseComparisonModule):
    @property
    def name(self) -> str:
        return "DEBUG V2：无分箱精确算法底噪分析"

    @property
    def description(self) -> str:
        return "使用解析积分与无分箱精确求和，对比(t=0)与最终时刻的非热能量，量化纯统计涨落。"

    def _get_metrics_with_error(self, run_or_group, is_final: bool) -> Dict[str, float]:
        """
        核心分发器...
        """
        from analysis.core.simulationGroup import SimulationRunGroup

        if isinstance(run_or_group, SimulationRunGroup):
            t_list, ratio_list, th_err_list = [], [], []
            for single_run in run_or_group.runs:
                # 这里会光速命中缓存，返回极小的字典！
                metrics = compute_run_tail_metrics(single_run, is_final)
                t_list.append(metrics['T_keV'])
                ratio_list.append(metrics['excess_ratio'])
                th_err_list.append(metrics['propagated_uncertainty'])

            # 计算平均值和标准差（Error Bar）
            return {
                'T_keV': np.mean(t_list),
                'T_keV_err': np.std(t_list, ddof=1) if len(t_list) > 1 else 0.0,
                'excess_ratio': np.mean(ratio_list),
                'excess_ratio_err': np.std(ratio_list, ddof=1) if len(ratio_list) > 1 else 0.0,
                # 把理论手算误差也平均一下传出去
                'propagated_uncertainty': np.mean(th_err_list)
            }
        else:
            # 它是单次模拟
            metrics = compute_run_tail_metrics(run_or_group, is_final)
            return {
                'T_keV': metrics['T_keV'],
                'T_keV_err': 0.0,
                'excess_ratio': metrics['excess_ratio'],
                'excess_ratio_err': 0.0,
                'propagated_uncertainty': metrics['propagated_uncertainty']
            }

    # =========================================================================
    # 运行与绘图
    # =========================================================================

    def run(self, loaded_runs: List[SimulationRun]):
        style = get_style()
        console.print("\n[bold magenta]执行: 算法底噪分析 V2 (无分箱完美积分)...[/bold magenta]")

        valid_runs = filter_valid_runs(loaded_runs, require_particles=True, min_particle_files=2)
        if len(valid_runs) < 1:
            console.print("[red]错误: 没有足够的数据进行对比。[/red]")
            return

        selector = ParameterSelector(valid_runs)
        x_label, x_vals, sorted_runs = selector.select()
        final_filename = selector.generate_filename(x_label, sorted_runs, prefix="debug_tail_v2")

        y_ratio_init, y_ratio_final = [], []
        y_temp_init, y_temp_final = [], []

        y_ratio_final_err, y_ratio_init_err = [], []
        y_ratio_init_th_err, y_ratio_final_th_err = [], []
        y_temp_init_err, y_temp_final_err = [], []

        console.print("  正在计算 Initial (底噪) 与 Final (信号) ...")

        for i, run in enumerate(sorted_runs):
            m_init = self._get_metrics_with_error(run, is_final=False)
            m_final = self._get_metrics_with_error(run, is_final=True)

            y_ratio_init.append(m_init['excess_ratio'])
            y_ratio_init_err.append(m_init['excess_ratio_err'])
            y_ratio_init_th_err.append(m_init['propagated_uncertainty'])

            y_ratio_final.append(m_final['excess_ratio'])
            y_ratio_final_err.append(m_final['excess_ratio_err'])
            y_ratio_final_th_err.append(m_final['propagated_uncertainty'])

            y_temp_init.append(m_init['T_keV'])
            y_temp_init_err.append(m_init['T_keV_err'])

            y_temp_final.append(m_final['T_keV'])
            y_temp_final_err.append(m_final['T_keV_err'])

            console.print(f"    [{run.name}] {x_label}={x_vals[i]}")

            console.print(
                f"Initial(T=0)  : Excess = {m_init['excess_ratio'] * 100:>7.4f}% ± {m_init['excess_ratio_err'] * 100:.4f}% (实) | 理论不确定度 = ± {m_init['propagated_uncertainty'] * 100:.4f}%")
            console.print(
                f"Final  (T=end): Excess = {m_final['excess_ratio'] * 100:>7.4f}% ± {m_final['excess_ratio_err'] * 100:.4f}% (实) | 理论不确定度 = ± {m_final['propagated_uncertainty'] * 100:.4f}%")

        try:
            x_num = [float(v) for v in x_vals]
            is_num = True
        except ValueError:
            x_num = range(len(x_vals))
            is_num = False

        with create_analysis_figure(sorted_runs, "debug_tail_v2", num_plots=2, override_filename=final_filename) as (fig, (ax1, ax2)):

            x_arr = np.array(x_num)

            # --- 图1: 信号 vs 底噪 ---

            # 1. 先画【理论误差带】(用阴影表示，作为背景基准)
            # 初始时刻的理论底噪范围
            ax1.fill_between(x_arr,
                             (np.array(y_ratio_init) - np.array(y_ratio_init_th_err)) * 100,
                             (np.array(y_ratio_init) + np.array(y_ratio_init_th_err)) * 100,
                             color=style.color_baseline_secondary, alpha=0.15,
                             label='初始理论误差范围 (Shot Noise Limit)')

            # 最终时刻的理论误差范围
            ax1.fill_between(x_arr,
                             (np.array(y_ratio_final) - np.array(y_ratio_final_th_err)) * 100,
                             (np.array(y_ratio_final) + np.array(y_ratio_final_th_err)) * 100,
                             color=style.color_comparison_primary, alpha=0.1,
                             label='最终理论误差范围 (Propagated Error)')

            # 2. 再画【实测误差棒】
            # 最终时刻信号
            ax1.errorbar(x_num, np.array(y_ratio_final) * 100,
                         yerr=np.array(y_ratio_final_err) * 100,
                         fmt='-o',  # 线条+圆点
                         capsize=4,  # 误差棒两端的横线长度
                         elinewidth=1.5,
                         color=style.color_comparison_primary,
                         label='最终时刻 (实测均值 ± 1σ)')

            # 初始时刻底噪
            ax1.errorbar(x_num, np.array(y_ratio_init) * 100,
                         yerr=np.array(y_ratio_init_err) * 100,
                         fmt='--x',
                         capsize=4,
                         alpha=0.7,
                         color=style.color_baseline_secondary,
                         label='初始时刻 (实测均值 ± 1σ)')

            ax1.axhline(0, color='black', linestyle='-', alpha=0.3, lw=1)
            ax1.set_ylabel("额外高能能量占比 (%)")
            # 缩小图例字号防止挡住图
            ax1.legend(fontsize='small', loc='best')
            ax1.grid(True, linestyle=':', alpha=0.4)

            # --- 图2: 温度对比 ---
            # 最终温度曲线 + 误差棒
            ax2.errorbar(x_num, y_temp_final,
                         yerr=np.array(y_temp_final_err),
                         fmt='-s',  # 使用实线 + 方块
                         capsize=4,
                         color=style.color_comparison_secondary,
                         lw=style.lw_base,
                         label='最终温度 $T$ ± 1σ')

            # 初始温度曲线 + 误差棒
            ax2.errorbar(x_num, y_temp_init,
                         yerr=np.array(y_temp_init_err),
                         fmt='--s',  # 使用虚线 + 方块
                         capsize=4,
                         color=style.color_baseline_secondary,
                         lw=style.lw_base,
                         label='初始温度 $T$ ± 1σ')

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
