# analysis/modules/single/tail_statisticsV2.py

import warnings
from typing import List, Dict, Optional

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
def compute_run_tail_metrics(
        run: 'SimulationRunSingle',
        is_final: bool,
        f_low: float,
        f_high: Optional[float] = None
) -> Dict[str, float]:
    """
    计算单个 Run 在指定能量区间 [f_low*T, f_high*T] 内的物理度量。并执行极其严谨的统计学误差传递。

    参数:
        run: 单次模拟数据对象
        is_final: 是否为最终时刻
        f_low: 区间下限倍数 (E > f_low * T)
        f_high: 区间上限倍数 (E < f_high * T)，若为 None 则表示到正无穷
    """
    step_index = -1 if is_final else 0
    spec = run.get_spectrum(step_index)

    if spec is None or spec.weights.size == 0:
        return {'T_keV': 0.0, 'excess_ratio': 0.0, 'propagated_uncertainty': 0.0, 'threshold_MeV': 0.0}

    # =========================================================================
    # 0. 精确基础统计与有效粒子数 (Effective Sample Size)
    # =========================================================================
    total_energy_MeV = np.sum(spec.energies_MeV * spec.weights)
    total_weight = np.sum(spec.weights)

    if total_weight == 0 or total_energy_MeV <= 0:
        return {'T_keV': 0.0, 'excess_ratio': 0.0, 'propagated_uncertainty': 0.0, 'threshold_MeV': 0.0}

    avg_energy_MeV = total_energy_MeV / total_weight

    # PIC 加权统计中的关键：有效粒子数 N_eff
    V1 = total_weight
    V2 = np.sum(spec.weights ** 2)
    if V2 > 0:
        N_eff = (V1 ** 2) / V2
    else:
        N_eff = 1.0

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

    max_sim_energy_MeV = np.max(spec.energies_MeV)

    # =========================================================================
    # 3. 步骤三：定义理论尾部能量函数，并计算其误差 sigma_Eth
    # =========================================================================
    def compute_theoretical_tail_energy(T_val: float) -> float:
        e_low = (f_low * T_val) / 1000.0
        # 如果没有上限，或者上限超过了模拟最大能量，则积分到最大能量
        if f_high is None:
            e_high = max_sim_energy_MeV
        else:
            e_high = (f_high * T_val) / 1000.0

        def integrand(e):
            return e * physics_mj.calculate_mj_pdf(np.array([e]), T_val)[0]

        def pdf_func(e):
            return physics_mj.calculate_mj_pdf(np.array([e]), T_val)[0]

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=IntegrationWarning)
            quad_result = quad(integrand, e_low, e_high, limit=200)[0]
            prob_norm = quad(pdf_func, 0, max_sim_energy_MeV, limit=200)[0]

        if prob_norm <= 0: return 0.0
        return (quad_result / prob_norm) * total_weight

    # 基准理论能量与由于温度波动引起的理论误差
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
    e_low_th = (f_low * T_keV) / 1000.0
    e_high_th = (f_high * T_keV / 1000.0) if f_high else float('inf')

    mask = (spec.energies_MeV >= e_low_th) & (spec.energies_MeV < e_high_th)

    if not np.any(mask):
        sim_tail_energy_MeV = 0.0
        sigma_sim_tail_energy = 0.0
    else:
        sim_tail_energy_MeV = np.sum(spec.energies_MeV[mask] * spec.weights[mask])
        # 核心：加权蒙特卡洛计数的方差公式 Var = sum( (W_i * E_i)^2 )
        var_sim_tail_energy = np.sum((spec.energies_MeV[mask] * spec.weights[mask]) ** 2)
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
        'threshold_low_MeV': e_low_th,
        'threshold_high_MeV': e_high_th if f_high else max_sim_energy_MeV
    }


class MultiBandTailStatisticsModule(BaseComparisonModule):
    @property
    def name(self) -> str:
        return "多能段高能尾部分析 (Multi-Band Exact Integration)"

    @property
    def description(self) -> str:
        return "扫描多个能量区间（如 1-3T, 3-5T, 5-10T），全面评估不同组分的高能非热信号与理论底噪。"

    def __init__(self):
        super().__init__()
        # 定义要扫描的区间列表 (low, high)。None 表示正无穷
        self.intervals = [
            (1.0, 2.0),
            (2.0, 3.0),
            (3.0, 5.0),
            (5.0, 10.0),
            (10.0, 15.0),
            (15.0, None)
        ]

    def _get_metrics_with_error(self, run_or_group, is_final: bool, low: float, high: Optional[float]) -> Dict[str, float]:
        """
        核心分发器...
        """
        from analysis.core.simulationGroup import SimulationRunGroup

        if isinstance(run_or_group, SimulationRunGroup):
            t_list, ratio_list, th_err_list = [], [], []
            for single_run in run_or_group.runs:
                # 这里会光速命中缓存，返回极小的字典！
                metrics = compute_run_tail_metrics(single_run, is_final, low, high)
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
            metrics = compute_run_tail_metrics(run_or_group, is_final, low, high)
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
        console.print(f"\n[bold magenta]执行: {self.name}...[/bold magenta]")

        valid_runs = filter_valid_runs(loaded_runs, require_particles=True, min_particle_files=2)
        if len(valid_runs) < 1:
            console.print("[red]错误: 没有足够的数据进行对比。[/red]")
            return

        selector = ParameterSelector(valid_runs)
        x_label, x_vals, sorted_runs = selector.select()
        final_filename = selector.generate_filename(x_label, sorted_runs, prefix="multiband_tail")

        # 数据结构初始化
        # results[factor] = {'init': [], 'init_err': [], 'init_th_err': [], 'final': [], 'final_err': [], 'final_th_err': []}
        results = {i: {'init': [], 'init_err': [], 'init_th_err': [],
                       'final': [], 'final_err': [], 'final_th_err': []}
                   for i in range(len(self.intervals))}

        # 温度不随 tail_factor 改变，只需存一份
        temps = {'init': [], 'init_err': [], 'final': [], 'final_err': []}

        console.print("  正在扫描多能段 Initial 与 Final 数据 ...")

        for i, run in enumerate(sorted_runs):
            console.print(f"    [{run.name}] {x_label}={x_vals[i]}")

            # 对各个能段进行遍历计算
            for f_idx, (low, high) in enumerate(self.intervals):
                m_init = self._get_metrics_with_error(run, is_final=False, low=low, high=high)
                m_final = self._get_metrics_with_error(run, is_final=True, low=low, high=high)

                # 只在第一个 factor 时记录温度
                if f_idx == 0:
                    temps['init'].append(m_init['T_keV'])
                    temps['init_err'].append(m_init['T_keV_err'])
                    temps['final'].append(m_final['T_keV'])
                    temps['final_err'].append(m_final['T_keV_err'])

                results[f_idx]['init'].append(m_init['excess_ratio'])
                results[f_idx]['init_err'].append(m_init['excess_ratio_err'])
                results[f_idx]['init_th_err'].append(m_init['propagated_uncertainty'])

                results[f_idx]['final'].append(m_final['excess_ratio'])
                results[f_idx]['final_err'].append(m_final['excess_ratio_err'])
                results[f_idx]['final_th_err'].append(m_final['propagated_uncertainty'])

        # X 轴处理
        try:
            x_num = [float(v) for v in x_vals]
            is_num = True
        except ValueError:
            x_num = range(len(x_vals))
            is_num = False

        num_plots = len(self.intervals) + 1

        # =========================================================================
        # 绘图逻辑：N个能段的 Excess 占比图 + 1个温度图
        # =========================================================================

        with create_analysis_figure(sorted_runs, "multiband_tail", num_plots=num_plots, override_filename=final_filename, figsize=(10, 3.5 * num_plots)) as (
                fig, axes):
            x_arr = np.array(x_num)

            # 确保 axes 是可迭代的列表
            if num_plots == 1: axes = [axes]

            # --- 图 1 到 N: 各能段 Excess Ratio ---
            for i, (low, high) in enumerate(self.intervals):
                ax = axes[i]
                d = results[i]
                label_str = f"${low}T < E < {high}T$" if high else f"$E > {low}T$"

                # 绘制理论误差带 (阴影)
                # ax.fill_between(x_arr,
                #                 (np.array(d['init']) - np.array(d['init_th_err'])) * 100,
                #                 (np.array(d['init']) + np.array(d['init_th_err'])) * 100,
                #                 color=style.color_baseline_secondary, alpha=0.15,
                #                 label='初始理论散粒误差 (Shot Noise)')

                ax.fill_between(x_arr,
                                (np.array(d['final']) - np.array(d['final_th_err'])) * 100,
                                (np.array(d['final']) + np.array(d['final_th_err'])) * 100,
                                color=style.color_comparison_primary, alpha=0.1,
                                label='最终理论传递误差')

                # 绘制实测值误差棒
                ax.errorbar(x_num, np.array(d['final']) * 100, yerr=np.array(d['final_err']) * 100,
                            fmt='-o', capsize=4, elinewidth=1.5,
                            color=style.color_comparison_primary, label='最终时刻 $t_{end}$')

                # ax.errorbar(x_num, np.array(d['init']) * 100, yerr=np.array(d['init_err']) * 100,
                #             fmt='--x', capsize=4, alpha=0.7,
                #             color=style.color_baseline_secondary, label='初始时刻 $t=0$')

                ax.axhline(0, color='black', linestyle='-', alpha=0.3, lw=1)
                ax.set_ylabel(f"Excess %\n({label_str})")
                ax.legend(fontsize='x-small', loc='upper left', ncol=2)
                ax.grid(True, linestyle=':', alpha=0.4)

                # 隐藏非底部的 X 轴标签以保持整洁
                if not is_num:
                    ax.set_xticks(x_num)
                    if i < num_plots - 1:
                        ax.set_xticklabels([])

            # --- 最后一个图: 温度变化 ---
            ax_temp = axes[-1]
            ax_temp.errorbar(x_num, temps['final'], yerr=np.array(temps['final_err']),
                             fmt='-s', capsize=4, color=style.color_comparison_secondary, lw=style.lw_base, label='最终温度 $T$')

            # ax_temp.errorbar(x_num, temps['init'], yerr=np.array(temps['init_err']),
            #                  fmt='--s', capsize=4, color=style.color_baseline_secondary, lw=style.lw_base, label='初始温度 $T$')

            ax_temp.set_ylabel("$T_{eff}$ (keV)")
            x_label_name = "磁场能量占比 $\sigma$" if x_label == "target_sigma" else x_label
            ax_temp.set_xlabel(x_label_name if is_num else "模拟案例")
            ax_temp.legend(fontsize='small', loc='best')
            ax_temp.grid(True, linestyle='--', alpha=0.5)

            if not is_num:
                ax_temp.set_xticks(x_num)
                ax_temp.set_xticklabels(x_vals, rotation=45)

            plt.subplots_adjust(hspace=0.25)

            console.print("\n[bold green]分析完成。[/bold green]")
            console.print("提示：已生成覆盖多组分的高能尾部演化图谱。")
            console.print(
                "可以观察：在较高阈值（如 >5T, >8T）下，初始底噪占比更低但相对散粒误差会被放大，\n如果在该能段信号显著脱离了阴影带，则证明产生了确凿的高能尾部物理加速机制。")
            console.print("理论上，Initial (Noise Floor) 的点应该非常紧密地围绕在 [bold yellow]0% (y=0 虚线)[/bold yellow] 上下均匀波动。")
