# analysis/modules/single/tail_statisticsV2.py

import warnings
from typing import List, Dict, Optional, NamedTuple, Any, Tuple

import h5py
import numpy as np
from scipy.constants import mu_0, e, epsilon_0
from scipy.integrate import quad, IntegrationWarning

from analysis.core.cache import cached_op
from analysis.core.data_loader import _get_step_from_filename
from analysis.core.simulation import SimulationRun
from analysis.core.simulationSingle import SimulationRunSingle
from analysis.core.utils import console
from analysis.modules.abstract.base_module import BaseComparisonModule
from analysis.modules.utils import physics_mj
from analysis.plotting.comparison_layout import ComparisonContext, ComparisonLayout
from analysis.plotting.styles import get_style


class TemperatureResult(NamedTuple):
    """光谱基础物理量及统计误差"""
    T_keV: float
    sigma_T: float
    total_weight: float
    total_energy_MeV: float
    avg_energy_MeV: float
    max_energy_MeV: float

    @staticmethod
    def null():
        return TemperatureResult(
            T_keV=0.0,
            sigma_T=0.0,
            total_weight=0.0,
            total_energy_MeV=0.0,
            avg_energy_MeV=0.0,
            max_energy_MeV=0.0
        )


@cached_op(file_dep="particle")
def compute_run_temperature_metrics(
        run: 'SimulationRunSingle',
        step_index: int
) -> TemperatureResult:
    """
    计算单个 Run 的温度和温度误差。并执行极其严谨的统计学误差传递。

    参数:
        run: 单次模拟数据对象
        step_index: 步骤的索引
    """
    spec = run.get_spectrum(step_index)

    if spec is None or spec.weights.size == 0:
        return TemperatureResult.null()

    # =========================================================================
    # 0. 精确基础统计与有效粒子数 (Effective Sample Size)
    # =========================================================================
    total_energy_MeV: Any = np.sum(spec.energies_MeV * spec.weights)
    total_weight = np.sum(spec.weights)

    if total_weight == 0 or total_energy_MeV <= 0:
        return TemperatureResult.null()

    avg_energy_MeV = total_energy_MeV / total_weight
    max_energy_MeV = np.max(spec.energies_MeV)

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

    return TemperatureResult(
        T_keV=T_keV,
        sigma_T=sigma_T,
        total_weight=total_weight,
        total_energy_MeV=total_energy_MeV,
        avg_energy_MeV=avg_energy_MeV,
        max_energy_MeV=max_energy_MeV
    )


class TailResult(NamedTuple):
    """特定能段的尾部超额指标"""
    excess_ratio: float
    propagated_uncertainty: float
    threshold_low_MeV: float
    threshold_high_MeV: float

    @staticmethod
    def null():
        return TailResult(
            excess_ratio=0.0,
            propagated_uncertainty=0.0,
            threshold_low_MeV=0.0,
            threshold_high_MeV=0.0
        )


@cached_op(file_dep="particle")
def compute_run_tail_metrics(
        run: 'SimulationRunSingle',
        step_index: int,
        temperature_metrics: TemperatureResult,
        f_low: float,
        f_high: Optional[float] = None
) -> TailResult:
    """
    计算单个 Run 在指定能量区间 [f_low*T, f_high*T] 内的物理度量。并执行极其严谨的统计学误差传递。

    参数:
        run: 单次模拟数据对象
        f_low: 区间下限倍数 (E > f_low * T)
        f_high: 区间上限倍数 (E < f_high * T)，若为 None 则表示到正无穷
    """
    spec = run.get_spectrum(step_index)

    T_keV = temperature_metrics.T_keV
    sigma_T = temperature_metrics.sigma_T
    total_weight = temperature_metrics.total_weight
    total_energy_MeV = temperature_metrics.total_energy_MeV
    max_sim_energy_MeV = temperature_metrics.max_energy_MeV

    if T_keV <= 0 or total_energy_MeV <= 0:
        return TailResult.null()

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

    return TailResult(
        excess_ratio=excess_ratio,
        propagated_uncertainty=propagated_uncertainty,  # <--- 这是我们手算出来的终极理论误差！
        threshold_low_MeV=e_low_th,
        threshold_high_MeV=e_high_th if f_high else max_sim_energy_MeV
    )


@cached_op(file_dep="auto")
def _get_mean_u_mag(run: 'SimulationRunSingle', fpath: str) -> float:
    """
    极速计算单个 HDF5 场文件中的全空间总磁场能量 (焦耳)。
    参数中包含 fpath，缓存会自动只绑定该单独文件，实现帧级别的精细复用。
    """
    step = _get_step_from_filename(fpath)
    with h5py.File(fpath, 'r') as f:
        bp = f"/data/{step}/fields/B"
        # 不管是 2D 还是 3D，直接对全数组取均值
        b_sq_mean = np.mean(f[bp + '/x'][:] ** 2 + f[bp + '/y'][:] ** 2 + f[bp + '/z'][:] ** 2)
    return float(b_sq_mean / (2 * mu_0))


@cached_op(file_dep="auto")
def _get_mean_u_elec(run: 'SimulationRunSingle', fpath: str) -> float:
    """计算全空间平均电场能量密度 (0.5 * epsilon_0 * E^2)"""
    step = _get_step_from_filename(fpath)
    with h5py.File(fpath, 'r') as f:
        ep = f"/data/{step}/fields/E"
        e_sq_mean = np.mean(f[ep + '/x'][:] ** 2 + f[ep + '/y'][:] ** 2 + f[ep + '/z'][:] ** 2)
    return float(0.5 * epsilon_0 * e_sq_mean)


def compute_run_energy_partition(run: 'SimulationRunSingle', step_index: int) -> Tuple[float, float, float]:
    """返回 (磁能占比, 电能占比, 总场能占比)"""
    t_metrics = compute_run_temperature_metrics(run, step_index)
    if t_metrics.avg_energy_MeV <= 0: return np.nan, np.nan, np.nan

    u_kin = (2.0 * run.sim.n_plasma) * (t_metrics.avg_energy_MeV * e * 1e6)  # TODO pair plasma 总密度是 2 * n_plasma，但是前提是我们没有引入beam
    files = run.field_files
    idx = step_index if step_index >= 0 else len(files) + step_index
    if not (0 <= idx < len(files)): return np.nan, np.nan, np.nan

    u_mag = _get_mean_u_mag(run, files[idx])
    u_elec = _get_mean_u_elec(run, files[idx])
    u_total = u_kin + u_mag + u_elec

    return float(u_mag / u_total), float(u_elec / u_total), float((u_mag + u_elec) / u_total)


def compute_run_energy_densities_normalized(run: 'SimulationRunSingle', step_index: int) -> Tuple[float, float, float, float]:
    """
    返回归一化能量密度 (以 n₀·mₑc² 为单位)
    """
    t_metrics = compute_run_temperature_metrics(run, step_index)
    if t_metrics.avg_energy_MeV <= 0:
        return 0.0, 0.0, 0.0, 0.0

    m_e_c2_J = 8.1871e-14  # 电子静止能量, J
    n0 = run.sim.n_plasma  # m^-3
    norm_factor = n0 * m_e_c2_J  # J/m^3, 归一化因子

    # 动能密度 (J/m^3) -> 归一化
    u_kin_J = (2.0 * n0) * (t_metrics.avg_energy_MeV * e * 1e6)
    u_kin_norm = u_kin_J / norm_factor

    files = run.field_files
    idx = step_index if step_index >= 0 else len(files) + step_index
    if not (0 <= idx < len(files)):
        return u_kin_norm, 0.0, 0.0, u_kin_norm

    u_mag_J = _get_mean_u_mag(run, files[idx])
    u_elec_J = _get_mean_u_elec(run, files[idx])

    u_mag_norm = u_mag_J / norm_factor
    u_elec_norm = u_elec_J / norm_factor
    u_total_norm = u_kin_norm + u_mag_norm + u_elec_norm

    return u_kin_norm, u_mag_norm, u_elec_norm, u_total_norm


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
            (0.0, 0.5),  # 低能区
            (0.5, 1.0),
            (1.0, 2.0),
            (2.0, 3.0),
            (3.0, 5.0),
            (5.0, 10.0),
            (10.0, 15.0),
            (15.0, None)
        ]

    def _get_field_metrics(self, run_or_group: 'SimulationRun', is_final: bool) -> Dict[str, Tuple[float, float]]:
        from analysis.core.simulationGroup import SimulationRunGroup
        step_idx = -1 if is_final else 0
        if isinstance(run_or_group, SimulationRunGroup):
            res = [compute_run_energy_partition(sr, step_idx) for sr in run_or_group.runs]
            m_vals = [v[0] for v in res if not np.isnan(v[0])]
            e_vals = [v[1] for v in res if not np.isnan(v[1])]
            s_vals = [v[2] for v in res if not np.isnan(v[2])]
            return {
                'mag': (float(np.mean(m_vals)), float(np.std(m_vals, ddof=1)) if len(m_vals) > 1 else 0.0),
                'elec': (float(np.mean(e_vals)), float(np.std(e_vals, ddof=1)) if len(e_vals) > 1 else 0.0),
                'sum': (float(np.mean(s_vals)), float(np.std(s_vals, ddof=1)) if len(s_vals) > 1 else 0.0)
            }
        else:
            m, e_v, s = compute_run_energy_partition(run_or_group, step_idx)
            return {'mag': (m, 0.0), 'elec': (e_v, 0.0), 'sum': (s, 0.0)}

    def _get_metrics_with_error(self, run_or_group: 'SimulationRun', is_final: bool, low: float, high: Optional[float]) -> Dict[str, float]:
        """
        核心分发器...
        """
        from analysis.core.simulationGroup import SimulationRunGroup

        if isinstance(run_or_group, SimulationRunGroup):
            t_list, ratio_list, th_err_list = [], [], []
            for single_run in run_or_group.runs:
                temp_metrics = compute_run_temperature_metrics(single_run, step_index=0 if not is_final else -1)
                metrics = compute_run_tail_metrics(
                    single_run,
                    step_index=0 if not is_final else -1,
                    temperature_metrics=temp_metrics,
                    f_low=low,
                    f_high=high
                )
                t_list.append(temp_metrics.T_keV)
                ratio_list.append(metrics.excess_ratio)
                th_err_list.append(metrics.propagated_uncertainty)

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
            step_idx = -1 if is_final else 0
            temp_metrics = compute_run_temperature_metrics(run_or_group, step_idx)
            metrics = compute_run_tail_metrics(run_or_group, step_idx, temp_metrics, low, high)
            return {
                'T_keV': temp_metrics.T_keV,
                'T_keV_err': 0.0,
                'excess_ratio': metrics.excess_ratio,
                'excess_ratio_err': 0.0,
                'propagated_uncertainty': metrics.propagated_uncertainty
            }

    def _get_energy_densities(self, run_or_group: 'SimulationRun', is_final: bool) -> Dict[str, Tuple[float, float]]:
        """获取各能量密度, 返回均值和标准差"""
        from analysis.core.simulationGroup import SimulationRunGroup
        step_idx = -1 if is_final else 0
        if isinstance(run_or_group, SimulationRunGroup):
            res = [compute_run_energy_densities_normalized(sr, step_idx) for sr in run_or_group.runs]
            kin_vals = [v[0] for v in res if not np.isnan(v[0])]
            mag_vals = [v[1] for v in res if not np.isnan(v[1])]
            elec_vals = [v[2] for v in res if not np.isnan(v[2])]
            tot_vals = [v[3] for v in res if not np.isnan(v[3])]

            def _mean_std(lst):
                if len(lst) == 0: return 0.0, 0.0
                m = float(np.mean(lst))
                s = float(np.std(lst, ddof=1)) if len(lst) > 1 else 0.0
                return m, s

            return {
                'kin': _mean_std(kin_vals),
                'mag': _mean_std(mag_vals),
                'elec': _mean_std(elec_vals),
                'total': _mean_std(tot_vals)
            }
        else:
            kin, mag, elec, tot = compute_run_energy_densities_normalized(run_or_group, step_idx)
            return {
                'kin': (kin, 0.0),
                'mag': (mag, 0.0),
                'elec': (elec, 0.0),
                'total': (tot, 0.0)
            }

    # =========================================================================
    # 运行与绘图
    # =========================================================================

    def run(self, loaded_runs: List[SimulationRun]):
        style = get_style()
        console.print(f"\n[bold magenta]执行: {self.name}...[/bold magenta]")

        ctx = ComparisonContext(loaded_runs, "multiband_tail")
        runs, x_scaled = ctx.unpack
        x_raw, _, x_label = ctx.x

        # 数据结构初始化
        # results[factor] = {'init': [], 'init_err': [], 'init_th_err': [], 'final': [], 'final_err': [], 'final_th_err': []}
        results = {i: {'init': [], 'init_err': [], 'init_th_err': [],
                       'final': [], 'final_err': [], 'final_th_err': []}
                   for i in range(len(self.intervals))}

        temps = {'init': [], 'init_err': [], 'final': [], 'final_err': []}
        mag_ratios = {'init': [], 'init_err': [], 'final': [], 'final_err': []}
        elec_ratios = {'init': [], 'init_err': [], 'final': [], 'final_err': []}
        sum_field_ratios = {'init': [], 'init_err': [], 'final': [], 'final_err': []}
        # 能量密度 (eV) 数据容器
        energy_densities = {
            'kin': {'final': [], 'final_err': []},
            'mag': {'final': [], 'final_err': []},
            'elec': {'final': [], 'final_err': []},
            'total': {'final': [], 'final_err': []}
        }

        console.print("  正在扫描多能段 Initial 与 Final 数据 ...")

        for i, run in enumerate(runs):
            console.print(f"    [{run.name}] {x_label}={x_raw[i]}")

            # 获取场比例
            fr_final = self._get_field_metrics(run, is_final=True)

            mag_ratios['final'].append(fr_final['mag'][0])
            mag_ratios['final_err'].append(fr_final['mag'][1])
            elec_ratios['final'].append(fr_final['elec'][0])
            elec_ratios['final_err'].append(fr_final['elec'][1])
            sum_field_ratios['final'].append(fr_final['sum'][0])
            sum_field_ratios['final_err'].append(fr_final['sum'][1])

            # 获取能量密度
            ed_final = self._get_energy_densities(run, is_final=True)
            for key in ['kin', 'mag', 'elec', 'total']:
                energy_densities[key]['final'].append(ed_final[key][0])
                energy_densities[key]['final_err'].append(ed_final[key][1])

            # 对各个能段进行遍历计算
            for f_idx, (low, high) in enumerate(self.intervals):
                m_init = self._get_metrics_with_error(run, is_final=False, low=low, high=high)
                m_final = self._get_metrics_with_error(run, is_final=True, low=low, high=high)

                # 只在第一个能段时记录温度
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

        # =========================================================================
        # 绘图逻辑：N个能段的 Excess 占比图 + 1个温度图
        # =========================================================================

        with ComparisonLayout(ctx, plot_ratio=(10, 3.5)) as layout:

            # --- 图 1 到 N: 各能段 Excess Ratio ---
            for i, (low, high) in enumerate(self.intervals):
                ax = layout.request_axes()
                d = results[i]

                # 将区间标注移入图内，作为图的说明
                label_str = f"${low:.2f}T < E < {high:.2f}T$" if high else f"$E > {low:.2f}T$"
                ax.text(0.98, 0.95, label_str, transform=ax.transAxes,
                        ha='right', va='top', fontsize='medium', fontweight='bold', bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))

                # 绘制理论误差带 (阴影)
                # ax.fill_between(x_scaled,
                #                 (np.array(d['init']) - np.array(d['init_th_err'])) * 100,
                #                 (np.array(d['init']) + np.array(d['init_th_err'])) * 100,
                #                 color=style.color_baseline_secondary, alpha=0.15,
                #                 label='初始理论散粒误差 (Shot Noise)')

                ax.fill_between(x_scaled,
                                (np.array(d['final']) - np.array(d['final_th_err'])) * 100,
                                (np.array(d['final']) + np.array(d['final_th_err'])) * 100,
                                color=style.color_comparison_primary, alpha=0.1,
                                label='最终理论传递误差')

                # 绘制实测值误差棒
                ax.errorbar(x_scaled, np.array(d['final']) * 100, yerr=np.array(d['final_err']) * 100,
                            fmt='-o', capsize=4, elinewidth=1.5,
                            color=style.color_comparison_primary, label='最终时刻 $t_{end}$')

                # ax.errorbar(x_scaled, np.array(d['init']) * 100, yerr=np.array(d['init_err']) * 100,
                #             fmt='--x', capsize=4, alpha=0.7,
                #             color=style.color_baseline_secondary, label='初始时刻 $t=0$')

                ax.axhline(0, color='black', linestyle='-', alpha=0.3, lw=1)
                ax.set_ylabel(f"超额能量占比 (%)")

                # 图例留在左上角，与右侧的区间标注互不干扰
                ax.legend(fontsize='x-small', loc='upper left', ncol=2)
                ax.grid(True, linestyle=':', alpha=0.4)

            # --- 温度变化 ---
            ax_temp = layout.request_axes()
            ax_temp.errorbar(x_scaled, temps['final'], yerr=np.array(temps['final_err']),
                             fmt='-s', capsize=4, color=style.color_comparison_secondary, lw=style.lw_base, label='最终温度 $T$')
            ax_temp.set_ylabel("$T_{eff}$ (keV)")
            ax_temp.legend(fontsize='small', loc='best')
            ax_temp.grid(True, linestyle='--', alpha=0.5)

            # --- 磁场能量占比 ---
            ax_mag = layout.request_axes()
            ax_mag.errorbar(x_scaled, mag_ratios['final'], yerr=np.array(mag_ratios['final_err']),
                            fmt='-d', capsize=4, color='darkorange', label='磁能占比 $E_B$')
            # ax_mag.errorbar(x_scaled, energy_ratios['init'], yerr=np.array(energy_ratios['init_err']),
            #                 fmt='--d', capsize=4, color='gray', lw=style.lw_base, label='初始时刻 $t=0$')

            ax_mag.set_ylabel(r"$E_B / E_{total}$")
            ax_mag.legend(fontsize='small', loc='best')

            # --- 电场比例图 ---
            ax_elec = layout.request_axes()
            ax_elec.errorbar(x_scaled, elec_ratios['final'], yerr=np.array(elec_ratios['final_err']), fmt='-^', capsize=4, color='crimson',
                             label='电能占比 $E_E$')
            ax_elec.set_ylabel(r"$E_E / E_{total}$")
            ax_elec.legend(fontsize='small')
            ax_elec.grid(True, alpha=0.3)

            # 总场能图 (Mag + Elec)
            ax_s = layout.request_axes()
            ax_s.errorbar(x_scaled, sum_field_ratios['final'], yerr=sum_field_ratios['final_err'], fmt='-P', color='indigo', label='总场能 (B+E)')
            ax_s.set_ylabel(r"$(E_B + E_E) / E_{total}$")

            # --- 能量密度图 (eV) ---
            color_kin = 'steelblue'
            color_mag = 'darkorange'
            color_elec = 'crimson'
            color_tot = 'darkgreen'

            # 动能密度
            ax_e_kin = layout.request_axes()
            ax_e_kin.errorbar(x_scaled, energy_densities['kin']['final'],
                              yerr=np.array(energy_densities['kin']['final_err']),
                              fmt='-o', capsize=4, color=color_kin, label='动能密度')
            ax_e_kin.set_ylabel(r"$U_{kin} / (n_0 m_e c^2)$")
            ax_e_kin.legend(fontsize='small', loc='best')
            ax_e_kin.grid(True, linestyle=':', alpha=0.4)
            ax_e_kin.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))

            # 磁能密度
            ax_e_mag = layout.request_axes()
            ax_e_mag.errorbar(x_scaled, energy_densities['mag']['final'],
                              yerr=np.array(energy_densities['mag']['final_err']),
                              fmt='-d', capsize=4, color=color_mag, label='磁能密度')
            ax_e_mag.set_ylabel(r"$U_B / (n_0 m_e c^2)$")
            ax_e_mag.legend(fontsize='small', loc='best')
            ax_e_mag.grid(True, linestyle=':', alpha=0.4)
            ax_e_mag.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))

            # 电能密度
            ax_e_elec = layout.request_axes()
            ax_e_elec.errorbar(x_scaled, energy_densities['elec']['final'],
                               yerr=np.array(energy_densities['elec']['final_err']),
                               fmt='-^', capsize=4, color=color_elec, label='电能密度')
            ax_e_elec.set_ylabel(r"$U_E / (n_0 m_e c^2)$")
            ax_e_elec.legend(fontsize='small', loc='best')
            ax_e_elec.grid(True, linestyle=':', alpha=0.4)
            ax_e_elec.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))

            # 总能量密度
            ax_e_tot = layout.request_axes()
            ax_e_tot.errorbar(x_scaled, energy_densities['total']['final'],
                              yerr=np.array(energy_densities['total']['final_err']),
                              fmt='-s', capsize=4, color=color_tot, linewidth=2, label='总能量密度')
            ax_e_tot.set_ylabel(r"$U_{total} / (n_0 m_e c^2)$")
            ax_e_tot.legend(fontsize='small', loc='best')
            ax_e_tot.grid(True, linestyle=':', alpha=0.4)
            ax_e_tot.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))

            console.print("\n[bold green]分析完成。[/bold green]")
            console.print("提示：已生成覆盖多组分的高能尾部演化图谱。")
            console.print(
                "可以观察：在较高阈值（如 >5T, >8T）下，初始底噪占比更低但相对散粒误差会被放大，\n如果在该能段信号显著脱离了阴影带，则证明产生了确凿的高能尾部物理加速机制。")
            console.print("理论上，Initial (Noise Floor) 的点应该非常紧密地围绕在 [bold yellow]0% (y=0 虚线)[/bold yellow] 上下均匀波动。")
