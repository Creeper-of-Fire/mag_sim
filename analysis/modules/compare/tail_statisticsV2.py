# analysis/modules/compare/tail_statisticsV2.py

import asyncio
import gc
from typing import List, Dict, Optional, Tuple

import numpy as np

from analysis.core.async_utils import asyncify
from analysis.core.simulation import SimulationRun
from analysis.core.simulationGroup import SimulationRunGroup
from analysis.core.utils import console
from analysis.modules.abstract.base_module import BaseComparisonModule, legacy
from analysis.physics.field import (
    compute_run_energy_partition, compute_run_energy_densities_normalized,
)
from analysis.physics.tail import (
    compute_run_tail_metrics, )
from analysis.physics.temperature import (
    compute_run_temperature_metrics, )
from analysis.plotting.comparison_layout import ComparisonContext, ComparisonLayout
from analysis.plotting.styles import get_style


@legacy(reason="已被 tail_statistics_timeseries 取代，后者支持全时序分析和多 run 分组统计")
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
        if isinstance(run_or_group, SimulationRunGroup):
            t_list, ratio_list, th_err_list = [], [], []
            for single_run in run_or_group.runs:
                step_idx = 0 if not is_final else -1
                fpath = single_run.get_particle_file(step_idx)
                temp_metrics = compute_run_temperature_metrics(single_run, fpath=fpath)
                metrics = compute_run_tail_metrics(
                    single_run,
                    temperature_metrics=temp_metrics,
                    f_low=low,
                    f_high=high,
                    fpath=fpath,
                )
                t_list.append(temp_metrics.T_keV)
                ratio_list.append(metrics.excess_ratio)
                th_err_list.append(metrics.propagated_uncertainty)
                gc.collect()

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
            fpath = run_or_group.get_particle_file(step_idx)
            temp_metrics = compute_run_temperature_metrics(run_or_group, fpath=fpath)
            metrics = compute_run_tail_metrics(run_or_group, temp_metrics, low, high, fpath=fpath)
            return {
                'T_keV': temp_metrics.T_keV,
                'T_keV_err': 0.0,
                'excess_ratio': metrics.excess_ratio,
                'excess_ratio_err': 0.0,
                'propagated_uncertainty': metrics.propagated_uncertainty
            }

    def _get_energy_densities(self, run_or_group: 'SimulationRun', is_final: bool) -> Dict[str, Tuple[float, float]]:
        """获取各能量密度, 返回均值和标准差"""
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

    async def run(self, loaded_runs: List[SimulationRun]):
        style = get_style()
        console.print(f"\n[bold magenta]执行: {self.name}...[/bold magenta]")

        ctx = ComparisonContext(loaded_runs, "multiband_tail")
        runs, x_scaled = ctx.unpack
        x_raw = ctx.x_raw
        x_label_key = ctx.x_label_key

        # 数据结构初始化
        results = {i: {'init': [], 'init_err': [], 'init_th_err': [],
                       'final': [], 'final_err': [], 'final_th_err': []}
                   for i in range(len(self.intervals))}

        temps = {'init': [], 'init_err': [], 'final': [], 'final_err': []}
        mag_ratios = {'init': [], 'init_err': [], 'final': [], 'final_err': []}
        elec_ratios = {'init': [], 'init_err': [], 'final': [], 'final_err': []}
        sum_field_ratios = {'init': [], 'init_err': [], 'final': [], 'final_err': []}
        energy_densities = {
            'kin': {'final': [], 'final_err': []},
            'mag': {'final': [], 'final_err': []},
            'elec': {'final': [], 'final_err': []},
            'total': {'final': [], 'final_err': []}
        }

        console.print("  正在并行扫描多能段 Initial 与 Final 数据 ...")

        async_field_metrics = asyncify(self._get_field_metrics)
        async_energy_densities = asyncify(self._get_energy_densities)
        async_metrics_with_error = asyncify(self._get_metrics_with_error)

        async def _process_run(run):
            fr_final, ed_final = await asyncio.gather(
                async_field_metrics(run, is_final=True),
                async_energy_densities(run, is_final=True),
            )

            interval_data = {}
            init_tasks = []
            final_tasks = []
            for f_idx, (low, high) in enumerate(self.intervals):
                init_tasks.append(asyncio.create_task(
                    async_metrics_with_error(run, is_final=False, low=low, high=high)))
                final_tasks.append(asyncio.create_task(
                    async_metrics_with_error(run, is_final=True, low=low, high=high)))

            for f_idx in range(len(self.intervals)):
                interval_data[f_idx] = (await init_tasks[f_idx], await final_tasks[f_idx])

            return fr_final, ed_final, interval_data

        all_results = await asyncio.gather(*[_process_run(run) for run in runs])

        for i, (fr_final, ed_final, interval_data) in enumerate(all_results):
            console.print(f"    [{runs[i].name}] {x_label_key}={x_raw[i]}")

            mag_ratios['final'].append(fr_final['mag'][0])
            mag_ratios['final_err'].append(fr_final['mag'][1])
            elec_ratios['final'].append(fr_final['elec'][0])
            elec_ratios['final_err'].append(fr_final['elec'][1])
            sum_field_ratios['final'].append(fr_final['sum'][0])
            sum_field_ratios['final_err'].append(fr_final['sum'][1])

            for key in ['kin', 'mag', 'elec', 'total']:
                energy_densities[key]['final'].append(ed_final[key][0])
                energy_densities[key]['final_err'].append(ed_final[key][1])

            for f_idx in range(len(self.intervals)):
                m_init, m_final = interval_data[f_idx]

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
