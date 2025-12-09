# analysis/modules/tail_statistics.py

from typing import List, Set, Tuple

import numpy as np
from scipy.constants import k as kB, c, m_e, e
from scipy.optimize import root_scalar
from scipy.special import kn as bessel_k

from .base_module import BaseAnalysisModule
from ..core.simulation import SimulationRun
from ..core.utils import console
from ..plotting.layout import create_analysis_figure

# 常量定义
ME_C2_J = m_e * c ** 2
J_PER_MEV = e * 1e6
J_TO_KEV = 1.0 / (e * 1e3)
STAT_THRESHOLD = 10  # 定义统计可靠性的宏粒子数阈值


class TailStatisticsModule(BaseAnalysisModule):
    @property
    def name(self) -> str:
        return "高能尾巴统计显著性分析"

    @property
    def description(self) -> str:
        return "分析高能尾部的宏粒子数量统计，判断偏离是否为物理真实或统计噪声。"

    @property
    def required_data(self) -> Set[str]:
        return {'final_spectrum'}

    def _calculate_mj_theory(self, energies_MeV: np.ndarray, T_keV: float, total_weight: float) -> np.ndarray:
        """计算给定能量点下的 Maxwell-Juttner 理论分布值 (dN/dE)"""
        if T_keV <= 0:
            return np.zeros_like(energies_MeV)

        T_J = T_keV * 1e3 * e
        theta = T_J / ME_C2_J

        # 转换能量单位到 J
        E_J = energies_MeV * J_PER_MEV

        # 相对论因子 gamma = 1 + Ek / mc^2
        gamma = 1.0 + E_J / ME_C2_J

        # 动量 p = sqrt(gamma^2 - 1) * mc
        # 或者 pc = sqrt(Ek^2 + 2*Ek*mc^2)
        pc_J = np.sqrt(E_J * (E_J + 2 * ME_C2_J))

        # 归一化系数 A = 1 / (mc^2 * theta * K2(1/theta))
        norm_factor = 1.0 / (ME_C2_J * theta * bessel_k(2, 1.0 / theta))

        # f(E) ~ p * gamma * exp(-gamma/theta)  (这是动能分布的核心项)
        # 注意：这里计算的是概率密度 PDF
        pdf = norm_factor * (pc_J / ME_C2_J) * gamma * np.exp(-gamma / theta)

        # 乘以总权重得到 dN/dE
        # 注意单位转换：pdf 是 per Joule，我们要 per MeV
        return total_weight * pdf * J_PER_MEV

    def _solve_temperature(self, avg_ek_mev: float) -> float:
        """(复用逻辑) 反推温度"""
        target_avg_ek_j = avg_ek_mev * J_PER_MEV

        def avg_kinetic_energy_MJ(T_K: float) -> float:
            if T_K <= 0: return -1e9
            theta = (kB * T_K) / ME_C2_J
            if theta < 1e-9: return 1.5 * kB * T_K
            return (3 * theta + bessel_k(1, 1.0 / theta) / bessel_k(2, 1.0 / theta) - 1.0) * ME_C2_J

        eq = lambda T_K: avg_kinetic_energy_MJ(T_K) - target_avg_ek_j
        T_guess = (2.0 / 3.0) * target_avg_ek_j / kB
        try:
            sol = root_scalar(eq, x0=T_guess, bracket=[T_guess * 0.1, T_guess * 10.0], method='brentq')
            return (sol.root * kB) * J_TO_KEV
        except:
            return 0.0

    def _find_unsafe_regions(self, counts_raw: np.ndarray, bins: np.ndarray) -> List[Tuple[float, float]]:
        """
        查找所有宏粒子数低于阈值的连续区间。
        返回一个包含 (start_energy, end_energy) 元组的列表。
        """
        is_unsafe = counts_raw < STAT_THRESHOLD
        if not np.any(is_unsafe):
            return []

        # 使用 padding 来优雅地处理数组开头和结尾就是不可靠区的情况
        padded_unsafe = np.concatenate(([False], is_unsafe, [False]))
        diffs = np.diff(padded_unsafe.astype(int))

        # 值为 1 的地方是区间的开始 (False -> True)
        start_indices = np.where(diffs == 1)[0]
        # 值为 -1 的地方是区间的结束 (True -> False)
        end_indices = np.where(diffs == -1)[0] - 1

        regions = []
        for start_idx, end_idx in zip(start_indices, end_indices):
            # 使用分箱的边界来定义色块的范围，更精确
            start_energy = bins[start_idx]
            end_energy = bins[end_idx + 1]
            regions.append((start_energy, end_energy))

        return regions

    def run(self, loaded_runs: List[SimulationRun]):
        console.print("\n[bold magenta]执行: 高能尾巴统计显著性分析...[/bold magenta]")

        valid_runs = [r for r in loaded_runs if r.final_spectrum]
        if not valid_runs:
            console.print("[yellow]无有效能谱数据。[/yellow]")
            return

        for run in valid_runs:
            self._analyze_single_run(run)

    def _analyze_single_run(self, run: SimulationRun):
        spec = run.final_spectrum
        if spec.weights.size == 0: return

        # 1. 自动计算温度 (如果尚未计算)
        avg_E = np.average(spec.energies_MeV, weights=spec.weights)
        T_keV = run.user_T_keV if run.user_T_keV else self._solve_temperature(avg_E)

        console.print(f"\n分析模拟: [cyan]{run.name}[/cyan] (Fit T = {T_keV:.2f} keV)")

        # 2. 建立分箱 (Log-spacing)
        # 只要正能量部分
        pos_E = spec.energies_MeV[spec.energies_MeV > 0]
        bins = np.logspace(np.log10(max(pos_E.min(), 1e-4)), np.log10(pos_E.max() * 1.1), 100)
        centers = np.sqrt(bins[:-1] * bins[1:])
        widths = np.diff(bins)

        # 3. 统计核心：获取加权计数 和 原始计数
        # counts_weighted: 物理上的粒子数 (dN * dE)
        counts_weighted, _ = np.histogram(spec.energies_MeV, bins=bins, weights=spec.weights)
        # counts_raw: 模拟中的宏粒子个数 (用于计算置信度)
        counts_raw, _ = np.histogram(spec.energies_MeV, bins=bins)  # 不加权重！

        # 4. 计算 dN/dE
        dN_dE_sim = counts_weighted / widths

        # 5. 计算统计误差 (基于泊松分布)
        # 相对误差 rel_err = 1 / sqrt(N_raw)
        # 绝对误差 abs_err = dN_dE * rel_err
        # 防止除以零
        with np.errstate(divide='ignore', invalid='ignore'):
            rel_err = 1.0 / np.sqrt(counts_raw)
            rel_err[counts_raw == 0] = 0
            abs_err = dN_dE_sim * rel_err

        # 6. 计算理论值
        total_weight = np.sum(spec.weights)
        dN_dE_theory = self._calculate_mj_theory(centers, T_keV, total_weight)

        # 7. 计算相对偏差 (Deviation)
        # Deviation = (Sim - Theory) / Theory
        with np.errstate(divide='ignore', invalid='ignore'):
            deviation = (dN_dE_sim - dN_dE_theory) / dN_dE_theory
            # 如果理论值极小，偏差会爆炸，设个上限或mask
            mask_valid = (counts_raw > 0) & (dN_dE_theory > 1e-20)

        # 找到所有不可靠区间
        unsafe_regions = self._find_unsafe_regions(counts_raw, bins)
        if unsafe_regions:
            console.print(f"  [yellow]发现 {len(unsafe_regions)} 个统计不可靠区域 (N_raw < {STAT_THRESHOLD})。[/yellow]")

        # --- 开始绘图 (3子图) ---
        # 覆盖文件名以避免哈希
        filename_override = f"analysis_tail_stats_{run.name}"

        is_reliable = counts_raw >= STAT_THRESHOLD
        is_significant_deviation = (deviation > 3 * rel_err)
        significant_mask = is_reliable & is_significant_deviation

        # 我们需要自定义比例：主图大一点，下面两个小一点
        with create_analysis_figure(run, "tail_stats", num_plots=3,
                                    plot_ratios=[3, 1, 1], figsize=(10, 12),
                                    override_filename=filename_override) as (fig, axes):
            (ax_main, ax_count, ax_dev) = axes

            # 先画一个假的色块，只为了在图例中显示
            if unsafe_regions:
                ax_main.fill_betweenx(ax_main.get_ylim(), 0, 0, color='orange', alpha=0.2,
                                      label=f'统计不可靠区 ($N_{{raw}} < {STAT_THRESHOLD}$)')

            # 循环给所有子图都画上色块背景
            for ax in [ax_main, ax_count, ax_dev]:
                for start_e, end_e in unsafe_regions:
                    ax.axvspan(start_e, end_e, color='orange', alpha=0.2, zorder=0)

            # --- 子图 1: 能谱 + 误差棒 ---
            ax_main.set_title(f"高能尾巴统计分析: {run.name} (T $\\approx$ {T_keV:.1f} keV)")
            ax_main.plot(centers, dN_dE_theory, 'r:', lw=2, label='理论 Maxwell-Juttner', zorder=10)
            m = counts_raw > 0
            # 1. 绘制所有普通点 (不显著的点)
            normal_mask = m & (~significant_mask)
            ax_main.errorbar(centers[normal_mask], dN_dE_sim[normal_mask], yerr=abs_err[normal_mask],
                             fmt='o', markersize=3, capsize=3, color='royalblue', alpha=0.7,
                             label='模拟数据 (Error = $1/\\sqrt{N_{macro}}$)')

            # 2. 如果存在显著点，用不同样式高亮绘制它们
            if np.any(significant_mask):
                ax_main.errorbar(centers[significant_mask], dN_dE_sim[significant_mask], yerr=abs_err[significant_mask],
                                 fmt='o',
                                 capsize=3, color='gold', ecolor='gold',
                                 markeredgecolor='black', markeredgewidth=0.5,
                                 label='显著非热点', zorder=20)  # zorder 让他们在最上层

            ax_main.set_xscale('log')
            ax_main.set_yscale('log')
            ax_main.set_ylabel('dN/dE [/MeV]')
            ax_main.legend()
            ax_main.grid(True, which='both', alpha=0.3)

            # --- 子图 2: 宏粒子原始计数 ---
            ax_count.bar(centers, counts_raw, width=widths, align='center', color='gray', alpha=0.6)
            ax_count.set_xscale('log')
            ax_count.set_yscale('log')
            ax_count.set_ylabel('宏粒子数量 $N_{raw}$')
            ax_count.set_title('每个能箱内的真实计算粒子数')
            ax_count.axhline(STAT_THRESHOLD, color='orange', linestyle='--', linewidth=1)
            ax_count.axhline(1, color='red', linestyle='-', linewidth=1)
            ax_count.text(centers[0], STAT_THRESHOLD * 1.1, f" 信度阈值 (N={STAT_THRESHOLD})", color='orange', fontsize=8, va='bottom')

            # --- 子图 3: 相对偏差 ---
            # 1. 绘制所有普通点
            normal_mask_dev = mask_valid & (~significant_mask)
            ax_dev.plot(centers[normal_mask_dev], deviation[normal_mask_dev], 'k.-', lw=1, label='_nolegend_')

            # 2. 如果存在显著点，用不同样式高亮绘制它们
            if np.any(significant_mask):
                significant_mask_dev = mask_valid & significant_mask
                ax_dev.plot(centers[significant_mask_dev], deviation[significant_mask_dev],
                            marker='o', color='gold',
                            markeredgecolor='black', markeredgewidth=0.5,
                            linestyle='None', label='显著非热点', zorder=20)

            ax_dev.axhline(0, color='r', linestyle=':', lw=1)
            sigma_rel = 1.0 / np.sqrt(counts_raw[mask_valid])
            ax_dev.fill_between(centers[mask_valid], 3 * sigma_rel, -3 * sigma_rel,
                                color='green', alpha=0.1, label='3$\sigma$ 泊松噪声范围')
            ax_dev.set_xscale('log')
            ax_dev.set_ylim(-2, 5)
            ax_dev.set_ylabel('相对偏差\n$(N_{sim}-N_{th})/N_{th}$')
            ax_dev.set_xlabel('动能 (MeV)')
            ax_dev.grid(True, alpha=0.3)
            ax_dev.legend(fontsize='small', loc='upper left')

        # --- 文本报告 ---
        # 找出 deviation 显著大于 3 sigma 且 N_raw > 10 的点
        console.print("[dim]  正在扫描非热特征...[/dim]")
        # 扫描时，要排除掉不可靠区
        significant_idx = np.where((deviation > 3 * rel_err) & is_reliable)[0]

        if len(significant_idx) > 0:
            console.print(f"  [red]⚠ 发现潜在的非热成分![/red]")
            console.print(f"    在能量区间 {centers[significant_idx[0]]:.2f} - {centers[significant_idx[-1]]:.2f} MeV")
            console.print(f"    观测值显著高于理论值 (超出 3倍泊松标准差)")
        else:
            console.print(f"  [green]✔ 未检测到显著的高能非热尾巴 (所有偏差均在统计噪声范围内)。[/green]")
