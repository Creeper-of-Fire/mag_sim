# analysis/modules/tail_statistics.py

from typing import List, Set, Tuple

import numpy as np
from scipy.constants import k as kB, c, m_e, e
from scipy.optimize import root_scalar
from scipy.special import kn as bessel_k

from analysis.modules.abstract.base_module import BaseAnalysisModule
from analysis.core.simulation import SimulationRun
from analysis.core.utils import console
from analysis.plotting.layout import create_analysis_figure

# 常量定义
ME_C2_J = m_e * c ** 2
J_PER_MEV = e * 1e6
J_TO_KEV = 1.0 / (e * 1e3)
STAT_THRESHOLD = 10  # 定义统计可靠性的宏粒子数阈值


class TailStatisticsModule(BaseAnalysisModule):
    @property
    def name(self) -> str:
        return "高能尾巴统计显著性分析v3"

    @property
    def description(self) -> str:
        return "【v3】分析高能尾部的宏粒子数量统计，判断偏离是否为物理真实或统计噪声。"

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

    def _perform_adaptive_binning(self,
                                  energies: np.ndarray,
                                  weights: np.ndarray,
                                  fine_bins: np.ndarray,
                                  threshold: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        执行自适应分箱逻辑。

        Args:
            energies: 粒子能量数组
            weights: 粒子权重数组
            fine_bins: 初始细分箱边界
            threshold: 合并阈值（最小宏粒子数）

        Returns:
            Tuple: (bins_edges, bin_centers, bin_widths, counts_weighted, counts_raw)
        """
        # 1. 先进行极其精细的直方图统计
        counts_w_fine, _ = np.histogram(energies, bins=fine_bins, weights=weights)
        counts_raw_fine, _ = np.histogram(energies, bins=fine_bins)

        new_counts_w = []
        new_counts_raw = []
        new_edges = [fine_bins[0]]

        temp_w = 0.0
        temp_raw = 0

        # 2. 遍历合并
        for i in range(len(counts_raw_fine)):
            temp_w += counts_w_fine[i]
            temp_raw += counts_raw_fine[i]

            # 只有当累积数量超过阈值时，才“结单”生成一个新 Bin
            if temp_raw >= threshold:
                new_counts_w.append(temp_w)
                new_counts_raw.append(temp_raw)
                new_edges.append(fine_bins[i + 1])  # 当前细分箱的右边界作为新Bin的右边界

                # 重置计数器
                temp_w = 0.0
                temp_raw = 0

        # 3. 处理残余尾部 (Key Fix)
        # 循环结束后，temp_raw 里可能还剩下了不足 threshold 个粒子（通常是最高能的那几个）
        if temp_raw > 0:
            if len(new_counts_raw) > 0:
                # 策略 A: 回缩合并 (Merge Back) - 推荐
                # 将这最后的一点点数据合并到前一个已经合格的 Bin 中
                # 这样可以避免生成一个统计误差巨大的末端 Bin
                new_counts_w[-1] += temp_w
                new_counts_raw[-1] += temp_raw
                new_edges[-1] = fine_bins[-1]  # 将最后一个 Bin 的右边界延伸到整个数据的最右端
            else:
                # 策略 B: 数据极其稀疏，连一个 Bin 都没凑齐
                # 这种情况下只能勉强作为一个 Bin
                new_counts_w.append(temp_w)
                new_counts_raw.append(temp_raw)
                new_edges.append(fine_bins[-1])

        # 4. 转换为 Numpy 数组并计算几何参数
        bins = np.array(new_edges)
        centers = np.sqrt(bins[:-1] * bins[1:])  # 对数坐标下使用几何平均值更准
        widths = np.diff(bins)
        counts_weighted = np.array(new_counts_w)
        counts_raw = np.array(new_counts_raw)

        return bins, centers, widths, counts_weighted, counts_raw

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

        # 1. 自动计算温度
        avg_E = np.average(spec.energies_MeV, weights=spec.weights)
        T_keV = run.user_T_keV if run.user_T_keV else self._solve_temperature(avg_E)

        # 核心阈值
        threshold_MeV = 3.0 * T_keV / 1000.0

        console.print(f"\n分析模拟: [cyan]{run.name}[/cyan] (Fit T = {T_keV:.2f} keV)")

        # 2. 准备初始细分箱 (Log-spacing)
        pos_E = spec.energies_MeV[spec.energies_MeV > 0]
        # 覆盖范围稍微宽一点，确保包含所有粒子
        bins_fine = np.logspace(np.log10(max(pos_E.min(), 1e-4)), np.log10(pos_E.max() * 1.01), 300)

        # 3. 调用自适应分箱 (OO Refactoring)
        bins, centers, widths, counts_weighted, counts_raw = self._perform_adaptive_binning(
            spec.energies_MeV,
            spec.weights,
            bins_fine,
            threshold=STAT_THRESHOLD
        )

        # 4. 计算 dN/dE (基于合并后的数据)
        dN_dE_sim = counts_weighted / widths

        # 5. 计算误差 (基于合并后的 counts_raw)
        with np.errstate(divide='ignore', invalid='ignore'):
            # 相对误差 = 1 / sqrt(N)
            rel_err = 1.0 / np.sqrt(counts_raw)
            rel_err[counts_raw == 0] = 0

            # dN/dE 的绝对误差
            abs_err = dN_dE_sim * rel_err

        # 6. 计算理论值 (使用新的 centers)
        total_weight = np.sum(spec.weights)
        dN_dE_theory = self._calculate_mj_theory(centers, T_keV, total_weight)

        # 7. 计算偏差 (Sim - Theory) / Theory
        with np.errstate(divide='ignore', invalid='ignore'):
            deviation = (dN_dE_sim - dN_dE_theory) / dN_dE_theory
            # 偏差的绝对误差 = (dN/dE_sim_err) / dN_dE_theory
            # (近似认为理论值无误差，只传导模拟值的统计误差)
            deviation_err = abs_err / dN_dE_theory

            mask_valid = (counts_raw > 0) & (dN_dE_theory > 1e-30)

        # --- 绘图逻辑更新 ---
        filename_override = f"{run.name}_tail_zoom_report"

        # 设定 X 轴缩放视野
        x_min_zoom = (T_keV / 1000.0) * 0.5
        x_max_zoom = max(pos_E.max(), threshold_MeV * 5)

        with create_analysis_figure(run, "tail_statistics", num_plots=2,
                                    plot_ratios=[3, 1], figsize=(8, 10),
                                    override_filename=filename_override) as (fig, axes):
            (ax_main, ax_dev) = axes

            # === 主图：能谱 ===
            # 1. 理论线
            ax_main.plot(centers, dN_dE_theory, 'r--', lw=2, label=f'理论热谱 ($T_{{eff}}={T_keV:.1f}$ keV)', zorder=10)

            # 2. 模拟点 (使用 errorbar，且带横向宽度)
            m = counts_raw > 0
            ax_main.errorbar(centers[m], dN_dE_sim[m],
                             yerr=abs_err[m],
                             fmt='o', markersize=4, capsize=3,
                             color='black', ecolor='gray', alpha=0.9,
                             label='模拟数据 (Adaptive Binning)', zorder=11)

            ax_main.axvline(threshold_MeV, color='blue', linestyle='-.', lw=1.5, label='非热阈值 $3kT$')

            # 视野调整
            view_mask = (centers >= x_min_zoom) & (centers <= x_max_zoom)
            vals_sim = dN_dE_sim[view_mask & m]
            vals_th = dN_dE_theory[view_mask]
            if len(vals_sim) > 0 and len(vals_th) > 0:
                y_max_vis = max(vals_sim.max(), vals_th.max())
                y_min_vis = min(vals_sim.min(), vals_th.min())
                ax_main.set_ylim(y_min_vis * 0.5, y_max_vis * 3.0)

            ax_main.set_xlim(x_min_zoom, x_max_zoom)
            ax_main.set_yscale('log')
            ax_main.set_ylabel('dN/dE (MeV$^{-1}$)')
            ax_main.legend(fontsize=10)
            ax_main.grid(True, which='both', alpha=0.3)

            # --- 子图2：偏差图 ---
            # 0 线
            ax_dev.axhline(0, color='r', linestyle='--', lw=1.5)

            # 绘制带有误差棒的偏差数据
            # 如果是单纯的热分布，这些点应该在 0 线附近波动，且 0 线穿过误差棒
            ax_dev.errorbar(centers[mask_valid], deviation[mask_valid],
                            yerr=deviation_err[mask_valid],
                            fmt='o', markersize=4, capsize=0,
                            color='black', ecolor='gray', alpha=0.8)

            # 3 sigma 阴影 (现在基于合并后的 counts，是一条平滑的带子)
            # 理论上，如果数据符合热分布，99.7% 的点应该落在这个绿带子里
            sigma_rel_merged = 1.0 / np.sqrt(counts_raw[mask_valid])

            # 这里的 3 sigma 是相对误差的 3 倍
            # 为了美观，我们用 fill_between 填充 +/- 3*sigma 区域
            # 注意：这是理论上的统计允许波动范围
            ax_dev.fill_between(centers[mask_valid], 3 * sigma_rel_merged, -3 * sigma_rel_merged,
                                step='mid',  # 使用 step 使得带子边缘清晰
                                color='green', alpha=0.1, label='3$\sigma$ 统计允许误差')

            ax_dev.axvline(threshold_MeV, color='blue', linestyle='-.')

            ax_dev.set_xlim(x_min_zoom, x_max_zoom)

            # Y轴限制：现在由于最后一格被合并了，这里的噪点应该会少很多
            # 但为了保险，还是做一个软限制
            dev_view = deviation[mask_valid & view_mask]
            if len(dev_view) > 0:
                # 找到视野内偏差的 95 分位数，避免被个别离群点撑开
                max_dev = np.percentile(np.abs(dev_view), 98)
                limit = min(max(max_dev * 1.5, 0.5), 5.0) # 动态范围
                ax_dev.set_ylim(-limit, limit)
            else:
                ax_dev.set_ylim(-1, 1)

            ax_dev.set_xlabel('能量 (MeV)')
            ax_dev.set_ylabel('相对偏差\n(Sim-Theory)/Theory')
            ax_dev.grid(True, alpha=0.5)
            ax_dev.legend(fontsize=8, loc='upper right')

        console.print(f"  [green]✔ 图表已生成: {filename_override}[/green]")