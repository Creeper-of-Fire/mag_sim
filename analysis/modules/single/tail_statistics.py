# analysis/modules/tail_statistics.py

from typing import List, Set, Tuple

import numpy as np

from analysis.core.simulation import SimulationRun
from analysis.core.utils import console
from analysis.modules.abstract.base_module import BaseAnalysisModule
from analysis.modules.utils import physics_mj
from analysis.plotting.layout import create_analysis_figure
from analysis.plotting.styles import get_style

# 常量定义
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

    def _perform_adaptive_binning(
            self,
            energies: np.ndarray,
            weights: np.ndarray,
            fine_bins: np.ndarray,
            threshold: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        执行自适应分箱逻辑，并计算加权统计量。

        Args:
            energies: 粒子能量数组
            weights: 粒子权重数组
            fine_bins: 初始细分箱边界
            threshold: 合并阈值（最小宏粒子数）

        Returns:
            Tuple: (
                bin_centers_weighted,  # 真实的能量重心
                counts_weighted,       # Sum(w)
                counts_w_squared,      # Sum(w^2) 用于误差计算
                bin_widths             # 宽度
            )
        """
        # 1. 预计算三种直方图
        # H_raw: 宏粒子数
        # H_w: 总权重 Sum(w)
        # H_w2: 权重平方和 Sum(w^2) -> 用于计算 Effective Sample Size
        # H_ew: 能量x权重 Sum(E*w) -> 用于计算 Bin 的能量重心

        counts_raw_fine, _ = np.histogram(energies, bins=fine_bins)
        counts_w_fine, _ = np.histogram(energies, bins=fine_bins, weights=weights)
        counts_w2_fine, _ = np.histogram(energies, bins=fine_bins, weights=weights ** 2)
        counts_ew_fine, _ = np.histogram(energies, bins=fine_bins, weights=energies * weights)

        new_counts_w = []
        new_counts_w2 = []
        new_counts_ew = []
        new_edges = [fine_bins[0]]

        # 临时累加器
        temp_raw = 0
        temp_w = 0.0
        temp_w2 = 0.0
        temp_ew = 0.0

        # 2. 遍历合并
        for i in range(len(counts_raw_fine)):
            temp_raw += counts_raw_fine[i]
            temp_w += counts_w_fine[i]
            temp_w2 += counts_w2_fine[i]
            temp_ew += counts_ew_fine[i]

            # 阈值判断：依然使用 raw counts 保证最基础的采样密度，
            # 但后续我们会用 w2 来计算误差
            if temp_raw >= threshold:
                new_counts_w.append(temp_w)
                new_counts_w2.append(temp_w2)
                new_counts_ew.append(temp_ew)
                new_edges.append(fine_bins[i + 1])

                # 重置
                temp_raw = 0
                temp_w = 0.0
                temp_w2 = 0.0
                temp_ew = 0.0

        # 3. 处理残余尾部
        if temp_raw > 0:
            if len(new_counts_w) > 0:
                # Merge Back
                new_counts_w[-1] += temp_w
                new_counts_w2[-1] += temp_w2
                new_counts_ew[-1] += temp_ew
                new_edges[-1] = fine_bins[-1]
            else:
                new_counts_w.append(temp_w)
                new_counts_w2.append(temp_w2)
                new_counts_ew.append(temp_ew)
                new_edges.append(fine_bins[-1])

        # 4. 转换为 Numpy 数组
        bins = np.array(new_edges)
        widths = np.diff(bins)
        counts_w = np.array(new_counts_w)
        counts_w2 = np.array(new_counts_w2)
        counts_ew = np.array(new_counts_ew)

        # 5. 计算加权能量重心 (Weighted Mean Energy)
        # 避免除以零（理论上 counts_w 在这里不会是0，因为 threshold logic）
        with np.errstate(divide='ignore', invalid='ignore'):
            centers_weighted = counts_ew / counts_w
            # 如果某个Bin没有粒子(极罕见)，回退到几何中心
            geometric_centers = np.sqrt(bins[:-1] * bins[1:])
            centers_weighted = np.where(counts_w > 0, centers_weighted, geometric_centers)

        return centers_weighted, counts_w, counts_w2, widths

    def run(self, loaded_runs: List[SimulationRun]):
        console.print("\n[bold magenta]执行: 高能尾巴统计显著性分析 (加权修正版)...[/bold magenta]")

        valid_runs = [r for r in loaded_runs if r.final_spectrum]
        if not valid_runs:
            console.print("[yellow]无有效能谱数据。[/yellow]")
            return

        for run in valid_runs:
            self._analyze_single_run(run)

    def _analyze_single_run(self, run: SimulationRun):
        spec = run.final_spectrum
        if spec.weights.size == 0: return

        # 获取当前绘图样式
        style = get_style()

        # 1. 自动计算温度
        avg_E = np.average(spec.energies_MeV, weights=spec.weights)
        T_keV = run.user_T_keV if run.user_T_keV else physics_mj.solve_mj_temperature_kev(avg_E)

        threshold_MeV = 3.0 * T_keV / 1000.0
        console.print(f"\n分析模拟: [cyan]{run.name}[/cyan] (Fit T = {T_keV:.2f} keV)")

        # 2. 准备初始细分箱
        pos_E = spec.energies_MeV[spec.energies_MeV > 0]
        bins_fine = np.logspace(np.log10(max(pos_E.min(), 1e-4)), np.log10(pos_E.max() * 1.01), 300)

        # 3. 调用自适应分箱 (获取加权统计量)
        centers_weighted, counts_w, counts_w2, widths = self._perform_adaptive_binning(
            spec.energies_MeV,
            spec.weights,
            bins_fine,
            threshold=STAT_THRESHOLD
        )

        # 4. 计算 dN/dE
        dN_dE_sim = counts_w / widths

        # 5. 计算误差 (关键修正)
        # 相对误差 Rel_Err = sqrt(Sum(w^2)) / Sum(w) = 1 / sqrt(N_eff)
        # 绝对误差 Abs_Err = dN/dE * Rel_Err
        with np.errstate(divide='ignore', invalid='ignore'):
            rel_err_1sigma = np.sqrt(counts_w2) / counts_w
            # 防止除以0
            rel_err_1sigma[counts_w == 0] = 0

            # 计算有效粒子数 N_eff，仅用于调试或展示
            n_eff = (counts_w ** 2) / counts_w2

            abs_err = dN_dE_sim * rel_err_1sigma

        # 6. 计算理论值 (使用加权能量重心，而不是几何中心，这样更准)
        total_weight = np.sum(spec.weights)
        dN_dE_theory = physics_mj.calculate_mj_pdf(centers_weighted, T_keV) * total_weight

        # 7. 计算偏差
        with np.errstate(divide='ignore', invalid='ignore'):
            deviation = (dN_dE_sim - dN_dE_theory) / dN_dE_theory
            deviation_err = abs_err / dN_dE_theory

            # 过滤无效点用于绘图
            mask_valid = (counts_w > 0) & (dN_dE_theory > 1e-30)

        # --- 绘图逻辑 ---
        filename_override = f"{run.name}_tail_zoom_report_v4"
        x_min_zoom = (T_keV / 1000.0) * 0.5
        x_max_zoom = max(pos_E.max(), threshold_MeV * 5)

        with create_analysis_figure(run, "tail_statistics", num_plots=2,
                                    plot_ratios=[3, 1], figsize=(8, 10),
                                    override_filename=filename_override) as (fig, axes):
            (ax_main, ax_dev) = axes

            # === 主图 ===
            # 理论线
            # 为了画出平滑的理论线，我们需要重新生成一组密集的 x 坐标，
            # 不能只用 sparse 的 centers_weighted
            x_smooth = np.logspace(np.log10(x_min_zoom), np.log10(x_max_zoom), 200)
            y_smooth = physics_mj.calculate_mj_pdf(x_smooth, T_keV) * total_weight
            ax_main.plot(x_smooth, y_smooth, 'r--', lw=2, label=f'理论热谱 ($T_{{eff}}={T_keV:.1f}$ keV)', zorder=10)

            # 模拟数据点 (x坐标现在是真实的能量重心)
            m = mask_valid
            ax_main.errorbar(centers_weighted[m], dN_dE_sim[m],
                             yerr=abs_err[m],
                             fmt='o', markersize=4, capsize=3,
                             color='black', ecolor='gray', alpha=0.9,
                             label='模拟数据 (N_eff Weighted)', zorder=11)

            ax_main.axvline(threshold_MeV, color='blue', linestyle='-.', lw=1.5, label='非热阈值 $3kT$')

            # 视野
            ax_main.set_xlim(x_min_zoom, x_max_zoom)

            # 自动调整 Y 轴 (基于视野内的数据)
            view_mask = (centers_weighted >= x_min_zoom) & (centers_weighted <= x_max_zoom)
            vals_in_view = dN_dE_sim[view_mask & m]
            if len(vals_in_view) > 0:
                ax_main.set_ylim(vals_in_view.min() * 0.5, vals_in_view.max() * 5.0)

            ax_main.set_yscale('log')
            ax_main.set_ylabel('dN/dE (MeV$^{-1}$)')
            ax_main.legend(fontsize=10)
            ax_main.grid(True, which='both', alpha=0.3)

            # === 子图：偏差 (Brazil Plot) ===

            # 1. 0 线 (理论完美符合)
            ax_dev.axhline(0, color='r', linestyle='--', lw=1.5)

            # 2. 绘制 "背景统计噪声带" (The Brazil Bands)
            # 使用 step='mid' 保持与分箱逻辑一致的方块感
            # 绿带: ± 1 sigma (68% CL)
            # 黄带: ± 3 sigma (99.7% CL) -> 原本是2sigma，这里改为3sigma更严格

            sigma_1 = rel_err_1sigma[m]
            sigma_3 = 3 * sigma_1

            # 这里的 x 轴用 centers_weighted 稍微有点断裂感，但更真实
            # 为了美观，fill_between 可以连接起来，或者用 step
            ax_dev.fill_between(centers_weighted[m], sigma_3, -sigma_3,
                                color='gold', alpha=0.3, step='mid',
                                label='$\pm 3\sigma$')

            ax_dev.fill_between(centers_weighted[m], sigma_1, -sigma_1,
                                color='limegreen', alpha=0.4, step='mid',
                                label='$\pm 1\sigma$')

            # 3. 数据点 (纯黑点，无误差棒)
            # 视觉逻辑：点落在绿带=符合；落在黄带=正常波动；落在外面=物理信号
            ax_dev.plot(centers_weighted[m], deviation[m], 'k.', markersize=3)

            ax_dev.axvline(threshold_MeV, color='blue', linestyle='-.')
            ax_dev.set_xlim(x_min_zoom, x_max_zoom)

            # 智能 Y 轴
            dev_view = deviation[m & view_mask]
            if len(dev_view) > 0:
                # 找到最大偏差，并确保视野能包住 3 sigma 带
                max_dev = np.max(np.abs(dev_view))
                max_sig = np.max(sigma_3[view_mask & m]) if np.any(view_mask & m) else 0
                limit = max(max_dev, max_sig) * 1.2
                limit = min(max(limit, 0.5), 10.0)  # 限制最大值防止崩坏
                ax_dev.set_ylim(-limit, limit)

            ax_dev.set_xlabel('能量 (MeV)')
            ax_dev.set_ylabel('相对偏差\n(Sim-Theory)/Theory')
            ax_dev.grid(True, alpha=0.5)
            ax_dev.legend(loc='upper right')

        console.print(f"  [green]✔ 图表已生成(修正版): {filename_override}[/green]")
