# analysis/modules/comparison/gof_test.py
from typing import List

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.stats import kstwo

from analysis.core.cache import cached_op
from analysis.core.parameter_selector import ParameterSelector
from analysis.core.simulation import SimulationRun
from analysis.core.utils import console
from analysis.modules.abstract.base_module import BaseComparisonModule
from analysis.modules.utils import physics_mj
from analysis.modules.utils.spectrum_tools import filter_valid_runs
from analysis.plotting.layout import create_analysis_figure
from analysis.plotting.styles import get_style


class GoodnessOfFitModule(BaseComparisonModule):
    @property
    def name(self) -> str:
        return "统计学检验：Maxwell-Jüttner 拟合优度 (K-S & A-D)"

    @property
    def description(self) -> str:
        return "使用加权 K-S 检验与 A-D 统计量，严谨评估等离子体能谱偏离纯热力学分布的显著性。"

    def _downsample_for_stats(self, energies: np.ndarray, weights: np.ndarray,
                              target_ppc: int = 100, n_bins: int = 500):
        """
        [核心逻辑：参考 slimmer.py]
        通过统计压缩，减少送入 quad 积分的粒子总数，同时保持分布形态不变。
        """
        if energies.size <= target_ppc * 10:
            return energies, weights  # 粒子数本来就少，不折腾

        # 1. 创建对数分箱作为“决策网格”
        bins = np.logspace(np.log10(energies.min()), np.log10(energies.max()), n_bins + 1)
        bin_indices = np.digitize(energies, bins)

        keep_idx = []
        new_weights = weights.copy()

        # 2. 遍历网格
        unique_bins, counts = np.unique(bin_indices, return_counts=True)
        for b_idx, count in zip(unique_bins, counts):
            idx_in_bin = np.where(bin_indices == b_idx)[0]

            # 如果 bin 内粒子数少于阈值（如高能尾部），100% 保留
            if count <= target_ppc:
                keep_idx.extend(idx_in_bin)
            else:
                # 如果是冗余的热核区域，进行等概率抽样并补偿权重
                factor = count / target_ppc
                selected = np.random.choice(idx_in_bin, target_ppc, replace=False)
                new_weights[selected] *= factor
                keep_idx.extend(selected)

        idx_final = np.array(keep_idx)
        return energies[idx_final], new_weights[idx_final]

    @cached_op(file_dep="particle")
    def _compute_gof_metrics(self, run: 'SimulationRun') -> dict:
        """
        核心算法：计算单个 run 的加权 K-S 统计量和 A-D 尾部统计量。
        """
        # 取最终时刻能谱
        spec = run.get_spectrum(step_index=-1)
        if spec is None or spec.weights.size == 0:
            return {'D_ks': 0.0, 'p_value': 1.0, 'AD_stat': 0.0, 'N_eff': 1.0}

        # 1. 获取基础物理量
        E_raw = spec.energies_MeV
        W_raw = spec.weights

        # 排除零能量或负能量产生的数值干扰
        valid_mask = E_raw > 0
        E_raw = E_raw[valid_mask]
        W_raw = W_raw[valid_mask]

        # --- 重要性采样压缩 ---
        # 我们将 65 万粒子压缩到约 2-3 万个，这不会损失任何统计显著性，但会让积分飞快
        console.print(f"      [统计预处理] 原始粒子数: {E_raw.size} ...")
        E_reduced, W_reduced = self._downsample_for_stats(E_raw, W_raw, target_ppc=80, n_bins=400)
        console.print(f"      [统计预处理] 压缩后参与计算粒子数: {E_reduced.size} (保留了所有稀疏高能粒子)")

        # 1. 基础物理量 (依然使用原始数据计算温度，保证拟合基准绝对正确)
        avg_E = np.sum(E_raw * W_raw) / np.sum(W_raw)
        T_keV = physics_mj.solve_mj_temperature_kev(avg_E)
        N_eff = (np.sum(W_raw) ** 2) / np.sum(W_raw ** 2)

        # 2. 计算有效粒子数 N_eff (必须基于原始数据，因为它反映了模拟的真实统计能力)
        N_eff = (np.sum(W_raw) ** 2) / np.sum(W_raw ** 2)

        # 3. 严格排序 (基于压缩后的数据)
        sort_idx = np.argsort(E_reduced)
        E_sorted = E_reduced[sort_idx]
        W_sorted = W_reduced[sort_idx]

        # 4. 经验累积分布 (eCDF)
        eCDF = np.cumsum(W_sorted) / np.sum(W_sorted)

        # 5. 理论累积分布 (Theoretical CDF)
        console.print(f"      [计算中] 正在对 {E_sorted.size} 个粒子进行精确 CDF 积分...")
        tCDF = physics_mj.calculate_mj_cdf(E_sorted, T_keV)

        # 6. 计算 Kolmogorov-Smirnov (K-S) 统计量
        # D_ks = max | eCDF - tCDF |
        D_ks = np.max(np.abs(eCDF - tCDF))

        # 使用 scipy.stats.kstwo.sf 计算大样本下的渐进 P 值 (Survival Function)
        p_value = kstwo.sf(D_ks, np.round(N_eff))

        # 7. 计算 Anderson-Darling (A-D) 加权统计量 (积分形式)
        # 公式: A^2 = N_eff * \int [ (eCDF - tCDF)^2 / (tCDF * (1 - tCDF)) ] d tCDF
        # 防止分母为 0，将理论 CDF 限制在 (1e-7, 1 - 1e-7)
        tCDF_safe = np.clip(tCDF, 1e-7, 1.0 - 1e-7)
        dF = np.diff(tCDF)  # 积分微元

        # 计算积分核 (去掉最后一个点以匹配 dF 的长度)
        integrand = ((eCDF[:-1] - tCDF[:-1]) ** 2) / (tCDF_safe[:-1] * (1.0 - tCDF_safe[:-1]))
        AD_stat = N_eff * np.sum(integrand * dF)

        return {
            'D_ks': D_ks,
            'p_value': p_value,
            'AD_stat': AD_stat,
            'N_eff': N_eff
        }

    def run(self, loaded_runs: List[SimulationRun]):
        style = get_style()
        console.print("\n[bold magenta]执行: 统计学假设检验 (K-S & A-D)...[/bold magenta]")

        valid_runs = filter_valid_runs(loaded_runs, require_particles=True)
        if len(valid_runs) < 1:
            console.print("[red]错误: 没有足够的数据进行分析。[/red]")
            return

        selector = ParameterSelector(valid_runs)
        x_label, x_vals, sorted_runs = selector.select()
        final_filename = selector.generate_filename(x_label, sorted_runs, prefix="gof_test")

        # 数据容器
        results_D_ks = []
        results_p_value = []
        results_AD_stat = []
        results_N_eff = []

        console.print("  正在计算每个参数点的累积分布拟合优度...")

        for i, run in enumerate(sorted_runs):
            from analysis.core.simulationGroup import SimulationRunGroup

            # 处理组 (Group) 或 单次 Run
            if isinstance(run, SimulationRunGroup):
                sub_d, sub_p, sub_ad, sub_neff = [], [], [], []
                for sub_run in run.runs:
                    m = self._compute_gof_metrics(sub_run)
                    sub_d.append(m['D_ks'])
                    sub_p.append(m['p_value'])
                    sub_ad.append(m['AD_stat'])
                    sub_neff.append(m['N_eff'])

                # 对组内多次模拟取均值
                avg_D = np.mean(sub_d)
                avg_P = np.mean(sub_p)
                avg_AD = np.mean(sub_ad)
                avg_Neff = np.mean(sub_neff)
            else:
                m = self._compute_gof_metrics(run)
                avg_D = m['D_ks']
                avg_P = m['p_value']
                avg_AD = m['AD_stat']
                avg_Neff = m['N_eff']

            results_D_ks.append(avg_D)
            results_p_value.append(avg_P)
            results_AD_stat.append(avg_AD)
            results_N_eff.append(avg_Neff)

            console.print(f"    [{run.name}] {x_label}={x_vals[i]} | N_eff={avg_Neff:.0f}")
            console.print(f"      -> K-S 差距 D_ks : {avg_D:.5f} (P-Value: {avg_P:.2e})")
            console.print(f"      -> A-D 统计量 A²  : {avg_AD:.2f}")

        # 将可能无法转为浮点数的 x 轴转换为序列
        try:
            x_num = [float(v) for v in x_vals]
            is_num = True
        except ValueError:
            x_num = range(len(x_vals))
            is_num = False
        x_arr = np.array(x_num)

        # ---------------- 绘图 ----------------
        with create_analysis_figure(sorted_runs, "gof_test", num_plots=2, override_filename=final_filename) as (fig, (ax1, ax2)):

            # 图1：K-S 统计量 与 拒绝域 (P-Value)
            # 为了可视化 P 值，我们将极小的 P 值钳制到一个下限，以防止在对数图上消失
            safe_p_values = np.clip(results_p_value, 1e-15, 1.0)

            # 使用 twinx 让一张图展现两个维度（左轴 D_ks, 右轴 P-value）
            ax1_twin = ax1.twinx()

            # 左轴：D_ks 统计量 (最大偏离度)
            line1 = ax1.plot(x_arr, results_D_ks, marker='o', color=style.color_comparison_primary, lw=2, label='K-S 统计量 $D_{ks}$ (左轴)')
            ax1.set_ylabel("K-S 最大分布偏离度 $D_{ks}$", color=style.color_comparison_primary)
            ax1.tick_params(axis='y', labelcolor=style.color_comparison_primary)

            # 右轴：P-Value
            line2 = ax1_twin.plot(x_arr, safe_p_values, marker='s', linestyle='--', color='gray', alpha=0.8, label='P 值 (右轴)')
            ax1_twin.set_ylabel("拒绝原假设的 $P$ 值", color='gray')
            ax1_twin.set_yscale('log')
            ax1_twin.tick_params(axis='y', labelcolor='gray')

            # 绘制显著性阈值线 (P = 0.05)
            ax1_twin.axhline(0.05, color='red', linestyle=':', lw=2, label=r'$\alpha = 0.05$ 显著性阈值')

            # 合并图例
            lines = line1 + line2 + [ax1_twin.lines[-1]]
            labels = [l.get_label() for l in lines]
            ax1.legend(lines, labels, loc='best')

            ax1.grid(True, linestyle='--', alpha=0.3)

            # 图2：A-D 尾部加权统计量
            ax2.plot(x_arr, results_AD_stat, marker='^', color=style.color_comparison_secondary, lw=2, label='Anderson-Darling 统计量 $A^2$')
            ax2.set_ylabel("A-D 统计量 $A^2$ (尾部偏离度)")
            x_label_name = "磁场扰动幅度 $\sigma$" if x_label == "target_sigma" else x_label
            ax2.set_xlabel(x_label_name if is_num else "模拟案例")

            # 添加文本标注解释
            ax2.text(0.02, 0.95, "提示: A² 越大代表尾部高能粒子偏离热谱越严重",
                     transform=ax2.transAxes, fontsize=10, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            ax2.legend()
            ax2.grid(True, linestyle='--', alpha=0.5)

            if not is_num:
                ax1.set_xticks(x_num)
                ax1.set_xticklabels(x_vals, rotation=45)
                ax2.set_xticks(x_num)
                ax2.set_xticklabels(x_vals, rotation=45)

        console.print("\n[bold green]拟合优度检验完成！[/bold green]")
        console.print("- [blue]P 值 < 0.05[/blue] 意味着我们在 95% 置信度下拒绝“这是一个纯热等离子体”。")
        console.print("- [blue]A-D 统计量骤增[/blue] 直接量化了高能尾部的不正常隆起。")