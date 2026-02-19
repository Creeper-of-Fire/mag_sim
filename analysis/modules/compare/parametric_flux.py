# analysis/modules/parametric_flux.py

from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from analysis.core.parameter_selector import ParameterSelector
from analysis.core.simulation import SimulationRun
from analysis.core.utils import console
from analysis.modules.abstract.base_module import BaseComparisonModule
from analysis.plotting.layout import create_analysis_figure

# 最小计数阈值，避免 1/1 或 0/0 产生噪音
MIN_COUNTS = 5


class ParametricFluxModule(BaseComparisonModule):
    @property
    def name(self) -> str:
        return "参数扫描：能谱通量热力图 (Inflow/Outflow)"

    @property
    def description(self) -> str:
        return "绘制 [参数-能量] 热力图。颜色表示粒子数的增益(红)或损耗(蓝)，直观展示加速区间随参数的移动。"

    # =========================================================================
    # 1. 数据处理核心
    # =========================================================================

    def _calculate_log_gain(self, run: SimulationRun, bins: np.ndarray, widths: np.ndarray) -> np.ndarray:
        """
        计算 Log10(Gain)。
        Gain = (dN/dE_final) / (dN/dE_initial)
        Result > 0: Inflow (Gain)
        Result < 0: Outflow (Loss)
        """
        spec_i = run.initial_spectrum
        spec_f = run.final_spectrum

        # 这里的 fill_value=0 意味着没有数据的地方设为0
        if not (spec_i and spec_f): return np.zeros(len(bins) - 1)

        counts_i, _ = np.histogram(spec_i.energies_MeV, bins=bins, weights=spec_i.weights)
        counts_f, _ = np.histogram(spec_f.energies_MeV, bins=bins, weights=spec_f.weights)

        dNdE_i = counts_i / widths
        dNdE_f = counts_f / widths

        # 计算比率
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = dNdE_f / dNdE_i

        # 处理数值问题
        # 1. 如果 Initial 很小，Ratio 会不稳定 -> Mask掉
        # 2. 如果 Initial=0 但 Final>0 (纯注入) -> Ratio=Inf -> 设为一个较大的数值
        # 3. 如果 Final=0 (纯耗散) -> Ratio=0 -> LogRatio=-Inf -> 设为一个较小的数值

        mask_stable = (counts_i >= MIN_COUNTS)

        # 初始化为 0 (对应 Ratio=1, 无变化，也就是白色背景)
        log_gain = np.zeros_like(ratio)

        # 只计算稳定区域
        valid_idx = mask_stable & (ratio > 0)
        log_gain[valid_idx] = np.log10(ratio[valid_idx])

        # 处理特殊边界 (可选，视需求而定，这里为了热图平滑，还是Mask掉极值比较好)
        # 将极端值截断，防止颜色条被撑爆
        log_gain = np.clip(log_gain, -2, 2)  # 限制在 0.01倍 到 100倍 之间

        # 将不稳定区域设为 NaN，这样绘图时会显示为背景色或特定颜色
        log_gain[~mask_stable] = np.nan

        return log_gain

    # =========================================================================
    # 2. 运行与绘图
    # =========================================================================

    def run(self, loaded_runs: List[SimulationRun]):
        console.print("\n[bold magenta]执行: 参数扫描能谱通量热力图...[/bold magenta]")

        valid_runs = [r for r in loaded_runs if r.initial_spectrum and r.final_spectrum]
        if len(valid_runs) < 2:
            console.print("[red]错误: 至少需要 2 个模拟。[/red]")
            return

        # 1. 统一分箱 (Y轴)
        try:
            bins, centers, widths = create_common_energy_bins(valid_runs)
        except ValueError as e:
            console.print(f"[red]{e}[/red]")
            return

        # 2. 使用 Selector 准备数据
        selector = ParameterSelector(valid_runs)
        x_label, x_vals, sorted_runs = selector.select()
        final_filename = selector.generate_filename(x_label, sorted_runs, prefix="scan_flux")

        # 3. 构建数据矩阵 Z
        # 矩阵形状: (Energy_Bins, Parameters)
        z_matrix_list = []

        console.print(f"  正在构建热力图矩阵...")
        for run in sorted_runs:
            log_gain = self._calculate_log_gain(run, bins, widths)
            z_matrix_list.append(log_gain)

        # 转置: 行是能量(Y)，列是参数(X)
        Z = np.array(z_matrix_list).T

        # 4. 处理 X 轴坐标
        try:
            # 尝试转为数值型坐标
            x_coords = np.array([float(v) for v in x_vals])
            is_numeric_x = True
        except (ValueError, TypeError):
            # 如果是字符串，使用索引 0, 1, 2...
            x_coords = np.arange(len(x_vals))
            is_numeric_x = False

        # 5. 绘图
        with create_analysis_figure(sorted_runs, "scan_flux", num_plots=1, figsize=(11, 7), override_filename=final_filename) as (fig, ax):

            # 设置颜色映射：红蓝发散色，中间(0)为白色
            cmap = plt.cm.RdBu_r
            # 设置不可靠区域(NaN)的颜色为灰色，以便区分
            cmap.set_bad(color='#e0e0e0')

            # 网格生成
            # 如果是数值型X，我们希望方块居中，所以需要计算边界
            if is_numeric_x and len(x_coords) > 1:
                # 简单的中间点插值法估算边界
                # 这对于非均匀网格也能工作
                x_mid = (x_coords[:-1] + x_coords[1:]) / 2
                x_edges = np.concatenate(([x_coords[0] - (x_mid[0] - x_coords[0])], x_mid, [x_coords[-1] + (x_coords[-1] - x_mid[-1])]))
            else:
                # 均匀或者单个点
                x_edges = np.arange(len(x_vals) + 1) - 0.5

            X_grid, Y_grid = np.meshgrid(x_edges, bins)  # Y_grid 用 bins 边界

            # 绘制 pcolormesh
            # Z 需要匹配 grid 形状 (Y-1, X-1)
            # 我们的 Z 是 (Energy, Param)，刚好匹配
            mesh = ax.pcolormesh(X_grid, Y_grid, Z, cmap=cmap, vmin=-2, vmax=2, shading='flat')

            # 添加等高线 (辅助看清 Ratio=1, Ratio=10 的边界)
            # 需要计算中心点网格用于 contour
            X_cntr, Y_cntr = np.meshgrid(x_coords, centers)

            # 使用 Masked Array 处理 NaN，否则 contour 会报错或画出奇怪的线
            Z_masked = np.ma.masked_invalid(Z)

            # 只有当数据点足够多时才画等高线
            if len(x_coords) > 3:
                # 绘制 Ratio = 1 (Log=0) 的线，表示 Gain/Loss 分界
                ax.contour(X_cntr, Y_cntr, Z_masked, levels=[0], colors='black', linewidths=1.5, linestyles='--')
                # 绘制 Ratio = 10 (Log=1) 的线，表示显著加速区
                ax.contour(X_cntr, Y_cntr, Z_masked, levels=[1], colors='black', linewidths=1.0, linestyles=':')

            # 颜色条
            cbar = fig.colorbar(mesh, ax=ax, pad=0.02)
            cbar.set_label(r"Log$_{10}$ (Gain Ratio) = $\log_{10}(f_{final}/f_{initial})$")
            # 在颜色条上标记物理意义
            cbar.ax.text(1.3, 1.5, "Inflow (Heating)", color='darkred', ha='left', va='center', rotation=90, fontsize=9)
            cbar.ax.text(1.3, -1.5, "Outflow (Cooling)", color='darkblue', ha='left', va='center', rotation=90, fontsize=9)
            cbar.ax.text(1.3, 0, "No Change", color='gray', ha='left', va='center', rotation=90, fontsize=9)

            # 坐标轴设置
            ax.set_yscale('log')
            ax.set_ylabel("动能 (MeV)")
            ax.set_title(f"能谱通量图: 加速区间 vs {x_label}", fontsize=14)

            if is_numeric_x:
                ax.set_xlabel(x_label)
                # 如果是 log 分布的参数，把 X 轴设为 log
                if x_coords.min() > 0 and (x_coords.max() / x_coords.min() > 10):
                    ax.set_xscale('log')
            else:
                ax.set_xticks(x_coords)
                ax.set_xticklabels(x_vals, rotation=45, ha='right')
                ax.set_xlabel("Simulation Case")

            # 在图上添加注释
            ax.text(0.02, 0.95, "Red = Net Particle Gain", color='darkred', transform=ax.transAxes, fontweight='bold')
            ax.text(0.02, 0.90, "Blue = Net Particle Loss", color='darkblue', transform=ax.transAxes, fontweight='bold')
