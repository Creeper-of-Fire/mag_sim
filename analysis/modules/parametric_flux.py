# analysis/modules/parametric_flux.py

import hashlib
import json
from pathlib import Path
from typing import List, Set, Dict, Any, Tuple

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from rich.prompt import Prompt
from rich.table import Table

from utils.project_config import FILENAME_HISTORY
from .base_module import BaseComparisonModule
from ..core.simulation import SimulationRun
from ..core.utils import console
from ..plotting.layout import create_analysis_figure

# 最小计数阈值，避免 1/1 或 0/0 产生噪音
MIN_COUNTS = 5


class ParametricFluxModule(BaseComparisonModule):
    @property
    def name(self) -> str:
        return "参数扫描：能谱通量热力图 (Inflow/Outflow)"

    @property
    def description(self) -> str:
        return "绘制 [参数-能量] 热力图。颜色表示粒子数的增益(红)或损耗(蓝)，直观展示加速区间随参数的移动。"

    @property
    def required_data(self) -> Set[str]:
        return {'initial_spectrum', 'final_spectrum'}

    # =========================================================================
    # 1. 数据处理核心
    # =========================================================================

    def _create_common_bins(self, runs: List[SimulationRun]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """创建全局统一的能量分箱 (Y轴)"""
        all_energies = []
        for run in runs:
            if run.initial_spectrum: all_energies.append(run.initial_spectrum.energies_MeV)
            if run.final_spectrum: all_energies.append(run.final_spectrum.energies_MeV)

        if not all_energies: raise ValueError("无有效能谱数据")
        combined = np.concatenate(all_energies)
        positive = combined[combined > 0]
        if positive.size < 2: raise ValueError("有效能谱数据不足")

        # 对数分箱
        min_e = max(positive.min() * 0.9, 1e-4)
        max_e = positive.max() * 1.1
        bins = np.logspace(np.log10(min_e), np.log10(max_e), 150)  # 150个能箱，保证解析度
        centers = np.sqrt(bins[:-1] * bins[1:])
        widths = np.diff(bins)
        return bins, centers, widths

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
    # 2. 参数探测逻辑 (复用)
    # =========================================================================

    def _get_input_params(self, run: SimulationRun) -> Dict[str, Any]:
        """从历史记录或模拟对象中提取输入参数"""
        run_path = Path(run.path).resolve()
        history_path = run_path.parent.parent / FILENAME_HISTORY

        if history_path.exists():
            try:
                with open(history_path, 'r', encoding='utf-8') as f:
                    # 倒序读取，匹配最新的记录
                    lines = f.readlines()
                    for line in reversed(lines):
                        try:
                            record = json.loads(line)
                            rec_path = Path(record.get('output_dir', '')).resolve()
                            if rec_path.name == run_path.name:
                                return record.get('params', {})
                        except:
                            continue
            except Exception as e:
                console.print(f"[yellow]读取 {FILENAME_HISTORY} 出错: {e}[/yellow]")

        # 回退策略
        params = {}
        for k, v in vars(run.sim).items():
            if isinstance(v, (int, float, str, bool)) and not k.startswith('_'):
                params[k] = v
        return params

    def _detect_variable_parameter(self, runs: List[SimulationRun]) -> Tuple[str, List[Any], List[SimulationRun]]:
        """
        交互式参数筛选与 X 轴选择 (完整逻辑)。
        """
        # 1. 初始收集所有参数
        current_data = []
        for run in runs:
            p = self._get_input_params(run)
            current_data.append({'run': run, 'params': p})

        while True:
            # --- A. 动态分析当前数据中的变化量 ---
            if not current_data:
                console.print("[red]错误：所有模拟数据都被过滤掉了！[/red]")
                return "Unknown", [], []

            # 提取所有键
            all_keys = set()
            for item in current_data:
                all_keys.update(item['params'].keys())

            # 找出当前真正变化的键
            varying_keys = []
            varying_details = {}

            for k in all_keys:
                values = set()
                for item in current_data:
                    val = item['params'].get(k, None)
                    values.add(str(val))

                if len(values) > 1:
                    varying_keys.append(k)
                    varying_details[k] = list(values)

            # --- B. 决策分支 ---

            # 情况 1: 没有变量了
            if len(varying_keys) == 0:
                console.print("[yellow]警告: 当前剩余的模拟参数完全一致。将使用 Run Name 作为 X 轴。[/yellow]")
                x_key = "Run Name"
                break

            # 情况 2: 只有一个变量 -> 自动选为 X 轴
            if len(varying_keys) == 1:
                x_key = varying_keys[0]
                console.print(f"[green]✔ 锁定单一扫描变量: [bold]{x_key}[/bold] (共 {len(current_data)} 个模拟)[/green]")
                break

            # 情况 3: 有多个变量 -> 进入交互菜单
            console.print(f"\n[bold cyan]检测到 {len(varying_keys)} 个变化的参数 (当前剩余 {len(current_data)} 个模拟)[/bold cyan]")

            table = Table(title="变化参数列表")
            table.add_column("ID", justify="right", style="cyan", no_wrap=True)
            table.add_column("参数名", style="magenta")
            table.add_column("当前包含的值 (示例)", style="green")

            for i, key in enumerate(varying_keys):
                vals = varying_details[key]
                val_str = ", ".join(vals[:3]) + ("..." if len(vals) > 3 else "")
                table.add_row(str(i + 1), key, val_str)

            console.print(table)
            console.print("[dim]提示: 选择参数后，你可以将其设定为 X 轴，或者根据其值过滤掉不需要的模拟。[/dim]")

            # 用户输入
            choices = [str(i + 1) for i in range(len(varying_keys))]
            idx_str = Prompt.ask(
                "[bold]请输入参数编号[/bold]",
                choices=choices
            )
            selected_key = varying_keys[int(idx_str) - 1]

            console.print(f"\n你选择了参数: [bold magenta]{selected_key}[/bold magenta]")
            action = Prompt.ask(
                "请选择操作 ([bold green]x[/]: 设为绘图 X 轴 / [bold red]f[/]: 筛选/剔除数据)",
                choices=["x", "f"],
                default="x",
                show_choices=False,
                case_sensitive=False
            )

            if action == "x":
                # 选定 X 轴，跳出循环
                x_key = selected_key
                break
            else:
                # --- C. 执行筛选逻辑 ---
                unique_vals = sorted(list(set([str(item['params'].get(selected_key)) for item in current_data])))

                console.print(f"\n在该参数下检测到以下值：")
                for i, v in enumerate(unique_vals):
                    console.print(f"  {i + 1}) {v}")

                keep_idx = Prompt.ask(
                    f"[bold yellow]请选择你要【保留】的一组数据的编号[/bold yellow] (其他将被剔除)",
                    choices=[str(i + 1) for i in range(len(unique_vals))]
                )
                target_val_str = unique_vals[int(keep_idx) - 1]

                before_count = len(current_data)
                current_data = [item for item in current_data if str(item['params'].get(selected_key)) == target_val_str]
                after_count = len(current_data)

                console.print(f"[green]已保留 {selected_key} = {target_val_str} 的数据。[/green]")
                console.print(f"[dim]模拟数量从 {before_count} 减少到 {after_count}。[/dim]\n")

        # 排序与输出
        def sort_key(item):
            if x_key == "Run Name":
                return item['run'].name
            val = item['params'].get(x_key, 0)
            try:
                return float(val)
            except:
                return str(val)

        current_data.sort(key=sort_key)

        sorted_runs = [item['run'] for item in current_data]
        sorted_values = [item['params'].get(x_key, item['run'].name if x_key == "Run Name" else "N/A") for item in current_data]

        return x_key, sorted_values, sorted_runs

    # =========================================================================
    # 3. 运行与绘图
    # =========================================================================

    def run(self, loaded_runs: List[SimulationRun]):
        console.print("\n[bold magenta]执行: 参数扫描能谱通量热力图...[/bold magenta]")

        valid_runs = [r for r in loaded_runs if r.initial_spectrum and r.final_spectrum]
        if len(valid_runs) < 2:
            console.print("[red]错误: 至少需要 2 个模拟。[/red]")
            return

        # 1. 统一分箱 (Y轴)
        try:
            bins, centers, widths = self._create_common_bins(valid_runs)
        except ValueError as e:
            console.print(f"[red]{e}[/red]")
            return

        # 2. 确定 X 轴参数
        x_label, x_vals, sorted_runs = self._detect_variable_parameter(valid_runs)

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
        run_names_concat = "".join(sorted([r.name for r in sorted_runs]))
        short_hash = hashlib.md5(run_names_concat.encode('utf-8')).hexdigest()[:6]
        user_suffix = Prompt.ask(f"文件名后缀", default=short_hash)
        final_filename = f"scan_flux_{x_label}_{user_suffix}"

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