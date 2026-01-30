# analysis/modules/parametric_gain.py

import hashlib
import json
from pathlib import Path
from typing import List, Set, Dict, Any, Tuple

import matplotlib.pyplot as plt
import numpy as np
from rich.prompt import Prompt
from rich.table import Table

from utils.project_config import FILENAME_HISTORY
from .base_module import BaseComparisonModule
from ..core.simulation import SimulationRun
from ..core.utils import console
from ..plotting.layout import create_analysis_figure

# 增加一个统计阈值，避免因为初始粒子数太少导致比率爆炸
MIN_INITIAL_COUNTS = 10


class ParametricGainModule(BaseComparisonModule):
    @property
    def name(self) -> str:
        return "参数扫描：能谱增益效率"

    @property
    def description(self) -> str:
        return "自动识别变化的输入参数(X轴)，绘制峰值增益比率(f_final/f_initial)和对应能量的变化趋势。"

    @property
    def required_data(self) -> Set[str]:
        return {'initial_spectrum', 'final_spectrum'}

    # =========================================================================
    # 1. 物理计算核心 (增益比率计算)
    # =========================================================================

    def _create_common_bins(self, runs: List[SimulationRun]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """为所有模拟创建统一的能量分箱，确保比率计算的基准一致。"""
        all_energies = []
        for run in runs:
            if run.initial_spectrum: all_energies.append(run.initial_spectrum.energies_MeV)
            if run.final_spectrum: all_energies.append(run.final_spectrum.energies_MeV)

        if not all_energies: raise ValueError("无有效能谱数据")

        combined = np.concatenate(all_energies)
        positive = combined[combined > 0]
        if positive.size < 2: raise ValueError("有效能谱数据不足")

        min_e = max(positive.min() * 0.9, 1e-4)
        max_e = positive.max() * 1.1

        bins = np.logspace(np.log10(min_e), np.log10(max_e), 200)
        centers = np.sqrt(bins[:-1] * bins[1:])
        widths = np.diff(bins)
        return bins, centers, widths

    def _calculate_gain_metrics(self, run: SimulationRun, bins: np.ndarray, centers: np.ndarray, widths: np.ndarray) -> Dict[str, float]:
        """
        核心算法：计算单个run的峰值增益和对应能量。
        """
        spec_i = run.initial_spectrum
        spec_f = run.final_spectrum

        if not all([spec_i, spec_f, spec_i.weights.size > 0, spec_f.weights.size > 0]):
            return {'peak_gain': 0.0, 'peak_gain_energy_mev': 0.0}

        # 1. 计算初始和最终的 dN/dE
        counts_i, _ = np.histogram(spec_i.energies_MeV, bins=bins, weights=spec_i.weights)
        counts_f, _ = np.histogram(spec_f.energies_MeV, bins=bins, weights=spec_f.weights)
        dNdE_i = counts_i / widths
        dNdE_f = counts_f / widths

        # 2. 计算比率
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = dNdE_f / dNdE_i

        # 3. 建立一个统计学上可靠的掩码
        #   - 初始粒子数必须大于阈值，防止 1/1 这种偶然情况
        #   - 初始 dN/dE 必须大于0
        #   - 比率必须是有效的数值
        mask = (counts_i >= MIN_INITIAL_COUNTS) & (dNdE_i > 0) & np.isfinite(ratio)

        if not np.any(mask):
            return {'peak_gain': 1.0, 'peak_gain_energy_mev': 0.0}

        # 4. 在可靠区域内寻找峰值
        valid_ratios = ratio[mask]
        valid_centers = centers[mask]

        max_idx = np.argmax(valid_ratios)
        peak_gain = valid_ratios[max_idx]
        peak_energy = valid_centers[max_idx]

        return {
            'peak_gain': peak_gain,
            'peak_gain_energy_mev': peak_energy
        }

    # =========================================================================
    # 2. 参数探测核心 (从 ParametricTailModule 复用)
    # =========================================================================

    def _get_input_params(self, run: SimulationRun) -> Dict[str, Any]:
        run_path = Path(run.path).resolve()
        history_path = run_path.parent.parent / FILENAME_HISTORY
        if history_path.exists():
            try:
                with open(history_path, 'r', encoding='utf-8') as f:
                    for line in reversed(f.readlines()):
                        try:
                            record = json.loads(line)
                            if Path(record.get('output_dir', '')).resolve().name == run_path.name:
                                return record.get('params', {})
                        except:
                            continue
            except:
                pass
        params = {}
        for k, v in vars(run.sim).items():
            if isinstance(v, (int, float, str, bool)) and not k.startswith('_'):
                params[k] = v
        return params

    def _detect_variable_parameter(self, runs: List[SimulationRun]) -> Tuple[str, List[Any], List[SimulationRun]]:
        current_data = [{'run': r, 'params': self._get_input_params(r)} for r in runs]
        while True:
            if not current_data: return "Unknown", [], []
            all_keys = set().union(*(d['params'].keys() for d in current_data))
            varying_keys = [k for k in all_keys if len(set(str(d['params'].get(k)) for d in current_data)) > 1]
            if len(varying_keys) <= 1:
                x_key = varying_keys[0] if varying_keys else "Run Name"
                break

            console.print(f"\n[bold cyan]检测到 {len(varying_keys)} 个变化的参数[/bold cyan]")
            table = Table(title="变化参数列表")
            table.add_column("ID", style="cyan")
            table.add_column("参数名", style="magenta")
            table.add_column("值 (示例)", style="green")
            for i, key in enumerate(varying_keys):
                vals = list(set(str(d['params'].get(key)) for d in current_data))
                table.add_row(str(i + 1), key, ", ".join(vals[:3]) + ("..." if len(vals) > 3 else ""))
            console.print(table)

            idx_str = Prompt.ask("[bold]请选择参数编号[/bold]", choices=[str(i + 1) for i in range(len(varying_keys))])
            selected_key = varying_keys[int(idx_str) - 1]

            action = Prompt.ask(f"操作 for [bold magenta]{selected_key}[/bold magenta]: ([bold green]x[/]) 设为X轴 / ([bold red]f[/]) 筛选", choices=["x", "f"],
                                default="x")
            if action == 'x':
                x_key = selected_key
                break
            else:
                unique_vals = sorted(list(set(str(item['params'].get(selected_key)) for item in current_data)))
                console.print(f"\n可用值：")
                for i, v in enumerate(unique_vals): console.print(f"  {i + 1}) {v}")
                keep_idx = Prompt.ask("[bold yellow]选择要【保留】的值的编号[/bold yellow]", choices=[str(i + 1) for i in range(len(unique_vals))])
                target_val_str = unique_vals[int(keep_idx) - 1]
                current_data = [item for item in current_data if str(item['params'].get(selected_key)) == target_val_str]
                console.print(f"[green]已筛选。剩余 {len(current_data)} 个模拟。[/green]\n")

        def sort_key(item):
            if x_key == "Run Name": return item['run'].name
            try:
                return float(item['params'].get(x_key, 0))
            except:
                return str(item['params'].get(x_key, ""))

        current_data.sort(key=sort_key)

        sorted_runs = [item['run'] for item in current_data]
        sorted_values = [item['params'].get(x_key, item['run'].name if x_key == "Run Name" else "N/A") for item in current_data]
        return x_key, sorted_values, sorted_runs

    # =========================================================================
    # 3. 运行与绘图
    # =========================================================================

    def run(self, loaded_runs: List[SimulationRun]):
        console.print("\n[bold magenta]执行: 参数扫描能谱增益分析...[/bold magenta]")

        valid_runs = [r for r in loaded_runs if r.initial_spectrum and r.final_spectrum]
        if len(valid_runs) < 2:
            console.print("[red]错误: 需要至少 2 个包含初始和最终能谱的模拟。[/red]")
            return

        # 1. 建立全局统一分箱
        try:
            bins, centers, widths = self._create_common_bins(valid_runs)
        except ValueError as e:
            console.print(f"[red]创建分箱失败: {e}[/red]")
            return

        # 2. 检测变量并排序 (完全复用)
        x_label, x_vals, sorted_runs = self._detect_variable_parameter(valid_runs)

        # 3. 文件名处理 (完全复用)
        run_names_concat = "".join(sorted([r.name for r in sorted_runs]))
        short_hash = hashlib.md5(run_names_concat.encode('utf-8')).hexdigest()[:6]
        user_suffix = Prompt.ask(f"请输入文件名后缀 (默认基于哈希)", default=short_hash)
        final_filename = f"scan_gain_{x_label}_{user_suffix}"

        # 4. 计算指标
        y_peak_gain = []
        y_peak_energy = []

        console.print(f"  正在计算每个run的峰值增益...")
        for run in sorted_runs:
            m = self._calculate_gain_metrics(run, bins, centers, widths)
            y_peak_gain.append(m['peak_gain'])
            y_peak_energy.append(m['peak_gain_energy_mev'])
            console.print(f"    - {run.name}: {x_label}={self._get_input_params(run).get(x_label)} -> "
                          f"Peak Gain={m['peak_gain']:.2f} @ {m['peak_gain_energy_mev']:.3f} MeV")

        try:
            x_num = [float(v) for v in x_vals]
            is_num = True
        except (ValueError, TypeError):
            x_num = range(len(x_vals))
            is_num = False

        with create_analysis_figure(sorted_runs, "scan_gain", num_plots=2, figsize=(9, 8), override_filename=final_filename) as (fig, (ax1, ax2)):

            # --- 图1: 峰值增益 ---
            ax1.plot(x_num, y_peak_gain, 'o-', color='crimson', lw=2, markersize=6)
            ax1.set_ylabel(r"峰值增益比率 $\max(f_{final}/f_{initial})$")
            ax1.set_title(f"峰值增益 vs {x_label}", fontsize=14)
            ax1.grid(True, linestyle='--', alpha=0.5)
            ax1.set_yscale('log')  # 增益通常是对数尺度更直观

            # --- 图2: 发生峰值增益的能量 ---
            ax2.plot(x_num, y_peak_energy, 's-', color='darkorange', lw=2, markersize=6)
            ax2.set_ylabel("峰值增益处能量 (MeV)")
            ax2.set_title(f"最有效加速能量点 vs {x_label}", fontsize=14)
            ax2.set_xlabel(x_label if is_num else "Simulation Case", fontsize=12)
            ax2.grid(True, linestyle='--', alpha=0.5)
            ax2.set_yscale('log')

            if not is_num:
                # 如果X轴是字符串，设置标签
                plt.setp(ax1.get_xticklabels(), visible=False)  # 隐藏上图的x轴标签
                ax2.set_xticks(x_num)
                ax2.set_xticklabels(x_vals, rotation=45, ha='right')

            fig.tight_layout(rect=[0, 0, 1, 0.96])  # 为fig.suptitle留出空间