# core/utils.py

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# --- 通用工具模块 ---
#
# 包含共享的、与具体物理计算无关的辅助函数。
#
import os
from collections import Counter
from typing import List, Tuple, Dict, Any, Optional

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.table import Table as mpl_Table
import numpy as np

# --- Rich 库用于漂亮的命令行交互 ---
from rich.console import Console
from rich.table import Table
from rich.prompt import Prompt

from .config import config
# --- 导入核心数据结构 ---
from .simulation import SimulationRun

# --- 全局常量和控制台 ---
console = Console()
from scipy import constants
C = constants.c
M_E = constants.m_e
E = constants.e
J_PER_MEV = E * 1e6

# =============================================================================
# 1. Matplotlib & 字体
# =============================================================================

def setup_chinese_font():
    """自动查找并设置支持中文的字体。"""
    from matplotlib import font_manager as fm
    chinese_fonts_priority = ['WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'Source Han Sans SC', 'SimHei', 'Microsoft YaHei']
    found_font = next((font for font in chinese_fonts_priority if fm.findfont(font, fontext='ttf')), None)
    if found_font:
        plt.rcParams['font.sans-serif'] = [found_font]
        console.print(f"[green]✔ Matplotlib 字体已设置为：{found_font}[/green]")
    else:
        console.print("[yellow]⚠ 警告：未能找到支持中文的字体。图表中的中文可能无法正常显示。[/yellow]")
    plt.rcParams['axes.unicode_minus'] = False

# =============================================================================
# 2. 交互式目录选择
# =============================================================================

def select_directories() -> List[str]:
    """扫描并让用户选择要分析的目录。"""
    console.print("\n[bold]扫描当前目录下的有效模拟文件夹...[/bold]")
    valid_dirs = [d.path for d in os.scandir('./sim_result') if
                  d.is_dir() and os.path.exists(os.path.join(d.path, 'sim_parameters.dpkl'))]

    if not valid_dirs:
        console.print("[red]错误: 未找到任何包含 'sim_parameters.dpkl' 的子目录。[/red]")
        return []

    table = Table(title="可用的模拟运行")
    table.add_column("索引", justify="right", style="cyan")
    table.add_column("文件夹名称", style="magenta")
    for i, dir_name in enumerate(valid_dirs):
        table.add_row(str(i), os.path.basename(dir_name))
    console.print(table)

    while True:
        try:
            prompt_text = "[bold]请输入要分析的模拟索引 (用逗号/空格分隔, [cyan]直接回车则全选[/cyan])[/bold]"
            choice_str = Prompt.ask(prompt_text, default="all")

            if choice_str.strip().lower() == "all":
                console.print(f"[green]已选择全部 {len(valid_dirs)} 个模拟。[/green]")
                return valid_dirs

            indices_str = choice_str.replace(',', ' ').split()
            if not indices_str:
                continue

            choices = [int(i) for i in indices_str]

            if all(0 <= c < len(valid_dirs) for c in choices):
                return [valid_dirs[c] for c in choices]
            else:
                console.print("[yellow]警告: 输入的索引超出范围，请重试。[/yellow]")
        except ValueError:
            console.print("[red]错误: 无效输入，请输入数字索引。[/red]")

# =============================================================================
# 3. 参数表绘制
# =============================================================================

def _get_param_map(run: SimulationRun) -> Dict[str, Any]:
    """
    为单个模拟生成参数名称到格式化函数的映射。
    将此逻辑提取出来以便重用。
    """
    m_e_c2_MeV = (M_E * C ** 2) / J_PER_MEV
    is_3d = hasattr(run.sim, 'NY') and hasattr(run.sim, 'Ly') and run.sim.NY > 1

    return {
        "--- 归一化 ---": None,
        "B_norm (β ≈ 1, T)": (lambda s: f"{s.B_norm:.2e}" if hasattr(s, 'B_norm') else "未定义"),
        "J_norm (极限电流密度, A/m²)": (lambda s: f"{s.J_norm:.2e}" if hasattr(s, 'J_norm') else "未定义"),
        "--- 物理参数 ---": None,
        "初始温度 T (keV)": (lambda s: f"{s.T_plasma / 1e3:.1f}"),
        "总数密度 n (/m³)": (lambda s: f"{s.n_plasma:.2e}"),
        "磁化强度 σ": (lambda s: f"{s.sigma:.3f}" if hasattr(s, 'sigma') and s.sigma > 0 else "N/A"),
        "--- 场与扰动 ---": None,
        "初始重联场 B0 (T)": (lambda s: f"{s.B0:.2f}" if hasattr(s, 'B0') and s.B0 > 0 else "0.0 (无)"),
        "磁场类型": (lambda s: s.B_field_type if hasattr(s, 'B_field_type') else "uniform (默认)"),
        "高斯场数量": (lambda s: f"{s.num_gaussians}" if hasattr(s, 'B_field_type') and s.B_field_type == 'multi_gaussian' else "N/A"),
        "高斯包宽度 (L_ratio)": (lambda s: f"{s.gaussian_width_L_ratio:.2f}" if hasattr(s, 'B_field_type') and 'gaussian' in s.B_field_type else "N/A"),
        "--- 束流参数 ---": None,
        "束流占比": (lambda s: f"{s.beam_fraction * 100:.0f} %" if hasattr(s, 'beam_fraction') and s.beam_fraction > 0 else "N/A"),
        "束流 p*c (MeV/c)": (lambda s: f"{(s.beam_u_drift * m_e_c2_MeV):.3f}" if hasattr(s, 'beam_u_drift') and s.beam_fraction > 0 else "N/A"),
        "束流能量 E_k (MeV)": (lambda s: f"{(s.beam_energy_eV / 1e6):.3f}" if hasattr(s, 'beam_energy_eV') and s.beam_fraction > 0 else "N/A"),
        "--- 真实尺寸 ---": None,
        "空间尺度 (m)": (lambda s: f"{s.Lx:.2e} x {s.Ly:.2e} x {s.Lz:.2e}" if is_3d else f"{s.Lx:.2e} x {s.Lz:.2e}"),
        "时间跨度 (s)": (lambda s: f"{s.total_steps * s.dt:.2e}"),
        "总粒子数 (加权)": "dynamic",
        "--- 数值参数 ---": None,
        "网格": (lambda s: f"{s.NX} x {s.NY} x {s.NZ}" if is_3d else f"{s.NX} x {s.NZ}"),
        "每单元模拟粒子数 (NPPC)": (lambda s: f"{s.NPPC}"),
    }

def _create_parameter_table_data(run: SimulationRun) -> List[List[str]]:
    """为单个模拟准备 Matplotlib 表格所需的数据。"""
    param_map = _get_param_map(run)
    table_data = []
    for param_name, formatter in param_map.items():
        if formatter is None:
            table_data.append([param_name, ''])
            continue

        value_str = "N/A"
        if formatter == "dynamic":
            if run.initial_spectrum and run.initial_spectrum.weights.size > 0:
                total_particles = np.sum(run.initial_spectrum.weights)
                value_str = f"{total_particles:.2e}"
        else:
            try:
                value_str = formatter(run.sim)
            except (AttributeError, TypeError):
                pass

        table_data.append([f"  {param_name}", value_str])
    return table_data

def plot_parameter_table(ax: Axes, run: SimulationRun) -> mpl_Table:
    """在给定的 Matplotlib Axes 对象上绘制一个模拟参数表。"""
    ax.axis('off')
    # ax.set_title('模拟参数详情', fontsize=16, y=1.0, pad=20)
    table_data = _create_parameter_table_data(run)
    table = ax.table(
        cellText=table_data,
        colLabels=['参数', '值'],
        loc='center',
        cellLoc='left',
        colWidths=[0.4, 0.4]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.0)
    for key, cell in table.get_celld().items():
        row, col = key
        cell.set_edgecolor('lightgray')
        if row == 0:
            cell.set_text_props(weight='bold', ha='center')
            cell.set_facecolor('#B0C4DE')
        else:
            if "---" in table_data[row - 1][0]:
                cell.set_text_props(weight='bold', ha='center')
                cell.set_facecolor('#E0E0E0')
            if col == 0:
                cell.set_text_props(ha='left')
            if row % 2 == 0:
                cell.set_facecolor('#F5F5F5')
    return table


# =============================================================================
# 4. 参数对比表
# =============================================================================
def _create_comparison_table_data(runs: List[SimulationRun]) -> Tuple[List[str], List[List[str]], List[List[bool]]]:
    """
    为多个模拟准备对比表数据，并生成一个用于高亮差异的掩码。
    """
    if not runs:
        return [], [], []

    # 使用第一个 run 来确定参数列表和顺序
    param_map = _get_param_map(runs[0])

    header = ["参数"] + [run.name for run in runs]
    table_data = []
    highlight_mask = []

    for param_name, formatter in param_map.items():
        # 分隔符行
        if formatter is None:
            table_data.append([param_name] + [''] * len(runs))
            highlight_mask.append([False] * (len(runs) + 1))
            continue

        # 数据行
        row_values = []
        for run in runs:
            value_str = "N/A"
            if formatter == "dynamic":
                if run.initial_spectrum and run.initial_spectrum.weights.size > 0:
                    total_particles = np.sum(run.initial_spectrum.weights)
                    value_str = f"{total_particles:.2e}"
            else:
                try:
                    value_str = formatter(run.sim)
                except (AttributeError, TypeError):
                    pass
            row_values.append(value_str)

        # "少数服从多数"高亮逻辑
        if len(row_values) > 1:
            # Counter 会统计每个值出现的次数
            counts = Counter(row_values)
            # 找到出现次数最多的值 (多数派)
            majority_value, _ = counts.most_common(1)[0]
            # 创建一个布尔掩码，如果值不等于多数派的值，则为 True
            row_mask = [val != majority_value for val in row_values]
        else:
            row_mask = [False] * len(row_values)

        # 完整的行数据和高亮掩码 (第一列参数名不高亮)
        table_data.append([f"  {param_name}"] + row_values)
        highlight_mask.append([False] + row_mask)

    return header, table_data, highlight_mask


def plot_comparison_parameter_table(ax: Axes, runs: List[SimulationRun]) -> mpl_Table:
    """在给定的 Axes 上绘制一个包含多个模拟参数的对比表，并高亮差异项。"""
    ax.axis('off')
    # ax.set_title('模拟参数对比', fontsize=16, y=1.0, pad=20)

    header, table_data, highlight_mask = _create_comparison_table_data(runs)
    if not table_data:
        return None

    # 动态调整列宽
    num_runs = len(runs)
    col_widths = [0.3] + [0.7 / num_runs] * num_runs

    table = ax.table(
        cellText=table_data,
        colLabels=header,
        loc='center',
        cellLoc='left',
        colWidths=col_widths
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.0)

    # 样式和高亮
    cells = table.get_celld()
    for (row, col), cell in cells.items():
        cell.set_edgecolor('lightgray')
        # 表头样式
        if row == 0:
            cell.set_text_props(weight='bold', ha='center')
            cell.set_facecolor('#B0C4DE')
            continue

        # 数据行样式
        # 分隔符
        if "---" in table_data[row - 1][0]:
            cell.set_text_props(weight='bold', ha='center')
            cell.set_facecolor('#E0E0E0')
        # 参数名列
        if col == 0:
            cell.set_text_props(ha='left')

        # 交替行颜色
        if row % 2 == 0:
            cell.set_facecolor('#F5F5F5')

        # 应用高亮
        if highlight_mask[row - 1][col]:
            cell.set_facecolor('#FFDDC1')  # 醒目的淡橙色
            cell.set_text_props(weight='bold')

    return table

# =============================================================================
# 绘图辅助函数
# =============================================================================
def save_figure(fig: Figure, filename: str, subfolder: Optional[str] = None):
    """
    将 Matplotlib Figure 保存到配置的输出目录中，可选择指定子文件夹。

    此函数封装了以下操作：
    1. 从全局配置 `config.output_dir` 构建基础输出路径。
    2. 如果提供了 `subfolder`，则将其拼接到路径中。
    3. 确保目标目录存在。
    4. 以标准参数 (dpi=200, bbox_inches='tight') 保存图像。
    Args:
        fig (Figure): 要保存的 Matplotlib Figure 对象。
        filename (str): 输出文件的基本名称 (例如 "spectrum_analysis.png")。
        subfolder (str, optional): 要在输出目录中创建/使用的子文件夹名称。
    """
    # 构造路径
    base_dir = config.output_dir
    if subfolder:
        output_dir = os.path.join(base_dir, subfolder)
    else:
        output_dir = base_dir

    # 确保目录存在
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, filename)

    # 保存图像
    fig.savefig(output_path, dpi=200, bbox_inches='tight')

    # 打印确认信息
    console.print(f"  [green]✔ 图已保存: {output_path}[/green]")