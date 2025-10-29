# core/utils.py

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# --- 通用工具模块 ---
#
# 包含共享的、与具体物理计算无关的辅助函数。
#
import os
from typing import List

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.table import Table as mpl_Table
import numpy as np

# --- Rich 库用于漂亮的命令行交互 ---
from rich.console import Console
from rich.table import Table
from rich.prompt import Prompt

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

def _create_parameter_table_data(run: SimulationRun) -> List[List[str]]:
    """为单个模拟准备 Matplotlib 表格所需的数据。"""
    m_e_c2_MeV = (M_E * C ** 2) / J_PER_MEV

    is_3d = hasattr(run.sim, 'NY') and hasattr(run.sim, 'Ly') and run.sim.NY > 1

    param_map = {
        "--- 归一化 ---": None,
        "B_norm (β ≈ 1, T)": (lambda s: f"{s.B_norm:.2e}" if hasattr(s, 'B_norm') else "未定义"),
        "J_norm (极限电流密度, A/m²)": (lambda s: f"{s.J_norm:.2e}" if hasattr(s, 'J_norm') else "未定义"),
        "--- 物理参数 ---": None,
        "初始温度 T (keV)": (lambda s: f"{s.T_plasma / 1e3:.1f}"),
        "总数密度 n (/m³)": (lambda s: f"{s.n_plasma:.2e}"),
        "初始重联场 B0 (T)": (lambda s: f"{s.B0:.2f}" if hasattr(s, 'B0') and s.B0 > 0 else "0.0 (无)"),
        "磁化强度 σ": (lambda s: f"{s.sigma:.3f}" if hasattr(s, 'sigma') and s.sigma > 0 else "N/A"),
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
    ax.set_title('模拟参数详情', fontsize=16, y=1.0, pad=20)
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