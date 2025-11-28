# core/utils.py

# !/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# --- 通用工具模块 ---
#
# 包含共享的、与具体物理计算无关的辅助函数。
#
import os
from typing import List, Optional

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
# --- Rich 库用于漂亮的命令行交互 ---
from rich.console import Console
from rich.prompt import Prompt
from rich.table import Table

from .config import config

# --- 导入核心数据结构 ---

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
    chinese_fonts_priority = ['WenQuanYi Micro Hei', 'Source Han Sans SC', 'Noto Sans CJK SC', 'SimHei', 'Microsoft YaHei']
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
