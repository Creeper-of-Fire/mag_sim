#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# --- 交互式磁场演化对比分析脚本 ---
#
# 功能:
# 1. 交互式选择多个模拟运行进行对比。
# 2. 从 diags/fields/ 目录加载 .npz 文件序列。
# 3. 计算每个时间步的 RMS 磁场和最大磁场。
# 4. 绘制磁场强度随时间的演化图，并与 β≈1 的能量均分场进行对比。
# 5. 附带详细的参数对比表。
#

import os
import glob
import dill
import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy import constants
from dataclasses import dataclass
from typing import List, Optional, Tuple

# --- Rich 库用于漂亮的命令行交互 ---
from rich.console import Console
from rich.table import Table
from rich.prompt import Prompt

# --- 从主分析脚本复制过来的辅助函数和数据结构 ---
# (为了脚本的独立性，我们在这里复制它们)

console = Console()
C = constants.c
M_E = constants.m_e
E = constants.e
J_PER_MEV = E * 1e6


@dataclass
class FieldEvolutionData:
    """存放磁场演化数据"""
    time: np.ndarray
    b_rms_normalized: np.ndarray
    b_max_normalized: np.ndarray


@dataclass
class SimulationRun:
    """存放一次模拟运行的所有相关数据"""
    path: str
    name: str
    sim: object  # 加载自 dill 的模拟参数对象
    field_data: Optional[FieldEvolutionData]


def setup_chinese_font():
    from matplotlib import font_manager as fm
    chinese_fonts_priority = ['WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'Source Han Sans SC', 'SimHei',
                              'Microsoft YaHei']
    found_font = next((font for font in chinese_fonts_priority if fm.findfont(font, fontext='ttf')), None)
    if found_font:
        plt.rcParams['font.sans-serif'] = [found_font]
        console.print(f"[green]✔ Matplotlib 字体已设置为：{found_font}[/green]")
    else:
        console.print("[yellow]⚠ 警告：未能找到支持中文的字体。[/yellow]")
    plt.rcParams['axes.unicode_minus'] = False


def select_directories() -> List[str]:
    # ... (此函数与粒子分析脚本完全相同)
    console.print("\n[bold]扫描当前目录下的有效模拟文件夹...[/bold]")
    valid_dirs = [d.path for d in os.scandir('.') if
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
            prompt_text = "[bold]请输入要对比的模拟索引 (用逗号/空格分隔, [cyan]直接回车则全选[/cyan])[/bold]"
            choice_str = Prompt.ask(prompt_text, default="all")
            if choice_str.strip().lower() == "all":
                console.print(f"[green]已选择全部 {len(valid_dirs)} 个模拟。[/green]")
                return valid_dirs
            indices_str = choice_str.replace(',', ' ').split()
            if not indices_str: continue
            choices = [int(i) for i in indices_str]
            if all(0 <= c < len(valid_dirs) for c in choices):
                return [valid_dirs[c] for c in choices]
            else:
                console.print("[yellow]警告: 输入的索引超出范围，请重试。[/yellow]")
        except ValueError:
            console.print("[red]错误: 无效输入，请输入数字索引。[/red]")


def _prepare_table_data(runs: List[SimulationRun]) -> Tuple[List[str], List[str], List[List[str]]]:
    # ... (此函数与粒子分析脚本完全相同，直接复用)
    headers = ["参数"] + [run.name for run in runs]
    m_e_c2_MeV = (M_E * C ** 2) / J_PER_MEV
    param_map = {
        "--- 归一化 ---": None,
        "B_norm (β ≈ 1, T)": (lambda s: f"{s.B_norm:.2e}" if hasattr(s, 'B_norm') else "未定义"),
        "J_norm (极限电流密度, A/m²)": (lambda s: f"{s.J_norm:.2e}" if hasattr(s, 'J_norm') else "未定义"),

        "--- 物理参数 ---": None, "温度 T (keV)": (lambda s: f"{s.T_plasma / 1e3:.1f}"),
        "总数密度 n (m⁻³)": (lambda s: f"{s.n_plasma:.2e}"),
        "初始重联场 B0 (T)": (lambda s: f"{s.B0:.2f}" if hasattr(s, 'B0') and s.B0 > 0 else "0.0 (无)"),
        "磁化强度 σ": (lambda s: f"{s.sigma:.3f}" if hasattr(s, 'sigma') and s.sigma > 0 else "N/A"),

        "--- 束流参数 ---": None,
        "束流占比": (lambda s: f"{s.beam_fraction * 100:.0f} %" if hasattr(s,
                                                                           'beam_fraction') and s.beam_fraction > 0 else "N/A"),
        "束流 p*c (MeV/c)": (lambda s: f"{(s.beam_u_drift * m_e_c2_MeV):.3f}" if hasattr(s,
                                                                                         'beam_u_drift') and s.beam_fraction > 0 else "N/A"),
        "束流能量 E_k (MeV)": (lambda s: f"{((np.sqrt(1 + s.beam_u_drift ** 2) - 1) * m_e_c2_MeV):.3f}" if hasattr(s,
                                                                                                                   'beam_u_drift') and s.beam_fraction > 0 else "N/A"),
        
        "--- 模拟尺度 ---": None,
        "空间尺度 (m)": (lambda s: f"{s.Lx:.2e} x {s.Lz:.2e}"),
        "时间跨度 (s)": (lambda s: f"{s.total_steps * s.dt:.2e}"),
        "--- 数值参数 ---": None, "网格": (lambda s: f"{s.NX} x {s.NZ}"),
        "每单元粒子数 (NPPC)": (lambda s: f"{s.NPPC}"),
    }
    rows, cell_text = list(param_map.keys()), []
    for param_name in rows:
        row_data = []
        if param_map[param_name] is None:
            cell_text.append([''] * len(runs))
            continue
        for run in runs:
            try:
                formatter = param_map[param_name]
                if hasattr(run, 'sim'):
                    row_data.append(formatter(run.sim))
                else:
                    row_data.append("N/A")
            except (AttributeError, TypeError):
                row_data.append("N/A")
        cell_text.append(row_data)
    formatted_rows = [f"  {r}" if "---" not in r else r for r in rows]
    return headers, formatted_rows, cell_text


# =============================================================================
# 核心数据加载与绘图函数 (针对磁场)
# =============================================================================

def _center_field(field: np.ndarray, target_shape: tuple) -> np.ndarray:
    """将一个交错网格上的场分量插值到单元中心。"""
    # 已经是目标形状，无需操作
    if field.shape == target_shape:
        return field

    nx, nz = target_shape

    # 假设 Bx 的形状是 (nx, nz+1) -> 在 z 方向插值
    if field.shape == (nx, nz + 1):
        return 0.5 * (field[:, :-1] + field[:, 1:])

    # 假设 Bz 的形状是 (nx+1, nz) -> 在 x 方向插值
    elif field.shape == (nx + 1, nz):
        return 0.5 * (field[:-1, :] + field[1:, :])

    # 假设 By 的形状是 (nx+1, nz+1) -> 在两个方向插值
    elif field.shape == (nx + 1, nz + 1):
        # 先在 x 方向插值
        field_x_centered = 0.5 * (field[:-1, :] + field[1:, :])
        # 再在 z 方向插值
        return 0.5 * (field_x_centered[:, :-1] + field_x_centered[:, 1:])

    else:
        # 如果遇到未知的形状，打印警告并尝试用切片强制匹配
        print(f"Warning: Unknown field shape {field.shape}. Attempting to crop to {target_shape}.")
        return field[:nx, :nz]


def load_field_evolution_data(dir_path: str, sim_obj: object) -> Optional[FieldEvolutionData]:
    """从 .npz 文件序列中加载磁场演化数据。"""
    field_files = sorted(glob.glob(os.path.join(dir_path, "diags/fields", "fields_*.npz")))
    if not field_files:
        console.print(f"  [yellow]⚠ 在 'diags/fields/' 目录下找不到任何 .npz 文件。[/yellow]")
        return None

    times, b_rms_vals, b_max_vals = [], [], []
    console.print(f"  [white]正在处理 {len(field_files)} 个磁场数据文件...[/white]")

    # 确定目标中心网格的形状
    target_shape = (sim_obj.NX, sim_obj.NZ)

    for fpath in field_files:
        try:
            step = int(os.path.basename(fpath).split('_')[-1].split('.')[0])

            with np.load(fpath) as data:
                # 获取原始场数据
                Bx_staggered = data['Bx']
                By_staggered = data['By']
                Bz_staggered = data['Bz']

                # 将所有分量插值到单元中心
                Bx = _center_field(Bx_staggered, target_shape)
                By = _center_field(By_staggered, target_shape)
                Bz = _center_field(Bz_staggered, target_shape)

                # 现在所有数组形状都是 (NX, NZ)，可以安全计算
                b_squared = Bx ** 2 + By ** 2 + Bz ** 2
                # --- END MODIFIED SECTION ---

                b_rms_vals.append(np.sqrt(np.mean(b_squared)))
                b_max_vals.append(np.sqrt(np.max(b_squared)))

                # 只有在成功处理数据后才添加时间点
                times.append(step * sim_obj.dt)

        except Exception as e:
            console.print(f"  [red]✗ 处理文件 {os.path.basename(fpath)} 时出错: {e}[/red]")
            # 打印更详细的追溯信息，便于调试
            import traceback
            traceback.print_exc()
            continue

    if not times:
        return None

    return FieldEvolutionData(np.array(times), np.array(b_rms_vals), np.array(b_max_vals))


def generate_field_evolution_plot(runs: List[SimulationRun]):
    """为选定的模拟生成磁场演化对比图。"""
    console.print("\n[bold magenta]正在生成磁场演化对比图...[/bold magenta]")
    output_name = f"field_evolution_{'_vs_'.join([run.name for run in runs])}.png"
    num_runs = len(runs)

    plt.rcParams.update({"font.size": 10})
    fig, (ax_field, ax_table) = plt.subplots(
        2, 1, figsize=(10, 8 + num_runs * 1.5),
        gridspec_kw={'height_ratios': [3, 1 + 0.3 * num_runs]}
    )
    fig.suptitle(f"磁场演化对比: {', '.join([run.name for run in runs])}", fontsize=16, y=0.99)
    ax_field.set_title('归一化磁场强度随时间演化')

    # --- 1. 绘制磁场演化曲线 ---
    cmap = plt.cm.get_cmap('tab10' if num_runs <= 10 else 'viridis')
    colors = [cmap(i / (num_runs - 1)) if num_runs > 1 else cmap(0.5) for i in range(num_runs)]
    if num_runs <= 10: colors = [cmap(i) for i in range(num_runs)]

    for i, run in enumerate(runs):
        if run.field_data:
            ax_field.plot(run.field_data.time, run.field_data.b_rms_normalized, '-', color=colors[i], lw=2,
                          label=f'{run.name} - RMS (均方根)')
            ax_field.plot(run.field_data.time, run.field_data.b_max_normalized, '--', color=colors[i], lw=1.5,
                          label=f'{run.name} - Max (最大值)')

    # --- 关键对比线: 能量均分 ---
    ax_field.axhline(1.0, color='red', linestyle=':', linewidth=2, label='β ≈ 1 (能量均分)')

    ax_field.set_xlabel('时间 (s)')
    ax_field.set_ylabel('磁场强度 B / B_norm')
    ax_field.set_yscale('log')
    ax_field.legend(fontsize=8)
    ax_field.grid(True, which="both", ls="--", alpha=0.5)

    # --- 2. 绘制参数表 ---
    ax_table.axis('off')
    ax_table.set_title('模拟参数对比', y=0.95)
    col_labels, row_labels, cell_text = _prepare_table_data(runs)
    table = ax_table.table(cellText=cell_text, rowLabels=row_labels, colLabels=col_labels[1:], loc='center',
                           cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.8)
    for (row, col), cell in table.get_celld().items():
        if row == 0: cell.set_text_props(weight='bold'); cell.set_facecolor('#B0C4DE')
        if col == -1:
            cell.set_text_props(ha='left', weight='normal')
            if "---" in cell.get_text().get_text(): cell.set_text_props(weight='bold'); cell.set_facecolor('#E0E0E0')

    # --- 3. 保存图像 ---
    plt.tight_layout(rect=[0, 0.02, 1, 0.97])
    plt.savefig(output_name, dpi=200, bbox_inches='tight')
    plt.close(fig)
    console.print(f"[bold green]✔ 磁场演化图已成功保存到: {output_name}[/bold green]")


# =============================================================================
# 主交互流程
# =============================================================================

def main():
    """主执行函数"""
    console.print("[bold inverse] WarpX 磁场演化交互式分析器 [/bold inverse]")
    setup_chinese_font()

    selected_dirs = select_directories()
    if not selected_dirs:
        console.print("\n[yellow]未选择任何目录，程序退出。[/yellow]")
        return

    loaded_runs = []
    for dir_path in selected_dirs:
        console.print(f"\n[bold cyan]正在加载模拟: {os.path.basename(dir_path)}[/bold cyan]")
        param_file = os.path.join(dir_path, "sim_parameters.dpkl")
        try:
            with open(param_file, "rb") as f:
                sim_obj = dill.load(f)
            console.print("  [green]✔ 成功加载参数文件。[/green]")

            field_data = load_field_evolution_data(dir_path, sim_obj)
            loaded_runs.append(SimulationRun(dir_path, os.path.basename(dir_path), sim_obj, field_data))

        except Exception as e:
            console.print(f"  [red]✗ 加载模拟 {os.path.basename(dir_path)} 失败: {e}[/red]")
            continue

    if not any(run.field_data for run in loaded_runs):
        console.print("\n[red]未能成功加载任何磁场演化数据，无法生成图像。[/red]")
        return

    generate_field_evolution_plot(loaded_runs)
    console.print("\n[bold]分析完成。[/bold]")


if __name__ == "__main__":
    main()
