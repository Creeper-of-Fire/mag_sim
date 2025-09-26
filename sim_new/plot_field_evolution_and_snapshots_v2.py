#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# --- 交互式磁场演化与快照对比分析脚本 ---
#
# 功能:
# 1. 交互式选择多个模拟运行进行对比。
# 2. 从 diags/fields/ 目录加载 .npz 文件序列。
# 3. 计算每个时间步的平均磁场和最大磁场。
# 4. 绘制磁场强度随时间的演化图，并与 β≈1 的能量均分场进行对比。
# 5. 为每次模拟生成包含多个时间点的磁场强度 |B| 空间分布快照图。
# 6. 附带详细的参数对比表。
#
import hashlib
import os
import glob
import dill
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
    b_mean_abs_normalized: np.ndarray  # 平均磁场强度 ⟨|B|⟩
    b_max_normalized: np.ndarray     # 最大磁场强度 max(|B|)
    b_mean_x_normalized: np.ndarray    # 平均X分量 ⟨Bx⟩
    b_mean_y_normalized: np.ndarray    # 平均Y分量 ⟨By⟩
    b_mean_z_normalized: np.ndarray    # 平均Z分量 ⟨Bz⟩


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
    # ... (此函数与原脚本完全相同，无需修改)
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
    # ... (此函数与原脚本完全相同，无需修改)
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
# 核心数据加载与绘图函数
# =============================================================================

def _center_field(field: np.ndarray, target_shape: tuple) -> np.ndarray:
    """将一个交错网格上的场分量插值到单元中心。"""
    # (此函数与原脚本完全相同，无需修改)
    if field.shape == target_shape:
        return field
    nx, nz = target_shape
    if field.shape == (nx, nz + 1):
        return 0.5 * (field[:, :-1] + field[:, 1:])
    elif field.shape == (nx + 1, nz):
        return 0.5 * (field[:-1, :] + field[1:, :])
    elif field.shape == (nx + 1, nz + 1):
        field_x_centered = 0.5 * (field[:-1, :] + field[1:, :])
        return 0.5 * (field_x_centered[:, :-1] + field_x_centered[:, 1:])
    else:
        print(f"Warning: Unknown field shape {field.shape}. Attempting to crop to {target_shape}.")
        return field[:nx, :nz]

def get_centered_magnetic_field(fpath: str, target_shape: tuple) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """从单个 npz 文件加载并返回中心化的 Bx, By, Bz。"""
    with np.load(fpath) as data:
        Bx_staggered = data['Bx']
        By_staggered = data['By']
        Bz_staggered = data['Bz']
        Bx = _center_field(Bx_staggered, target_shape)
        By = _center_field(By_staggered, target_shape)
        Bz = _center_field(Bz_staggered, target_shape)
        return Bx, By, Bz

def load_field_evolution_data(dir_path: str, sim_obj: object) -> Optional[FieldEvolutionData]:
    """
    从 .npz 文件序列中加载磁场演化数据。
    此版本计算 Bx, By, Bz 各分量的空间平均值。
    """
    field_files = sorted(glob.glob(os.path.join(dir_path, "diags/fields", "fields_*.npz")))
    if not field_files:
        console.print(f"  [yellow]⚠ 在 'diags/fields/' 目录下找不到任何 .npz 文件。[/yellow]")
        return None

    # --- 数据列表初始化：现在专注于各分量的平均值 ---
    times, b_max_vals = [], []
    b_mean_x_vals, b_mean_y_vals, b_mean_z_vals = [], [], []
    # (b_mean_abs_vals 不再是主要关注点，但可以保留用于其他分析)
    b_mean_abs_vals = []

    console.print(f"  [white]正在处理 {len(field_files)} 个磁场数据文件...[/white]")
    target_shape = (sim_obj.NX, sim_obj.NZ)

    for fpath in field_files:
        try:
            step = int(os.path.basename(fpath).split('_')[-1].split('.')[0])
            Bx, By, Bz = get_centered_magnetic_field(fpath, target_shape)

            # --- 核心修改：计算每个分量的平均值 ---
            # 这些是有符号的平均值，可以揭示场的净方向
            b_mean_x_vals.append(np.mean(Bx))
            b_mean_y_vals.append(np.mean(By))
            b_mean_z_vals.append(np.mean(Bz))

            # 同时保留最大值和幅值平均值的计算，以供参考
            b_magnitude = np.sqrt(Bx ** 2 + By ** 2 + Bz ** 2)
            b_max_vals.append(np.max(b_magnitude))
            b_mean_abs_vals.append(np.mean(b_magnitude)) # 幅值的平均值

            times.append(step * sim_obj.dt)

        except Exception as e:
            console.print(f"  [red]✗ 处理文件 {os.path.basename(fpath)} 时出错: {e}[/red]")
            continue

    if not times:
        return None

    # --- 返回包含各分量平均值的数据结构 ---
    return FieldEvolutionData(
        time=np.array(times),
        b_max_normalized=np.array(b_max_vals),
        b_mean_x_normalized=np.array(b_mean_x_vals),
        b_mean_y_normalized=np.array(b_mean_y_vals),
        b_mean_z_normalized=np.array(b_mean_z_vals),
        b_mean_abs_normalized=np.array(b_mean_abs_vals)
    )


def _generate_safe_filename(prefix: str, runs: List[SimulationRun], max_length: int = 32) -> str:
    """
    生成一个安全的文件名，如果描述性名称太长，则使用哈希值缩短它。

    Args:
        prefix (str): 文件名的前缀 (例如 'field_components_evolution').
        runs (List[SimulationRun]): 正在比较的 SimulationRun 对象列表.
        max_length (int): 文件名的最大安全长度.

    Returns:
        str: 一个不会超长的安全文件名.
    """
    # 1. 尝试创建完整的描述性名称
    descriptive_part = '_vs_'.join(sorted([run.name for run in runs]))  # 排序以保证同样组合下名称一致
    ideal_name = f"{prefix}_{descriptive_part}.png"

    # 2. 检查长度，如果不超长，直接返回
    if len(ideal_name) < max_length:
        return ideal_name

    # 3. 如果太长，创建一个更短的、基于哈希的名称
    console.print(f"  [yellow]⚠ 组合文件名过长，将生成一个简短的哈希文件名。[/yellow]")

    # 使用所有运行名称的组合来生成一个唯一的哈希ID
    # encode() 是必需的，因为哈希函数处理的是字节
    hasher = hashlib.md5(descriptive_part.encode('utf-8'))
    hash_id = hasher.hexdigest()[:8]  # 取前8位作为唯一标识符通常足够了

    # 生成简短的文件名
    short_name = f"{prefix}_comparison_of_{len(runs)}_runs_{hash_id}.png"

    return short_name

def generate_field_evolution_plot(runs: List[SimulationRun]):
    """
    为选定的模拟生成磁场各分量演化的对比图，以研究各向异性。
    """
    console.print("\n[bold magenta]正在生成磁场分量演化对比图...[/bold magenta]")
    output_name = _generate_safe_filename("field_components_evolution", runs)
    num_runs = len(runs)

    plt.rcParams.update({"font.size": 10})
    fig, (ax_field, ax_table) = plt.subplots(
        2, 1, figsize=(10, 8 + num_runs * 1.5),
        gridspec_kw={'height_ratios': [3, 1 + 0.3 * num_runs]}
    )
    fig.suptitle(f"磁场各向异性演化对比: {', '.join([run.name for run in runs])}", fontsize=16, y=0.99)
    ax_field.set_title('磁场各分量平均值 <B> 随时间演化')

    cmap = plt.cm.get_cmap('tab10' if num_runs <= 10 else 'viridis')
    colors = [cmap(i) for i in range(num_runs)]

    # --- 核心修改: 绘制 Bx, By, Bz 各分量的平均值 ---
    for i, run in enumerate(runs):
        if run.field_data:
            # 使用不同的线型来区分 Bx, By, Bz
            # 同一个 run 的不同分量使用相同的颜色
            ax_field.plot(run.field_data.time, run.field_data.b_mean_x_normalized, '-', color=colors[i], lw=2,
                          label=f'{run.name} - <Bx>')
            ax_field.plot(run.field_data.time, run.field_data.b_mean_y_normalized, '--', color=colors[i], lw=1.5,
                          label=f'{run.name} - <By>')
            ax_field.plot(run.field_data.time, run.field_data.b_mean_z_normalized, ':', color=colors[i], lw=1.5,
                          label=f'{run.name} - <Bz>')

            # # (可选) 仍然可以绘制最大值作为参考，但使用较低的透明度
            # ax_field.plot(run.field_data.time, run.field_data.b_max_normalized, ls='-', color=colors[i], lw=1, alpha=0.3,
            #               label=f'{run.name} - Max |B|')

    # --- 修改坐标轴和图例以适应新数据 ---
    ax_field.axhline(0.0, color='black', linestyle='-', linewidth=1, alpha=0.7) # 添加 y=0 参考线
    ax_field.set_xlabel('时间 (s)')
    ax_field.set_ylabel('平均磁场分量 <B> / B_norm')

    # 因为平均值可以为负，所以必须使用线性坐标轴，而不是对数坐标轴
    ax_field.set_yscale('linear')

    ax_field.legend(fontsize=8, ncol=max(1, num_runs)) # 使用多列图例防止重叠
    ax_field.grid(True, which="both", ls="--", alpha=0.5)

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

    plt.tight_layout(rect=[0, 0.02, 1, 0.97])
    plt.savefig(output_name, dpi=200, bbox_inches='tight')
    plt.close(fig)
    console.print(f"[bold green]✔ 磁场演化图已成功保存到: {output_name}[/bold green]")


# --- 新增功能: 绘制磁场快照 ---
def generate_field_snapshots_plot(runs: List[SimulationRun]):
    """为每次模拟生成包含 4 个时间点的磁场强度 |B| 快照图。"""
    console.print("\n[bold magenta]正在生成磁场快照图...[/bold magenta]")

    for run in runs:
        console.print(f"  [white]为 '{run.name}' 生成快照...[/white]")
        field_files = sorted(glob.glob(os.path.join(run.path, "diags/fields", "fields_*.npz")))
        if len(field_files) < 4:
            console.print(f"  [yellow]⚠ 快照文件不足4个，跳过 '{run.name}'。[/yellow]")
            continue

        num_files = len(field_files)
        indices = np.linspace(0, num_files - 1, 4, dtype=int)
        selected_files = [field_files[i] for i in indices]

        # --- GridSpec 布局核心 ---
        # 创建一个稍宽的图像，为颜色条留出空间
        fig = plt.figure(figsize=(13, 10))
        fig.suptitle(f'磁场强度 |B|/B_norm 快照: {run.name}', fontsize=16)

        # 定义一个 2x3 的网格。前两列用于绘图，最后一列专门给颜色条。
        # width_ratios 控制列的相对宽度，让颜色条列很窄。
        # wspace/hspace 控制子图间的间距。
        gs = fig.add_gridspec(2, 3, width_ratios=[15, 15, 1], wspace=0.1, hspace=0.15)

        # 手动创建 4 个子图和一个颜色条的坐标轴
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[1, 0])
        ax4 = fig.add_subplot(gs[1, 1])
        cax = fig.add_subplot(gs[:, 2]) # 颜色条轴，跨越所有行

        axes = [ax1, ax2, ax3, ax4]

        # --- 坐标轴共享与美化 ---
        # 共享坐标轴，这样缩放和平移会同步
        ax1.sharex(ax3)
        ax2.sharex(ax4)
        ax1.sharey(ax2)
        ax3.sharey(ax4)

        # 隐藏内部子图多余的刻度标签，让图像更整洁
        # 上排子图 (ax1, ax2) 不显示 X 轴刻度标签
        ax1.tick_params(axis='x', labelbottom=False)
        ax2.tick_params(axis='x', labelbottom=False)
        # 右列子图 (ax2, ax4) 不显示 Y 轴刻度标签
        ax2.tick_params(axis='y', labelleft=False)
        ax4.tick_params(axis='y', labelleft=False)
        # --- 布局结束 ---

        vmax = 0
        target_shape = (run.sim.NX, run.sim.NZ)
        for fpath in selected_files:
             Bx, By, Bz = get_centered_magnetic_field(fpath, target_shape)
             b_mag = np.sqrt(Bx**2 + By**2 + Bz**2)
             vmax = max(vmax, np.max(b_mag))

        im = None # 初始化 im
        for i, fpath in enumerate(selected_files):
            ax = axes[i]
            step = int(os.path.basename(fpath).split('_')[-1].split('.')[0])
            time_s = step * run.sim.dt

            Bx, By, Bz = get_centered_magnetic_field(fpath, target_shape)
            b_mag = np.sqrt(Bx**2 + By**2 + Bz**2)

            # extent 参数是关键，它将数组索引映射到物理坐标
            im = ax.imshow(b_mag.T, origin='lower',
                           extent=[0, run.sim.Lx, 0, run.sim.Lz],
                           aspect='auto', cmap='viridis', vmin=0, vmax=vmax)

            ax.set_title(f'时间 = {time_s:.2e} s')
            # 只在最外围的子图上设置坐标轴标签 (x/y label)
            if i >= 2:  # 底部子图 (ax3, ax4)
                ax.set_xlabel('x (m)')
            if i % 2 == 0:  # 左侧子图 (ax1, ax3)
                ax.set_ylabel('z (m)')

        # 将颜色条绘制到预留的专属坐标轴 cax 上
        fig.colorbar(im, cax=cax, label='|B| (T)')

        # 调整总标题的位置，防止重叠
        fig.subplots_adjust(top=0.94)

        output_name = f"field_snapshots_{run.name}.png"
        plt.savefig(output_name, dpi=200)
        plt.close(fig)
        console.print(f"  [bold green]✔ 快照图已保存到: {output_name}[/bold green]")


def generate_field_streamlines_plot(runs: List[SimulationRun]):
    """为每次模拟生成包含 4 个时间点的磁场流线图。"""
    console.print("\n[bold magenta]正在生成磁场流线图...[/bold magenta]")

    for run in runs:
        console.print(f"  [white]为 '{run.name}' 生成流线图...[/white]")
        field_files = sorted(glob.glob(os.path.join(run.path, "diags/fields", "fields_*.npz")))
        if len(field_files) < 4:
            console.print(f"  [yellow]⚠ 快照文件不足4个，跳过 '{run.name}' 的流线图绘制。[/yellow]")
            continue

        num_files = len(field_files)
        indices = np.linspace(0, num_files - 1, 4, dtype=int)
        selected_files = [field_files[i] for i in indices]

        fig = plt.figure(figsize=(13, 10))
        fig.suptitle(f'平面内磁场线 (Bx, Bz) 与总场强 |B|: {run.name}', fontsize=16)

        gs = fig.add_gridspec(2, 3, width_ratios=[15, 15, 1], wspace=0.1, hspace=0.15)

        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[1, 0])
        ax4 = fig.add_subplot(gs[1, 1])
        cax = fig.add_subplot(gs[:, 2])
        axes = [ax1, ax2, ax3, ax4]

        # --- 坐标轴共享与美化 ---
        # 共享坐标轴，这样缩放和平移会同步
        ax1.sharex(ax3)
        ax2.sharex(ax4)
        ax1.sharey(ax2)
        ax3.sharey(ax4)

        # 隐藏内部子图多余的刻度标签，让图像更整洁
        # 上排子图 (ax1, ax2) 不显示 X 轴刻度标签
        ax1.tick_params(axis='x', labelbottom=False)
        ax2.tick_params(axis='x', labelbottom=False)
        # 右列子图 (ax2, ax4) 不显示 Y 轴刻度标签
        ax2.tick_params(axis='y', labelleft=False)
        ax4.tick_params(axis='y', labelleft=False)
        # --- 布局结束 ---

        vmax = 0
        target_shape = (run.sim.NX, run.sim.NZ)
        for fpath in selected_files:
            Bx, By, Bz = get_centered_magnetic_field(fpath, target_shape)
            b_mag = np.sqrt(Bx ** 2 + By ** 2 + Bz ** 2)
            vmax = max(vmax, np.max(b_mag))

        im = None
        # 创建网格坐标
        x_coords = np.linspace(0, run.sim.Lx, run.sim.NX)
        z_coords = np.linspace(0, run.sim.Lz, run.sim.NZ)

        for i, fpath in enumerate(selected_files):
            ax = axes[i]
            step = int(os.path.basename(fpath).split('_')[-1].split('.')[0])
            time_s = step * run.sim.dt

            Bx, By, Bz = get_centered_magnetic_field(fpath, target_shape)
            b_mag = np.sqrt(Bx ** 2 + By ** 2 + Bz ** 2)
            b_in_plane_mag = np.sqrt(Bx ** 2 + Bz ** 2)

            # 1. 绘制背景：总磁场强度
            # extent 参数是关键，它将数组索引映射到物理坐标
            im = ax.imshow(b_mag.T, origin='lower',
                           extent=[0, run.sim.Lx, 0, run.sim.Lz],
                           aspect='auto', cmap='viridis', vmin=0, vmax=vmax)

            # 2. 绘制前景：磁场流线
            linewidth = 2.5 * b_in_plane_mag / (b_in_plane_mag.max() + 1e-10)
            ax.streamplot(x_coords, z_coords, Bx.T, Bz.T,
                          color='white',
                          linewidth=linewidth.T,
                          density=1.5,
                          arrowstyle='->',
                          arrowsize=0.8)

            ax.set_title(f'时间 = {time_s:.2e} s')
            # 只在最外围的子图上设置坐标轴标签 (x/y label)
            if i >= 2:  # 底部子图 (ax3, ax4)
                ax.set_xlabel('x (m)')
            if i % 2 == 0:  # 左侧子图 (ax1, ax3)
                ax.set_ylabel('z (m)')

            ax.set_title(f'时间 = {time_s:.2e} s')
            if i in [2, 3]: ax.set_xlabel('x (m)')
            if i in [0, 2]: ax.set_ylabel('z (m)')

        fig.colorbar(im, cax=cax, label='总磁场强度 |B| (T)')
        fig.subplots_adjust(top=0.94)

        output_name = f"field_streamlines_{run.name}.png"
        plt.savefig(output_name, dpi=200)
        plt.close(fig)
        console.print(f"  [bold green]✔ 磁场流线图已保存到: {output_name}[/bold green]")

# =============================================================================
# 主交互流程
# =============================================================================

def main():
    """主执行函数"""
    console.print("[bold inverse] WarpX 磁场演化与快照分析器 [/bold inverse]")
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

    # 过滤掉没有成功加载数据的模拟
    valid_runs = [run for run in loaded_runs if run.field_data]
    if not valid_runs:
        console.print("\n[red]未能成功加载任何磁场演化数据，无法生成图像。[/red]")
        return

    # 生成两种图
    generate_field_evolution_plot(valid_runs)
    generate_field_snapshots_plot(valid_runs)
    generate_field_streamlines_plot(valid_runs)

    console.print("\n[bold]分析完成。[/bold]")


if __name__ == "__main__":
    main()