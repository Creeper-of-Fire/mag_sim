#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# --- 交互式多重模拟对比分析脚本 ---
#
# 功能:
# 1. 使用 Rich 库提供一个美观的交互式命令行界面。
# 2. 自动扫描当前目录下的子文件夹，识别有效的 WarpX 模拟运行。
# 3. 允许用户选择多个模拟进行并行对比。
# 4. 生成一张包含 "2*n" 条曲线的能谱对比图 (每个模拟的初始和最终能谱)。
# 5. 在图下方附带一个详细的、对齐的参数对比表。
# 6. 自动计算并显示老师关心的关键物理量（空间/时间尺度、总粒子数、束流能量）。
#

import os
import glob
import dill
import h5py
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy import constants
from scipy.special import kv
import matplotlib.font_manager as fm
from dataclasses import dataclass
from typing import List, Optional, Tuple

# --- Rich 库用于漂亮的命令行交互 ---
from rich.console import Console
from rich.table import Table
from rich.prompt import Prompt


# --- 数据结构定义 ---
@dataclass
class SpectrumData:
    """存放能谱数据"""
    energies_MeV: np.ndarray
    weights: np.ndarray


@dataclass
class SimulationRun:
    """存放一次模拟运行的所有相关数据"""
    path: str
    name: str
    sim: object  # 加载自 dill 的模拟参数对象
    initial_spectrum: Optional[SpectrumData]
    final_spectrum: Optional[SpectrumData]


# --- 全局常量和控制台 ---
console = Console()
C = constants.c
M_E = constants.m_e
E = constants.e
J_PER_MEV = E * 1e6


# =============================================================================
# 1. 辅助函数 (字体设置, 理论分布)
# =============================================================================

def setup_chinese_font():
    """自动查找并设置支持中文的字体。"""
    chinese_fonts_priority = ['WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'Source Han Sans SC', 'SimHei',
                              'Microsoft YaHei']
    found_font = next((font for font in chinese_fonts_priority if fm.findfont(font, fontext='ttf')), None)
    if found_font:
        plt.rcParams['font.sans-serif'] = [found_font]
        console.print(f"[green]✔ Matplotlib 字体已设置为：{found_font}[/green]")
    else:
        console.print("[yellow]⚠ 警告：未能找到支持中文的字体。图表中的中文可能无法正常显示。[/yellow]")
    plt.rcParams['axes.unicode_minus'] = False


def get_maxwell_juttner_distribution(E_bins_J: np.ndarray, T_J: float) -> np.ndarray:
    """计算相对论麦克斯韦-朱特纳分布的概率密度函数 (PDF)。"""
    if T_J <= 0: return np.zeros_like(E_bins_J)
    m_e_c2 = M_E * C ** 2
    theta = T_J / m_e_c2
    gamma = 1.0 + E_bins_J / m_e_c2
    pc = np.sqrt(E_bins_J * (E_bins_J + 2 * m_e_c2))
    # 归一化因子 K_2 是第二类修正贝塞尔函数
    normalization = 1.0 / (m_e_c2 * theta * kv(2, 1.0 / theta))
    return normalization * (pc / m_e_c2) * gamma * np.exp(-gamma / theta)


# =============================================================================
# 2. 数据加载核心函数
# =============================================================================

def _load_spectrum_from_file(h5_filepath: str) -> Optional[SpectrumData]:
    """从单个 HDF5 文件中加载所有带电粒子的能谱。"""
    all_energies_MeV, all_weights = [], []
    m_e_c2_J = M_E * C ** 2

    try:
        with h5py.File(h5_filepath, 'r') as f:
            step_key = list(f['data'].keys())[0]
            particles_group = f[f'data/{step_key}/particles']
            species_in_file = list(particles_group.keys())
            charged_species = [s for s in species_in_file if 'photon' not in s]

            if not charged_species:
                console.print(f"  [yellow]  -> ⚠ 在文件 {os.path.basename(h5_filepath)} 中未找到带电粒子。[/yellow]")
                return None

            for species in charged_species:
                base_path = f"data/{step_key}/particles/{species}/"

                # 定义所需数据集的完整路径
                px_path = base_path + 'momentum/x'
                py_path = base_path + 'momentum/y'
                pz_path = base_path + 'momentum/z'
                w_path = base_path + 'weighting'
                required_datasets = [px_path, py_path, pz_path, w_path]

                # 在读取前，检查所有路径是否存在
                if not all(p in f and isinstance(f[p], h5py.Dataset) for p in required_datasets):
                    console.print(f"  [yellow]  -> ⚠ 物种 '{species}' 缺少必要的数据集(Dataset)，已跳过。[/yellow]")
                    continue

                # 只有当所有数据集都存在时，才加载数据
                px, py, pz = f[px_path][:], f[py_path][:], f[pz_path][:]
                weights = f[w_path][:]

                # 如果权重为空（没有粒子），也跳过
                if weights.size == 0:
                    continue

                p_sq = px ** 2 + py ** 2 + pz ** 2
                kinetic_energy_J = np.sqrt(p_sq * C ** 2 + m_e_c2_J ** 2) - m_e_c2_J
                all_energies_MeV.append(kinetic_energy_J / J_PER_MEV)
                all_weights.append(weights)

        if not all_energies_MeV:
            console.print(f"  [yellow]  -> ⚠ 文件 {os.path.basename(h5_filepath)} 中没有可用的粒子能谱数据。[/yellow]")
            return None

        return SpectrumData(np.concatenate(all_energies_MeV), np.concatenate(all_weights))

    except Exception as e:
        console.print(f"  [red]  -> ✗ 加载能谱 {os.path.basename(h5_filepath)} 时发生意外错误: {e}[/red]")
        return None


def load_simulation_data(dir_path: str) -> Optional[SimulationRun]:
    """加载一个模拟文件夹中的所有必要数据。"""
    console.print(f"\n[bold cyan]正在加载模拟: {os.path.basename(dir_path)}[/bold cyan]")
    param_file = os.path.join(dir_path, "sim_parameters.dpkl")
    if not os.path.exists(param_file):
        console.print(f"  [red]✗ 错误: 找不到参数文件 '{param_file}'。跳过此目录。[/red]")
        return None

    try:
        with open(param_file, "rb") as f:
            sim_obj = dill.load(f)
        console.print("  [green]✔ 成功加载参数文件。[/green]")
    except Exception as e:
        console.print(f"  [red]✗ 错误: 加载 '{param_file}' 时出错: {e}。[/red]")
        return None

    particle_files = sorted(glob.glob(os.path.join(dir_path, "diags/particle_states", "openpmd_*.h5")))
    if not particle_files:
        console.print("  [yellow]⚠ 警告: 在 'diags/particle_states/' 目录下找不到任何 HDF5 文件。[/yellow]")
        return SimulationRun(dir_path, os.path.basename(dir_path), sim_obj, None, None)

    console.print("  [white]正在加载初始能谱 (第一个文件)...[/white]")
    initial_spectrum = _load_spectrum_from_file(particle_files[0])
    final_spectrum = None
    if len(particle_files) > 1:
        console.print("  [white]正在加载最终能谱 (最后一个文件)...[/white]")
        final_spectrum = _load_spectrum_from_file(particle_files[-1])
    else:
        console.print("  [yellow]⚠ 只有一个数据文件，最终能谱将与初始能谱相同。[/yellow]")
        final_spectrum = initial_spectrum

    return SimulationRun(dir_path, os.path.basename(dir_path), sim_obj, initial_spectrum, final_spectrum)


# =============================================================================
# 3. 绘图与分析核心函数
# =============================================================================

def _prepare_table_data(runs: List[SimulationRun]) -> Tuple[List[str], List[str], List[List[str]]]:
    """准备 Matplotlib 表格所需的数据，并计算特殊参数。"""
    headers = ["参数"] + [run.name for run in runs]

    m_e_c2_MeV = (M_E * C ** 2) / J_PER_MEV  # ~0.511 MeV

    # 定义要显示的参数和格式化函数
    param_map = {
        "--- 归一化 ---": None,
        "B_norm (β ≈ 1, T)": (lambda s: f"{s.B_norm:.2e}" if hasattr(s, 'B_norm') else "未定义"),
        "J_norm (极限电流密度, A/m²)": (lambda s: f"{s.J_norm:.2e}" if hasattr(s, 'J_norm') else "未定义"),

        "--- 物理参数 ---": None,
        "温度 T (keV)": (lambda s: f"{s.T_plasma / 1e3:.1f}"),
        "总数密度 n (m^-3)": (lambda s: f"{s.n_plasma:.2e}"),
        "初始重联场 B0 (T)": (lambda s: f"{s.B0:.2f}" if hasattr(s, 'B0') and s.B0 > 0 else "0.0 (无)"),
        "磁化强度 σ": (lambda s: f"{s.sigma:.3f}" if hasattr(s, 'sigma') and s.sigma > 0 else "N/A"),

        "--- 束流参数 ---": None,
        "束流占比": (lambda s: f"{s.beam_fraction * 100:.0f} %" if hasattr(s,
                                                                           'beam_fraction') and s.beam_fraction > 0 else "N/A"),
        "束流 p*c (MeV/c)": (
            lambda s: f"{(s.beam_u_drift * m_e_c2_MeV):.3f}" if hasattr(s,
                                                                        'beam_u_drift') and s.beam_fraction > 0 else "N/A"),
        "束流能量 E_k (MeV)": (
            lambda s: f"{((np.sqrt(1 + s.beam_u_drift ** 2) - 1) * (M_E * C ** 2 / J_PER_MEV)):.3f}" if hasattr(s,
                                                                                                                'beam_u_drift') and s.beam_fraction > 0 else "N/A"),
        "--- 真实尺寸 ---": None,
        "空间尺度 (m)": (lambda s: f"{s.Lx:.2e} x {s.Lz:.2e}"),
        "时间跨度 (s)": (lambda s: f"{s.total_steps * s.dt:.2e}"),
        "总粒子数 (加权)": "dynamic",  # 特殊处理
        "--- 数值参数 ---": None,
        "网格": (lambda s: f"{s.NX} x {s.NZ}"),
        "每单元粒子数 (NPPC)": (lambda s: f"{s.NPPC}"),
    }

    rows = list(param_map.keys())
    cell_text = []

    for param_name in rows:
        row_data = []
        if param_map[param_name] is None:  # 分隔符
            cell_text.append([''] * len(runs))
            continue

        for run in runs:
            if param_map[param_name] == "dynamic":
                # 动态计算总粒子数
                if run.initial_spectrum:
                    total_particles = np.sum(run.initial_spectrum.weights)
                    row_data.append(f"{total_particles:.2e}")
                else:
                    row_data.append("N/A")
            else:
                try:
                    formatter = param_map[param_name]
                    row_data.append(formatter(run.sim))
                except AttributeError:
                    row_data.append("N/A")
        cell_text.append(row_data)

    # 调整行标签以增加可读性
    formatted_rows = [f"  {r}" if "---" not in r else r for r in rows]
    return headers, formatted_rows, cell_text


def generate_comparison_plot(runs: List[SimulationRun]):
    """为选定的模拟运行生成一张包含能谱和参数表的摘要图。"""
    console.print("\n[bold magenta]正在生成对比图...[/bold magenta]")

    num_runs = len(runs)
    output_name = f"comparison_{'_vs_'.join([run.name for run in runs])}.png"

    plt.rcParams.update({"font.size": 10})
    fig, (ax_spec, ax_table) = plt.subplots(
        2, 1, figsize=(10, 8 + num_runs * 1.5),
        gridspec_kw={'height_ratios': [3, 1 + 0.3 * num_runs]}
    )
    fig.suptitle(f"模拟对比: {', '.join([run.name for run in runs])}", fontsize=16, y=0.99)
    ax_spec.set_title('能谱演化对比')

    # --- 1. 绘制能谱 ---
    if len(runs) <= 10:
        cmap = matplotlib.colormaps.get_cmap('tab10')
        colors = [cmap(i) for i in range(len(runs))]
    else:  # 如果超过10个，就从连续谱中采样
        cmap = matplotlib.colormaps.get_cmap('viridis')
        colors = [cmap(i / len(runs)) for i in range(len(runs))]

    all_energies = []
    for run in runs:
        if run.initial_spectrum: all_energies.append(run.initial_spectrum.energies_MeV)
        if run.final_spectrum: all_energies.append(run.final_spectrum.energies_MeV)

    if not all_energies:
        ax_spec.text(0.5, 0.5, '所有模拟均无能谱数据', ha='center', va='center', color='red')
    else:
        combined_energies = np.concatenate(all_energies)
        positive_energies = combined_energies[combined_energies > 0]

        if positive_energies.size > 1:
            # 使用对数分箱以更好地展示高能尾部
            num_bins = 150
            min_E = max(positive_energies.min() * 0.5, 1e-4)
            max_E = positive_energies.max() * 1.2
            common_bins_MeV = np.logspace(np.log10(min_E), np.log10(max_E), num_bins + 1)
            bin_centers_MeV = np.sqrt(common_bins_MeV[:-1] * common_bins_MeV[1:])
            bin_widths_MeV = np.diff(common_bins_MeV)

            for i, run in enumerate(runs):
                color = colors[i]

                # 绘制初始谱: 使用虚线，并过滤掉0值点
                if run.initial_spectrum:
                    counts, _ = np.histogram(run.initial_spectrum.energies_MeV, bins=common_bins_MeV,
                                             weights=run.initial_spectrum.weights)
                    dN_dE = counts / bin_widths_MeV
                    # 只选择 dN/dE > 0 的点进行连接
                    valid_mask = dN_dE > 0
                    ax_spec.plot(bin_centers_MeV[valid_mask], dN_dE[valid_mask],
                                 linestyle='--',  # 虚线
                                 color=color,
                                 lw=1.5,  # Line width
                                 label=f'{run.name} - 初始')

                # 绘制最终谱: 使用实线，并过滤掉0值点
                if run.final_spectrum:
                    counts, _ = np.histogram(run.final_spectrum.energies_MeV, bins=common_bins_MeV,
                                             weights=run.final_spectrum.weights)
                    dN_dE = counts / bin_widths_MeV
                    # 同样，只选择 dN/dE > 0 的点
                    valid_mask = dN_dE > 0
                    ax_spec.plot(bin_centers_MeV[valid_mask], dN_dE[valid_mask],
                                 linestyle='-',  # 实线
                                 color=color,
                                 lw=2.0,
                                 label=f'{run.name} - 最终')

            # 绘制理论麦克斯韦-朱特纳分布作为参考 (使用第一个模拟的参数)
            ref_run = runs[0]
            if ref_run.initial_spectrum:
                T_plasma_J = ref_run.sim.T_plasma * E
                # 理论谱的归一化系数是加权后的总粒子数
                total_thermal_particles = np.sum(ref_run.initial_spectrum.weights) * (1.0 - ref_run.sim.beam_fraction)
                pdf_juttner_per_J = get_maxwell_juttner_distribution(bin_centers_MeV * J_PER_MEV, T_plasma_J)
                dN_dE_juttner = total_thermal_particles * pdf_juttner_per_J * J_PER_MEV
                ax_spec.plot(bin_centers_MeV, dN_dE_juttner, '--', color='black', alpha=0.7,
                             label=f'理论热谱 (T={ref_run.sim.T_plasma / 1e3:.1f} keV)')

    ax_spec.set_xlabel('动能 (MeV)')
    ax_spec.set_ylabel('粒子数谱密度 (dN/dE [MeV^-1])')
    ax_spec.set_xscale('log')
    ax_spec.set_yscale('log')
    ax_spec.legend(fontsize=8)
    # ax_spec.grid(True, which="both", ls="--", alpha=0.5)

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
        if row == 0:
            cell.set_text_props(weight='bold')
            cell.set_facecolor('#B0C4DE')
        if col == -1:
            cell.set_text_props(ha='left', weight='normal')
            if "---" in cell.get_text().get_text():
                cell.set_text_props(weight='bold')
                cell.set_facecolor('#E0E0E0')

    # --- 3. 保存图像 ---
    plt.tight_layout(rect=[0, 0.02, 1, 0.97])
    plt.savefig(output_name, dpi=200, bbox_inches='tight')
    plt.close(fig)
    console.print(f"[bold green]✔ 对比图已成功保存到: {output_name}[/bold green]")


# =============================================================================
# 4. 主交互流程
# =============================================================================

def select_directories() -> List[str]:
    """扫描并让用户选择要分析的目录。"""
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
            # 如果用户输入了内容但解析后为空（例如只输入了空格），则重新提示
            if not indices_str:
                continue

            choices = [int(i) for i in indices_str]

            if all(0 <= c < len(valid_dirs) for c in choices):
                return [valid_dirs[c] for c in choices]
            else:
                console.print("[yellow]警告: 输入的索引超出范围，请重试。[/yellow]")
        except ValueError:
            console.print("[red]错误: 无效输入，请输入数字索引。[/red]")


def main():
    """主执行函数"""
    console.print("[bold inverse] WarpX 多重模拟交互式分析器 [/bold inverse]")
    setup_chinese_font()

    selected_dirs = select_directories()
    if not selected_dirs:
        console.print("\n[yellow]未选择任何目录，程序退出。[/yellow]")
        return

    loaded_runs = []
    for dir_path in selected_dirs:
        run_data = load_simulation_data(dir_path)
        if run_data:
            loaded_runs.append(run_data)

    if not loaded_runs:
        console.print("\n[red]未能成功加载任何模拟数据，无法生成图像。[/red]")
        return

    generate_comparison_plot(loaded_runs)
    console.print("\n[bold]分析完成。[/bold]")


if __name__ == "__main__":
    main()
