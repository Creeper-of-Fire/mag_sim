#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# --- 交互式多重模拟对比分析脚本 (V3 - 独立分析版) ---
#
# 功能:
# 1. 使用 Rich 库提供一个美观的交互式命令行界面。
# 2. 自动扫描当前目录下的子文件夹，识别有效的 WarpX 模拟运行。
# 3. 允许用户选择多个模拟进行分析。
# 4. 对每个选定的模拟：
#    a. 计算并显示最终时刻粒子的加权平均动能。
#    b. 提示用户输入一个外部计算的“真实温度”(keV)。
#    c. 生成一张独立的分析图，包含初始/最终能谱和基于输入温度的理论热谱。
#    d. 在图下方附带该模拟的详细参数表。
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
from dataclasses import dataclass, field
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
    user_T_keV: Optional[float] = field(default=None)  # 新增：用于存储用户输入的温度


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

                px_path = base_path + 'momentum/x'
                py_path = base_path + 'momentum/y'
                pz_path = base_path + 'momentum/z'
                w_path = base_path + 'weighting'
                required_datasets = [px_path, py_path, pz_path, w_path]

                if not all(p in f and isinstance(f[p], h5py.Dataset) for p in required_datasets):
                    console.print(f"  [yellow]  -> ⚠ 物种 '{species}' 缺少必要的数据集(Dataset)，已跳过。[/yellow]")
                    continue

                px, py, pz = f[px_path][:], f[py_path][:], f[pz_path][:]
                weights = f[w_path][:]

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

def _prepare_single_run_table_data(run: SimulationRun) -> List[List[str]]:
    """为单个模拟准备 Matplotlib 表格所需的数据。"""
    m_e_c2_MeV = (M_E * C ** 2) / J_PER_MEV  # ~0.511 MeV

    param_map = {
        "--- 归一化 ---": None,
        "B_norm (β ≈ 1, T)": (lambda s: f"{s.B_norm:.2e}" if hasattr(s, 'B_norm') else "未定义"),
        "J_norm (极限电流密度, A/m²)": (lambda s: f"{s.J_norm:.2e}" if hasattr(s, 'J_norm') else "未定义"),
        "--- 物理参数 ---": None,
        "初始温度 T (keV)": (lambda s: f"{s.T_plasma / 1e3:.1f}"),
        "总数密度 n (m⁻³)": (lambda s: f"{s.n_plasma:.2e}"),
        "初始重联场 B0 (T)": (lambda s: f"{s.B0:.2f}" if hasattr(s, 'B0') and s.B0 > 0 else "0.0 (无)"),
        "磁化强度 σ": (lambda s: f"{s.sigma:.3f}" if hasattr(s, 'sigma') and s.sigma > 0 else "N/A"),
        "--- 束流参数 ---": None,
        "束流占比": (lambda s: f"{s.beam_fraction * 100:.0f} %" if hasattr(s, 'beam_fraction') and s.beam_fraction > 0 else "N/A"),
        "束流 p*c (MeV/c)": (lambda s: f"{(s.beam_u_drift * m_e_c2_MeV):.3f}" if hasattr(s, 'beam_u_drift') and s.beam_fraction > 0 else "N/A"),
        "束流能量 E_k (MeV)": (
            lambda s: f"{((np.sqrt(1 + s.beam_u_drift ** 2) - 1) * m_e_c2_MeV):.3f}" if hasattr(s, 'beam_u_drift') and s.beam_fraction > 0 else "N/A"),
        "--- 真实尺寸 ---": None,
        "空间尺度 (m)": (lambda s: f"{s.Lx:.2e} x {s.Lz:.2e}"),
        "时间跨度 (s)": (lambda s: f"{s.total_steps * s.dt:.2e}"),
        "总粒子数 (加权)": "dynamic",
        "--- 数值参数 ---": None,
        "网格": (lambda s: f"{s.NX} x {s.NZ}"),
        "每单元粒子数 (NPPC)": (lambda s: f"{s.NPPC}"),
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
            except AttributeError:
                pass  # 保持 "N/A"

        table_data.append([f"  {param_name}", value_str])

    return table_data


def generate_individual_plots(runs: List[SimulationRun]):
    """为每个选定的模拟运行生成一张独立的分析图。"""
    console.print("\n[bold magenta]正在为每个模拟生成独立分析图...[/bold magenta]")

    for i, run in enumerate(runs):
        output_name = f"analysis_{run.name}.png"
        console.print(f"\n--- ({i + 1}/{len(runs)}) 正在处理 [bold]{run.name}[/bold] ---")

        # --- 1. 创建 Figure 和布局 ---
        fig, (ax_plot, ax_table) = plt.subplots(2, 1, figsize=(10, 14),
                                                gridspec_kw={'height_ratios': [3, 2]})
        fig.suptitle(f"模拟分析: {run.name}", fontsize=20, y=0.98)

        # --- 2. 绘制能谱图 ---
        ax_plot.set_title("粒子能谱演化", fontsize=16)

        all_energies = []
        if run.initial_spectrum: all_energies.append(run.initial_spectrum.energies_MeV)
        if run.final_spectrum: all_energies.append(run.final_spectrum.energies_MeV)

        if not all_energies:
            ax_plot.text(0.5, 0.5, '无能谱数据', ha='center', va='center', color='red', fontsize=16)
        else:
            combined_energies = np.concatenate(all_energies)
            positive_energies = combined_energies[combined_energies > 0]
            if positive_energies.size > 1:
                num_bins = 200
                min_E = max(positive_energies.min() * 0.5, 1e-4)
                max_E = positive_energies.max() * 1.2
                common_bins_MeV = np.logspace(np.log10(min_E), np.log10(max_E), num_bins + 1)
                bin_centers_MeV = np.sqrt(common_bins_MeV[:-1] * common_bins_MeV[1:])
                bin_widths_MeV = np.diff(common_bins_MeV)

                # 绘制初始能谱
                if run.initial_spectrum:
                    counts, _ = np.histogram(run.initial_spectrum.energies_MeV, bins=common_bins_MeV,
                                             weights=run.initial_spectrum.weights)
                    dN_dE = counts / bin_widths_MeV
                    valid_mask = dN_dE > 0
                    ax_plot.plot(bin_centers_MeV[valid_mask], dN_dE[valid_mask], linestyle='--', color='gray', lw=2,
                                 label='初始')
                # 绘制最终能谱
                if run.final_spectrum:
                    counts, _ = np.histogram(run.final_spectrum.energies_MeV, bins=common_bins_MeV,
                                             weights=run.final_spectrum.weights)
                    dN_dE = counts / bin_widths_MeV
                    valid_mask = dN_dE > 0
                    ax_plot.plot(bin_centers_MeV[valid_mask], dN_dE[valid_mask], linestyle='-', color='royalblue', lw=2.5,
                                 label='最终')

                # 绘制基于用户输入温度的麦克斯韦-朱特纳分布
                if run.user_T_keV is not None and run.user_T_keV > 0 and run.initial_spectrum:
                    T_plasma_J = run.user_T_keV * 1e3 * E
                    total_particles = np.sum(run.initial_spectrum.weights)
                    pdf_juttner_per_J = get_maxwell_juttner_distribution(bin_centers_MeV * J_PER_MEV, T_plasma_J)
                    dN_dE_juttner = total_particles * pdf_juttner_per_J * J_PER_MEV
                    valid_mask = dN_dE_juttner > 0
                    ax_plot.plot(bin_centers_MeV[valid_mask], dN_dE_juttner[valid_mask], ':', color='red', alpha=0.9, lw=2,
                                 label=f'理论热谱 (T={run.user_T_keV:.2f} keV)')

            ax_plot.set_xscale('log')
            ax_plot.set_yscale('log')
            ax_plot.set_xlabel('动能 (MeV)', fontsize=14)
            ax_plot.set_ylabel('粒子数谱密度 (dN/dE [MeV⁻¹])', fontsize=14)
            ax_plot.grid(True, which="both", ls="--", alpha=0.5)
            ax_plot.legend(fontsize=12, loc='best')

        # --- 3. 绘制参数表 ---
        ax_table.axis('off')
        ax_table.set_title('模拟参数详情', fontsize=16, y=1.0, pad=20)

        table_data = _prepare_single_run_table_data(run)
        table = ax_table.table(cellText=table_data,
                               colLabels=['参数', '值'],
                               loc='center',
                               cellLoc='left',
                               colWidths=[0.4, 0.4])

        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 2.0)  # 增加行高

        # 美化表格
        for key, cell in table.get_celld().items():
            row, col = key
            cell.set_edgecolor('lightgray')
            if row == 0:  # 表头
                cell.set_text_props(weight='bold', ha='center')
                cell.set_facecolor('#B0C4DE')
            else:
                if "---" in table_data[row - 1][0]:
                    cell.set_text_props(weight='bold', ha='center')
                    cell.set_facecolor('#E0E0E0')
                if col == 0:  # 参数名列
                    cell.set_text_props(ha='left')
                if row % 2 == 0:  # 数据行交替颜色
                    cell.set_facecolor('#F5F5F5')

        # --- 4. 保存图像 ---
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(output_name, dpi=200, bbox_inches='tight')
        plt.close(fig)
        console.print(f"[bold green]✔ 分析图已成功保存到: {output_name}[/bold green]")


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


def main():
    """主执行函数"""
    console.print("[bold inverse] WarpX 独立模拟交互式分析器 [/bold inverse]")
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

    # --- 新增：交互式获取温度 ---
    console.print("\n" + "=" * 50)
    console.print("[bold yellow]      交互式温度输入环节[/bold yellow]")
    console.print("=" * 50)

    for run in loaded_runs:
        console.print(f"\n[bold]正在处理模拟: [cyan]{run.name}[/cyan][/bold]")

        if not run.final_spectrum or run.final_spectrum.weights.size == 0:
            console.print("[yellow]⚠ 最终能谱数据为空，无法计算平均能量，将跳过此模拟的理论谱绘制。[/yellow]")
            continue

        # 计算加权平均动能
        avg_energy_MeV = np.average(
            run.final_spectrum.energies_MeV,
            weights=run.final_spectrum.weights
        )

        console.print(f"  [green]➔ 计算出的最终加权平均动能为: [bold white]{avg_energy_MeV:.6f} MeV[/bold white][/green]")
        console.print("  [white]  (请使用此值在 Mathematica 等工具中计算对应的温度)[/white]")

        # 提示用户输入温度
        try:
            user_temp = Prompt.ask(
                f"  [bold spring_green2]请输入您为 [cyan]{run.name}[/cyan] 计算出的温度 (keV)[/bold spring_green2]",
                # default="0.0",
                console=console
            )
            run.user_T_keV = float(user_temp)
            console.print(f"  [green]✔ 已记录温度: {run.user_T_keV:.2f} keV[/green]")
        except (ValueError, TypeError):
            console.print("[yellow]⚠ 输入无效，将不绘制此模拟的理论谱。[/yellow]")
            run.user_T_keV = None

    generate_individual_plots(loaded_runs)
    console.print("\n[bold]所有分析完成。[/bold]")


if __name__ == "__main__":
    main()