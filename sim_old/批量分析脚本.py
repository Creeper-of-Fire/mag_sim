#!/usr/bin/env python3
#
# --- 自动分析脚本，用于全动力学、相对论性对等离子体模拟 ---
#
# 使用方法:
# 1. 将此脚本放置在包含一个或多个模拟输出目录的文件夹中。
# 2. 模拟输出目录必须包含 'sim_parameters.dpkl' 文件在其父目录中，
#    以及 'fields' 和 'particle_states' 子目录。
# 3. 运行此脚本: python3 analysis_script.py
# 4. 脚本会询问是否选择一个模拟作为对比基准。
# 5. 所有结果将保存在一个名为 'analysis_output_pair_plasma' 的新目录中。
#

import glob
import os
import shutil

import dill
import h5py
from matplotlib.animation import FFMpegWriter, FuncAnimation
import matplotlib.pyplot as plt
from datetime import timedelta
import numpy as np
from matplotlib import colors
from scipy import constants

# =============================================================================
# 【WSL 最终解决方案】设置 Matplotlib 支持中文显示 (保持不变)
# =============================================================================
import matplotlib.font_manager as fm

# 定义备选的中文字体名称列表
chinese_fonts_priority = [
    'WenQuanYi Micro Hei', 'WenQuanYi Micro Hei Mono', 'Noto Sans CJK SC',
    'Source Han Sans SC', 'SimHei', 'Microsoft YaHei'
]
found_font = None
for font_name in chinese_fonts_priority:
    try:
        if fm.findfont(font_name, fontext='ttf'):
            found_font = font_name
            break
    except Exception:
        pass
if found_font:
    plt.rcParams['font.sans-serif'] = [found_font]
    print(f"Matplotlib 字体已设置为：{found_font}")
else:
    print("\n警告：未能找到任何支持中文的字体。图表中的中文可能无法正常显示。")
plt.rcParams['axes.unicode_minus'] = False


# =============================================================================


# =============================================================================
# 辅助数据加载函数，适配对等离子体模拟
# =============================================================================
def _load_reconnection_data(source_folder, sim):
    """从 plane.dat 加载重联率数据，使用新的归一化。"""
    plane_data_path = os.path.join(source_folder, "plane.dat")
    try:
        plane_data = np.loadtxt(plane_data_path, skiprows=1)
        x_idx, z_idx, Ey_idx, Bx_idx = 2, 4, 6, 8
        steps = np.unique(plane_data[:, 0])
        if len(steps) == 0: return None, None
        num_cells = plane_data.shape[0] // len(steps)
        plane_data = plane_data.reshape((len(steps), num_cells, plane_data.shape[1]))
        # 时间归一化: t -> t * w_pe
        times = plane_data[:, 0, 1] * sim.w_pe
        # 重联率归一化: E_y -> E_y / (c * B0)
        rates = np.mean(plane_data[:, :, Ey_idx], axis=1) / (constants.c * sim.B0)
        return times, rates
    except (FileNotFoundError, IndexError, ValueError):
        return None, None


def _load_energy_evolution_data(source_folder, sim):
    """加载总等离子体能量(电子+正电子)演化数据，使用相对论公式。"""
    particle_files = sorted(glob.glob(os.path.join(source_folder, "particle_states", "openpmd_*.h5")))
    if not particle_files: return [], []

    times, total_energies = [], []
    m_e_c2 = constants.m_e * constants.c ** 2
    m_e2_c4 = m_e_c2 ** 2

    for filename in particle_files:
        try:
            step_str = filename.split('_')[-1].split('.')[0]
            current_time_step = int(step_str)
            total_energy_frame = 0.0

            with h5py.File(filename, 'r') as f:
                # 遍历电子和正电子
                for species in ["electrons", "positrons"]:
                    base_path = f"data/{current_time_step}/particles/{species}/"
                    if base_path + 'momentum/x' not in f: continue

                    px, py, pz = f[base_path + 'momentum/x'][:], f[base_path + 'momentum/y'][:], f[
                        base_path + 'momentum/z'][:]
                    weights = f[base_path + 'weighting'][:]
                    p_sq = px ** 2 + py ** 2 + pz ** 2

                    # 使用相对论动能公式: E_k = sqrt(p^2*c^2 + m^2*c^4) - m*c^2
                    kinetic_energy_J = np.sqrt(p_sq * constants.c ** 2 + m_e2_c4) - m_e_c2
                    total_energy_frame += np.sum(kinetic_energy_J * weights)

            # 时间归一化: t -> t * w_pe
            times.append(current_time_step * sim.dt * sim.w_pe)
            total_energies.append(total_energy_frame)
        except Exception:
            continue
    return times, total_energies


def _get_final_particle_count(source_folder):
    """加载并计算最终时刻的总物理粒子数 (电子+正电子)。"""
    particle_files = sorted(glob.glob(os.path.join(source_folder, "particle_states", "openpmd_*.h5")))
    if not particle_files: return None

    last_file = particle_files[-1]
    total_physical_particles = 0.0
    try:
        step_str = last_file.split('_')[-1].split('.')[0]
        current_time_step = int(step_str)
        with h5py.File(last_file, 'r') as f:
            # 遍历电子和正电子
            for species in ["electrons", "positrons"]:
                base_path = f"data/{current_time_step}/particles/{species}/"
                weighting_path = base_path + 'weighting'
                if weighting_path in f:
                    total_physical_particles += np.sum(f[weighting_path][:])
        return total_physical_particles
    except Exception as e:
        print(f"处理文件 {last_file} 时出错: {e}")
        return None


# =============================================================================
# 1. 重联率绘图函数
# =============================================================================
def plot_reconnection_rate(source_folder, output_folder, run_prefix, sim, ref_folder=None, ref_sim=None,
                           ref_prefix=None):
    print("  (1/4) 正在绘制重联率...")
    plt.rcParams.update({"font.size": 20})
    fig, ax = plt.subplots(figsize=(10, 6))

    if ref_folder and ref_sim and ref_prefix:
        ref_times, ref_rates = _load_reconnection_data(ref_folder, ref_sim)
        if ref_times is not None:
            ax.plot(ref_times, ref_rates, "o--", color='gray', label=f'基准: {ref_prefix}')

    current_times, current_rates = _load_reconnection_data(source_folder, sim)
    if current_times is not None:
        ax.plot(current_times, current_rates, "o-", label=f'当前: {run_prefix}')
    else:
        plt.close(fig)
        return

    ax.grid(True)
    # 更新坐标轴标签
    ax.set_xlabel(r"$t \omega_{pe}$")
    ax.set_ylabel("$<E_y>/cB_0$")
    title = f"Reconnection Rate [{run_prefix}]"
    if ref_prefix: title += f"\nvs [{ref_prefix}]"
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    output_path = os.path.join(output_folder, f"{run_prefix}_reconnection_rate_comp.png")
    plt.savefig(output_path)
    plt.close(fig)
    print(f"        -> 已保存到: {output_path}")


# =============================================================================
def _get_run_duration(source_folder):
    fields_dir = os.path.join(source_folder, "fields")
    if not os.path.isdir(fields_dir): return None
    try:
        field_files = sorted(glob.glob(os.path.join(fields_dir, "*.npz")))
        if len(field_files) < 2: return None
        start_time = os.path.getmtime(field_files[0])
        end_time = os.path.getmtime(field_files[-1])
        return timedelta(seconds=end_time - start_time)
    except Exception:
        return None


def _write_spectrum_bins_to_csv(output_path, current_data, ref_data, current_prefix, ref_prefix):
    current_bins, current_spec = current_data
    ref_bins, ref_spec = ref_data
    total_current, total_ref = np.sum(current_spec), np.sum(ref_spec)
    try:
        with open(output_path, 'w', newline='') as f:
            f.write(f"# 最终时刻能谱分bin数据对比\n# 当前运行: {current_prefix}\n# 基准运行: {ref_prefix}\n")
            f.write(f"# 总粒子数 (当前): {total_current:.4e}\n# 总粒子数 (基准): {total_ref:.4e}\n\n")
            f.write(f"Energy_Bin_Center(MeV),Counts_Current({current_prefix}),Counts_Reference({ref_prefix})\n")
            num_rows = min(100, len(current_bins), len(ref_bins))
            for i in range(num_rows):
                f.write(f"{current_bins[i]:.6f},{current_spec[i]:.6e},{ref_spec[i]:.6e}\n")
        print(f"        -> 能谱分bin数据已保存到: {output_path}")
    except Exception as e:
        print(f"        -> 警告：写入CSV文件 {output_path} 时出错: {e}")


# =============================================================================

# =============================================================================
# 2. 生成最终状态摘要图
# =============================================================================
def create_summary_plot(source_folder, output_folder, run_prefix, sim, ref_folder=None, ref_sim=None, ref_prefix=None):
    """创建一张包含最终能谱和对等离子体模拟参数的摘要图。"""

    # --- 嵌套辅助函数，用于加载指定时刻的能谱 ---
    def _load_spectrum_data(folder, sim_obj, file_index=-1):
        """
        加载指定时刻（由 file_index 决定，0 为初始，-1 为最终）的
        电子和正电子的能量(MeV)和权重，使用相对论公式。
        返回合并后的 (energies_MeV, weights) 元组。
        """
        particle_files = sorted(glob.glob(os.path.join(folder, "particle_states", "openpmd_*.h5")))
        if not particle_files or len(particle_files) < abs(file_index):
            return None, None

        target_file = particle_files[file_index]
        all_energies_MeV = []
        all_weights = []
        m_e_c2 = constants.m_e * constants.c ** 2
        m_e2_c4 = m_e_c2 ** 2

        try:
            step_str = target_file.split('_')[-1].split('.')[0]
            current_time_step = int(step_str)
            with h5py.File(target_file, 'r') as f:
                for species in ["electrons", "positrons"]:
                    base_path = f"data/{current_time_step}/particles/{species}/"
                    if base_path + 'momentum/x' not in f: continue

                    px, py, pz = f[base_path + 'momentum/x'][:], f[base_path + 'momentum/y'][:], f[
                        base_path + 'momentum/z'][:]
                    weights = f[base_path + 'weighting'][:]
                    p_sq = px ** 2 + py ** 2 + pz ** 2

                    kinetic_energy_J = np.sqrt(p_sq * constants.c ** 2 + m_e2_c4) - m_e_c2
                    kinetic_energy_MeV = kinetic_energy_J / (constants.e * 1e6)

                    all_energies_MeV.append(kinetic_energy_MeV)
                    all_weights.append(weights)

            if not all_energies_MeV: return None, None
            return np.concatenate(all_energies_MeV), np.concatenate(all_weights)
        except Exception as e:
            print(f"        -> 警告: 加载能谱数据 {target_file} 时出错: {e}")
            return None, None

    def get_val(sim_obj, attr_expr):
        """更简洁、更强大的版本，直接使用 eval() 处理表达式。
           在这种受控的科学计算环境中，这是完全合适的。"""
        try:
            # 创建一个局部命名空间，只包含 sim_obj，并将其命名为 'sim'
            # 这样表达式就可以写成 'sim.Lz / sim.NZ' 或 '1.0 / sim.w_pe'
            local_namespace = {'sim': sim_obj}

            # 为了让 f-string 也能工作，我们稍微调整一下
            if attr_expr.startswith("f'"):
                # 对于 f-string，我们用一个稍微不同的上下文
                fstring_namespace = {"current_sim": sim_obj, "ref_sim": sim_obj}
                return eval(attr_expr, {}, fstring_namespace)

            # 对于所有其他表达式，直接求值
            return eval(attr_expr, {}, local_namespace)

        except Exception:
            # 捕获任何可能的计算错误（如 AttributeError, ZeroDivisionError 等）
            return "N/A"

    def prepare_table_data(current_sim, ref_sim=None, **kwargs):
        """为 Matplotlib 表格准备对等离子体的数据和颜色。"""
        # 全新的参数列表
        params_to_check = [
            ("--- 输入参数 ---",),
            ("T", "sim.T_plasma * 1e-6", ".2f", "MeV"),
            ("n0", "sim.n_plasma", ".1e", "m^-3"),
            ("B0", "sim.B0", ".2f", "T"),
            ("--- 等离子体参数 ---",),
            ("d_e", "sim.d_e", ".1e", "m"),
            ("1/w_pe", "1.0 / sim.w_pe", ".1e", "s"),
            ("Magnetization σ", "sim.sigma", ".2f", ""),
            ("Theta θ", "sim.theta", ".2f", ""),
            ("--- 数值参数 ---",),
            ("Grid", "f'{current_sim.NX}x{current_sim.NZ}'", "", ""),  # 支持f-string
            ("dz/d_e", "sim.Lz / sim.NZ", ".2f", ""),  # Lz已经是归一化的
            ("dt*w_pe", "sim.DT", ".3f", ""),  # DT是归一化的
            ("NPPC/species", "sim.NPPC", "d", ""),
            ("--- 运行信息 ---",),
            ("总粒子数", "particle_count", ".2e", ""),
            ("运行时间", "runtime", "", ""),
        ]
        row_labels, cell_text, row_colors = [], [], []
        COLOR_HEADER, COLOR_DIFF, COLOR_NORMAL = '#E0E0E0', '#FFDDDD', 'w'

        for item in params_to_check:
            if len(item) == 1:
                row_labels.append(item[0])
                cell_text.append([''] * (2 if ref_sim else 1))
                row_colors.append(COLOR_HEADER)
                continue
            label, attr, fmt, unit = item

            # 简化特殊项处理
            if attr in kwargs:
                row_labels.append(f"  {label}")
                val_current = kwargs[attr]
                str_current = str(val_current).split('.')[0] if isinstance(val_current, timedelta) else (
                    f"{val_current:{fmt}}" if val_current else "N/A")

                if ref_sim:
                    val_ref = kwargs.get(f"ref_{attr}")
                    str_ref = str(val_ref).split('.')[0] if isinstance(val_ref, timedelta) else (
                        f"{val_ref:{fmt}}" if val_ref else "N/A")
                    cell_text.append([str_current, str_ref])
                    row_colors.append(COLOR_DIFF if val_current != val_ref else COLOR_NORMAL)
                else:
                    cell_text.append([str_current])
                    row_colors.append(COLOR_NORMAL)
                continue

            row_labels.append(f"  {label}")
            if attr.startswith("f'"):
                val_current = eval(attr)  # Eval for f-string
            else:
                val_current = get_val(current_sim, attr)
            str_current = f"{val_current:{fmt}} {unit}".strip() if isinstance(val_current, (int, float)) else str(
                val_current)

            if ref_sim:
                if attr.startswith("f'"):
                    val_ref = eval(attr.replace('current_sim', 'ref_sim'))
                else:
                    val_ref = get_val(ref_sim, attr)
                str_ref = f"{val_ref:{fmt}} {unit}".strip() if isinstance(val_ref, (int, float)) else str(val_ref)
                cell_text.append([str_current, str_ref])
                is_diff = not np.isclose(val_current, val_ref) if isinstance(val_current, (int, float)) and isinstance(
                    val_ref, (int, float)) else (val_current != val_ref)
                row_colors.append(COLOR_DIFF if is_diff else COLOR_NORMAL)
            else:
                cell_text.append([str_current])
                row_colors.append(COLOR_NORMAL)
        return row_labels, cell_text, row_colors

    # --- 主函数体 ---
    print("  (2/4) 正在生成最终摘要图...")
    plt.rcParams.update({"font.size": 12})
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 16), gridspec_kw={'height_ratios': [2, 3]})

    # --- Part 1: 能谱图 ---
    fig.suptitle(f"最终状态摘要 [{run_prefix}]" + (f" vs [{ref_prefix}]" if ref_prefix else ""), fontsize=20, y=0.98)
    ax1.set_title('能谱演化 (e- + e+)')

    # 加载当前模拟的初始和最终能谱
    current_energies_final, current_weights_final = _load_spectrum_data(source_folder, sim, file_index=-1)
    current_energies_initial, current_weights_initial = _load_spectrum_data(source_folder, sim, file_index=0)

    is_comparison = ref_folder and ref_sim and ref_prefix

    # 初始化能谱数据容器
    bin_centers, current_spec_final, current_spec_initial, ref_spec_final = None, None, None, None
    num_bins = 100
    all_energies_list = []

    # 收集所有能谱数据以确定公共的bin范围
    if current_energies_final is not None: all_energies_list.append(current_energies_final)
    if current_energies_initial is not None: all_energies_list.append(current_energies_initial)
    if is_comparison:
        ref_energies_final, ref_weights_final = _load_spectrum_data(ref_folder, ref_sim, file_index=-1)
        if ref_energies_final is not None: all_energies_list.append(ref_energies_final)

    # 如果有任何能谱数据，则进行分bin
    if all_energies_list:
        combined_energies = np.concatenate(all_energies_list)
        positive_energies = combined_energies[combined_energies > 0]
        if positive_energies.size > 1:
            # 创建覆盖所有数据的对数坐标bin
            # ================================================================
            # 采用单纯的线性分bin策略
            # ================================================================
            min_E = positive_energies.min()
            max_E = positive_energies.max()

            total_bins = 200  # 定义总的bin数量

            # 创建线性间隔的bin
            common_bins = np.linspace(min_E, max_E, total_bins + 1)
            print(f"        -> 使用线性分bin策略，共 {total_bins} 个bin。")

            bin_centers = (common_bins[:-1] + common_bins[1:]) / 2.0
            # ================================================================

            # 对每个能谱进行分bin
            if current_energies_final is not None:
                current_spec_final, _ = np.histogram(current_energies_final, bins=common_bins,
                                                     weights=current_weights_final)
            if current_energies_initial is not None:
                current_spec_initial, _ = np.histogram(current_energies_initial, bins=common_bins,
                                                       weights=current_weights_initial)
            if is_comparison and ref_energies_final is not None:
                ref_spec_final, _ = np.histogram(ref_energies_final, bins=common_bins, weights=ref_weights_final)

    # 写入最终能谱对比的CSV文件 (功能保持不变)
    if is_comparison and bin_centers is not None:
        csv_path = os.path.join(output_folder, f"{run_prefix}_vs_{ref_prefix}_spectrum_bins.csv")
        _write_spectrum_bins_to_csv(csv_path,
                                    (bin_centers,
                                     current_spec_final if current_spec_final is not None else np.zeros_like(
                                         bin_centers)),
                                    (bin_centers,
                                     ref_spec_final if ref_spec_final is not None else np.zeros_like(bin_centers)),
                                    run_prefix, ref_prefix)

    # --- 开始绘图 ---
    if current_spec_initial is not None and bin_centers is not None:
        ax1.plot(bin_centers, current_spec_initial, ':', color='k', alpha=0.7, label=f'初始: {run_prefix}')
    if is_comparison and ref_spec_final is not None and bin_centers is not None:
        ax1.plot(bin_centers, ref_spec_final, '--', color='gray', label=f'最终 (基准): {ref_prefix}')
    if current_spec_final is not None and bin_centers is not None:
        ax1.plot(bin_centers, current_spec_final, '-', label=f'最终 (当前): {run_prefix}')

    ax1.set_xlabel('能量 (MeV)')
    ax1.set_ylabel('计数 (dN/dE)')
    # ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.legend()
    ax1.grid(True, which="both", ls="--", alpha=0.5)

    # --- Part 2: 参数对比表格 ---
    ax2.axis('off')

    # 组织 kwargs for prepare_table_data
    kwargs = {
        'runtime': _get_run_duration(source_folder),
        'particle_count': _get_final_particle_count(source_folder)
    }
    col_labels = [f"当前: {run_prefix[:25]}"]
    if ref_sim:
        kwargs['ref_runtime'] = _get_run_duration(ref_folder)
        kwargs['ref_particle_count'] = _get_final_particle_count(ref_folder)
        col_labels.append(f"基准: {ref_prefix[:25]}")

    row_labels, cell_text, row_colors = prepare_table_data(sim, ref_sim, **kwargs)

    # ... (表格创建和美化逻辑保持不变) ...
    table = ax2.table(cellText=cell_text, rowLabels=row_labels, rowColours=row_colors, colLabels=col_labels,
                      colWidths=[0.3] * len(col_labels), loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.8)
    for (row, col), cell in table.get_celld().items():
        if row == 0: cell.set_fontsize(12); cell.set_text_props(weight='bold', color='k'); cell.set_facecolor('#B0C4DE')
        if col == -1:
            cell.set_text_props(ha='left')
            if "---" in cell.get_text().get_text(): cell.set_text_props(weight='bold'); cell.set_facecolor('#E0E0E0')

    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    output_path = os.path.join(output_folder,
                               f"{run_prefix}_vs_{ref_prefix}_summary.png" if ref_prefix else f"{run_prefix}_summary.png")
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"        -> 摘要图已保存到: {output_path}")


# =============================================================================
# 3. 绘制总粒子能量演化图
# =============================================================================
def plot_total_energy_evolution(source_folder, output_folder, run_prefix, sim, ref_folder=None, ref_sim=None,
                                ref_prefix=None):
    print("  (3/4) 正在计算总粒子能量演化...")
    plt.rcParams.update({"font.size": 16})
    fig, ax = plt.subplots(figsize=(10, 6))

    if ref_folder and ref_sim and ref_prefix:
        ref_times, ref_energies = _load_energy_evolution_data(ref_folder, ref_sim)
        if ref_times: ax.plot(ref_times, ref_energies, '--', color='gray', label=f'基准: {ref_prefix}')

    current_times, current_energies = _load_energy_evolution_data(source_folder, sim)
    if not current_times: plt.close(fig); return
    ax.plot(current_times, current_energies, 'o-', label=f'当前: {run_prefix}')

    # 更新标签和标题
    ax.set_xlabel(r"Time, $t \omega_{pe}$")
    ax.set_ylabel("Total Kinetic Energy (e- + e+) (J)")
    title = f"Total Plasma Energy Evolution [{run_prefix}]"
    if ref_prefix: title += f"\nvs [{ref_prefix}]"
    ax.set_title(title)
    ax.grid(True)
    ax.legend()
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    plt.tight_layout()
    output_path = os.path.join(output_folder, f"{run_prefix}_total_energy_evolution_comp.png")
    plt.savefig(output_path)
    plt.close(fig)
    print(f"        -> 总能量演化图已保存到: {output_path}")


# =============================================================================
# 4. 绘制最终时刻粒子分布图
# =============================================================================
def plot_final_particle_distribution(source_folder, output_folder, run_prefix, sim, ref_folder=None, ref_sim=None,
                                     ref_prefix=None):
    print("  (4/4) 正在绘制最终粒子分布切片...")
    plt.rcParams.update({"font.size": 16})

    def _load_and_bin_particles(folder, sim_obj):
        particle_files = sorted(glob.glob(os.path.join(folder, "particle_states", "openpmd_*.h5")))
        if not particle_files: return None, None
        last_file = particle_files[-1]
        try:
            step_str = last_file.split('_')[-1].split('.')[0]
            current_time_step = int(step_str)
            with h5py.File(last_file, 'r') as f:
                # 只加载电子数据作为代表
                base_path = f"data/{current_time_step}/particles/electrons/"
                if base_path + 'position/x' not in f: return None, None
                # 坐标归一化: x -> x / d_e
                pos_x = f[base_path + 'position/x'][:] / sim_obj.d_e
                pos_z = f[base_path + 'position/z'][:] / sim_obj.d_e
                weights = f[base_path + 'weighting'][:]

            if pos_x.size == 0: return None, None

            # 使用归一化的模拟域尺寸定义bins
            bins_x = np.linspace(-sim_obj.LX / 2.0, sim_obj.LX / 2.0, 257)
            bins_z = np.linspace(-sim_obj.LZ / 2.0, sim_obj.LZ / 2.0, 129)
            density, _, _ = np.histogram2d(pos_x, pos_z, bins=[bins_x, bins_z], weights=weights)
            extent = [bins_x[0], bins_x[-1], bins_z[0], bins_z[-1]]
            return density.T, extent
        except Exception:
            return None, None

    has_ref = ref_folder and ref_sim and ref_prefix
    fig, axes = plt.subplots(1, 2 if has_ref else 1, figsize=(16 if has_ref else 8, 6), sharey=True)
    if not has_ref: axes = [axes]

    density_current, extent_current = _load_and_bin_particles(source_folder, sim)
    if density_current is not None:
        im = axes[0].imshow(density_current, origin='lower', extent=extent_current, aspect='equal',
                            norm=colors.LogNorm(), cmap='jet')
        axes[0].set_title(f"Current: {run_prefix}")
        # 更新标签
        axes[0].set_xlabel("$x/d_e$")
        axes[0].set_ylabel("$z/d_e$")
        fig.colorbar(im, ax=axes[0], label="Electron Density (Arb. Units)")
    else:
        axes[0].text(0.5, 0.5, "No particle data", ha='center', va='center')

    if has_ref:
        density_ref, extent_ref = _load_and_bin_particles(ref_folder, ref_sim)
        if density_ref is not None:
            im = axes[1].imshow(density_ref, origin='lower', extent=extent_ref, aspect='equal', norm=colors.LogNorm(),
                                cmap='jet')
            axes[1].set_title(f"Reference: {ref_prefix}")
            axes[1].set_xlabel("$x/d_e$")
            fig.colorbar(im, ax=axes[1], label="Electron Density (Arb. Units)")
        else:
            axes[1].text(0.5, 0.5, "No particle data", ha='center', va='center')

    # 更新标题
    fig.suptitle("Final Electron Density Distribution", fontsize=20)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    output_path = os.path.join(output_folder, f"{run_prefix}_final_particle_distribution_comp.png")
    plt.savefig(output_path)
    plt.close(fig)
    print(f"        -> 最终粒子分布图已保存到: {output_path}")


# =============================================================================
# 主执行函数
# =============================================================================
def find_sim_params(start_folder):
    current_path = os.path.normpath(start_folder)
    for _ in range(5):
        potential_path = os.path.join(current_path, "sim_parameters.dpkl")
        if os.path.exists(potential_path): return potential_path
        parent_path = os.path.dirname(current_path)
        if parent_path == current_path: break
        current_path = parent_path
    return None


def main():
    """主函数：扫描、查找、选择基准并处理所有有效的模拟目录。"""
    # 更新输出目录名
    OUTPUT_FOLDER = "analysis_output_pair_plasma"
    if not os.path.exists(OUTPUT_FOLDER): os.makedirs(OUTPUT_FOLDER)
    print(f"所有分析结果将保存在 '{OUTPUT_FOLDER}' 目录中。\n")

    print("正在扫描有效的模拟数据目录...")
    analysis_folders = []
    for dirpath, dirnames, _ in os.walk('..'):
        dirnames[:] = [d for d in dirnames if not d.startswith('.') and d != OUTPUT_FOLDER]
        if 'fields' in dirnames and 'particle_states' in dirnames:
            if os.path.isdir(os.path.join(dirpath, 'fields')) and os.path.isdir(
                    os.path.join(dirpath, 'particle_states')):
                analysis_folders.append(os.path.normpath(dirpath))

    if not analysis_folders:
        print("\n错误：在当前目录或其子目录中，未找到任何有效的模拟数据目录。")
        return
    print(f"找到了 {len(analysis_folders)} 个有效的模拟数据目录。")

    ref_folder, ref_sim, ref_prefix = None, None, None
    if len(analysis_folders) > 1:
        print("\n请选择一个“原始模拟”作为对比基准：")
        for i, folder in enumerate(analysis_folders): print(f"  [{i + 1}] {folder}")
        while True:
            choice = input("请输入基准模拟的编号 (1, 2, ...)，或直接按 Enter 跳过对比模式: ")
            if not choice: print("已跳过对比模式，将独立分析每个目录。\n"); break
            try:
                choice_idx = int(choice) - 1
                if 0 <= choice_idx < len(analysis_folders):
                    ref_folder = analysis_folders[choice_idx]
                    param_path = find_sim_params(ref_folder)
                    if not param_path:
                        print(f"\n错误：在基准目录 '{ref_folder}' ...中找不到 'sim_parameters.dpkl'。取消对比...\n")
                        ref_folder = None
                        break
                    ref_prefix = ref_folder.replace(os.sep, '_').strip('./_')
                    with open(param_path, "rb") as f:
                        ref_sim = dill.load(f)
                    print(f"已选择 '{ref_folder}' 作为对比基准。\n")
                    break
                else:
                    print("无效的编号，请重试。")
            except (ValueError, Exception) as e:
                print(f"加载基准参数时出错: {e}。取消对比...\n")
                ref_folder = None
                break

    for folder in analysis_folders:
        if ref_folder and os.path.samefile(folder, ref_folder):
            print(f"--- 跳过基准目录本身: '{folder}' ---\n")
            continue
        print(f"--- 开始处理目录: '{folder}' ---")
        run_prefix = folder.replace(os.sep, '_').strip('./_')
        sim_param_path = find_sim_params(folder)
        if not sim_param_path:
            print(f"  错误：找不到 'sim_parameters.dpkl'。跳过。")
            continue
        try:
            with open(sim_param_path, "rb") as f:
                sim = dill.load(f)
            print(f"  成功加载参数文件: '{sim_param_path}'")
        except Exception as e:
            print(f"  错误：加载 '{sim_param_path}' 时出错: {e}。跳过。")
            continue

        plot_reconnection_rate(folder, OUTPUT_FOLDER, run_prefix, sim, ref_folder, ref_sim, ref_prefix)
        create_summary_plot(folder, OUTPUT_FOLDER, run_prefix, sim, ref_folder, ref_sim, ref_prefix)
        plot_total_energy_evolution(folder, OUTPUT_FOLDER, run_prefix, sim, ref_folder, ref_sim, ref_prefix)
        plot_final_particle_distribution(folder, OUTPUT_FOLDER, run_prefix, sim, ref_folder, ref_sim, ref_prefix)
        print(f"--- 完成处理目录: '{folder}' ---\n")
    print("所有分析任务已完成。")


if __name__ == "__main__":
    main()