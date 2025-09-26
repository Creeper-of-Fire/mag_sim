#!/usr/bin/env python3
#
# --- 单次模拟运行的简化分析脚本 ---
#
# 使用方法:
# 1. 将此脚本放置在单个模拟的输出目录中。
#    该目录必须包含 'sim_parameters.dpkl' 文件和 'diags/particle_states' 子目录。
# 2. 在该目录下运行此脚本: python3 post_analysis.py
# 3. 脚本会自动生成一张包含最终能谱、理论分布对比和参数摘要的图片，
#    保存在一个名为 'analysis_plots' 的新子目录中。
#

import os
import glob
import dill
import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy import constants
from scipy.special import kv

# =============================================================================
# 设置 Matplotlib 支持中文显示 (保持不变)
# =============================================================================
import matplotlib.font_manager as fm


def setup_chinese_font():
    """自动查找并设置支持中文的字体。"""
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


setup_chinese_font()


# =============================================================================


# =============================================================================
# 理论分布函数
# =============================================================================
def get_maxwell_boltzmann_distribution(E_bins_J, T_J):
    """
    计算非相对论性的麦克斯韦-玻尔兹曼能量分布的概率密度函数 (PDF)。
    返回 dN/(N_total * dE)。
    """
    if T_J <= 0: return np.zeros_like(E_bins_J)
    pdf = (2.0 / np.sqrt(np.pi)) * (1.0 / T_J) ** 1.5 * np.sqrt(E_bins_J) * np.exp(-E_bins_J / T_J)
    return pdf


def get_maxwell_juttner_distribution(E_bins_J, T_J):
    """
    计算相对论性的麦克斯韦-Jüttner能量分布的概率密度函数 (PDF)。
    返回 dN/(N_total * dE)。
    """
    m_e_c2 = constants.m_e * constants.c ** 2
    if T_J <= 0: return np.zeros_like(E_bins_J)

    theta = T_J / m_e_c2
    gamma = 1.0 + E_bins_J / m_e_c2
    pc = np.sqrt(E_bins_J * (E_bins_J + 2 * m_e_c2))

    # 使用 kv(2, ...) 保证兼容性
    normalization_factor = 1.0 / (m_e_c2 * theta * kv(2, 1.0 / theta))

    pdf = normalization_factor * (pc / m_e_c2) * gamma * np.exp(-gamma / theta)
    return pdf


# =============================================================================
# 核心绘图函数
# =============================================================================
def create_summary_plot(sim, source_folder='.'):
    """
    为单次模拟运行创建一张摘要图。
    该图包含最终能谱与理论分布的对比，以及模拟参数表。
    """
    print("  -> (1/2) 正在生成摘要图...")
    output_folder = os.path.join(source_folder, 'analysis_plots')
    os.makedirs(output_folder, exist_ok=True)
    run_prefix = os.path.basename(os.path.abspath(source_folder))

    plt.rcParams.update({"font.size": 12})
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 14), gridspec_kw={'height_ratios': [2, 3]})

    # --- Part 1: 能谱图 (模拟 vs 理论) ---
    fig.suptitle(f"最终状态摘要: {run_prefix}", fontsize=20, y=0.98)
    ax1.set_title('最终能谱 (e- + e+) 与理论分布对比')

    # --- 嵌套辅助函数，用于加载能谱 ---
    def _load_spectrum_data(folder, file_index=-1):
        particle_files = sorted(glob.glob(os.path.join(folder, "diags/particle_states", "openpmd_*.h5")))
        if not particle_files or len(particle_files) < abs(file_index):
            return None, None

        target_file = particle_files[file_index]
        all_energies_MeV, all_weights = [], []
        m_e_c2 = constants.m_e * constants.c ** 2

        try:
            with h5py.File(target_file, 'r') as f:
                step_str = os.path.basename(target_file).split('_')[-1].split('.')[0]
                current_time_step = int(step_str)
                for species in ["electrons", "positrons"]:
                    base_path = f"data/{current_time_step}/particles/{species}/"
                    if base_path + 'momentum/x' not in f: continue
                    px, py, pz = f[base_path + 'momentum/x'][:], f[base_path + 'momentum/y'][:], f[
                        base_path + 'momentum/z'][:]
                    weights = f[base_path + 'weighting'][:]
                    p_sq = px ** 2 + py ** 2 + pz ** 2
                    kinetic_energy_J = np.sqrt(p_sq * constants.c ** 2 + m_e_c2 ** 2) - m_e_c2
                    all_energies_MeV.append(kinetic_energy_J / (constants.e * 1e6))
                    all_weights.append(weights)
            if not all_energies_MeV: return None, None
            return np.concatenate(all_energies_MeV), np.concatenate(all_weights)
        except Exception as e:
            print(f"     -> 警告: 加载能谱数据 {target_file} 时出错: {e}")
            return None, None

    # 加载初始和最终能谱
    initial_energies, initial_weights = _load_spectrum_data(source_folder, file_index=0)
    final_energies, final_weights = _load_spectrum_data(source_folder, file_index=-1)

    # 收集所有能量数据以确定公共的bin范围
    all_energies_list = []
    if initial_energies is not None: all_energies_list.append(initial_energies)
    if final_energies is not None: all_energies_list.append(final_energies)

    if all_energies_list:
        combined_energies = np.concatenate(all_energies_list)
        positive_energies = combined_energies[combined_energies > 0]
        if positive_energies.size > 1:
            # 1. 使用线性分bin
            num_bins = 200
            min_E = 0  # 从0开始，或者用 positive_energies.min()
            max_E = positive_energies.max()
            common_bins_MeV = np.linspace(min_E, max_E, num_bins + 1)
            bin_centers_MeV = (common_bins_MeV[:-1] + common_bins_MeV[1:]) / 2.0
            bin_widths_MeV = common_bins_MeV[1:] - common_bins_MeV[:-1]

            # 绘制初始能谱
            if initial_energies is not None:
                initial_counts, _ = np.histogram(initial_energies, bins=common_bins_MeV, weights=initial_weights)
                dN_dE_initial = initial_counts / bin_widths_MeV
                ax1.plot(bin_centers_MeV, dN_dE_initial, ':', color='gray', label=f'模拟初始能谱')

            # 绘制最终能谱
            if final_energies is not None:
                final_counts, _ = np.histogram(final_energies, bins=common_bins_MeV, weights=final_weights)
                dN_dE_final = final_counts / bin_widths_MeV
                ax1.plot(bin_centers_MeV, dN_dE_final, '-', color='crimson', lw=2, label=f'模拟最终能谱')

                # 绘制理论分布
                N_total_final = np.sum(final_weights)
                if N_total_final > 0:
                    T_plasma_J = sim.T_plasma * constants.e
                    bin_centers_J = bin_centers_MeV * 1e6 * constants.e

                    # 定义单位转换系数 J/MeV
                    J_per_MeV = constants.e * 1e6

                    # Jüttner (相对论)
                    pdf_juttner_per_J = get_maxwell_juttner_distribution(bin_centers_J, T_plasma_J)
                    # ----> 单位转换 <----
                    pdf_juttner_per_MeV = pdf_juttner_per_J * J_per_MeV
                    dN_dE_juttner = N_total_final * pdf_juttner_per_MeV
                    ax1.plot(bin_centers_MeV, dN_dE_juttner, '--', color='black',
                             label=f'Maxwell-Jüttner (T={sim.T_plasma / 1e3:.1f} keV)')

                    # Boltzmann (经典)
                    pdf_boltzmann_per_J = get_maxwell_boltzmann_distribution(bin_centers_J, T_plasma_J)
                    # ----> 单位转换 <----
                    pdf_boltzmann_per_MeV = pdf_boltzmann_per_J * J_per_MeV
                    dN_dE_boltzmann = N_total_final * pdf_boltzmann_per_MeV
                    ax1.plot(bin_centers_MeV, dN_dE_boltzmann, '--', color='dodgerblue', alpha=0.7,
                             label=f'Maxwell-Boltzmann (经典)')

                    # ===== 诊断性打印 =====
                    print("\n--- 能谱数值诊断 ---")
                    print(f"总物理粒子数 (最终): {N_total_final:.2e}")
                    # 使用 nanmedian 忽略空的bin
                    print(f"模拟能谱 dN/dE 中位数: {np.nanmedian(dN_dE_final[dN_dE_final > 0]):.2e} [粒子数/MeV]")
                    print(
                        f"理论(Jüttner) dN/dE 中位数: {np.nanmedian(dN_dE_juttner[dN_dE_juttner > 0]):.2e} [粒子数/MeV]")
                    print("--------------------\n")
        else:
            print("\n错误：未能从 HDF5 文件加载任何能谱数据。")
            ax1.text(0.5, 0.5, '无法加载能谱数据',
                     horizontalalignment='center', verticalalignment='center',
                     transform=ax1.transAxes, fontsize=14, color='red')

    ax1.set_xlabel('动能 (MeV)')
    ax1.set_ylabel('粒子数密度 (dN/dE)')

    ax1.set_yscale('log')
    ax1.legend()
    ax1.grid(True, which="both", ls="--", alpha=0.5)
    ax1.set_ylim(bottom=1e5) # 设置一个合理的y轴下限，避免显示过多噪音

    # --- Part 2: 参数表格 ---
    ax2.axis('off')
    ax2.set_title('模拟参数', y=0.95)

    def prepare_table_data(sim_obj):
        params = [
            ("--- 物理参数 ---",),
            ("温度 T", f"{sim_obj.T_plasma * 1e-6:.3f} MeV"),
            ("数密度 n_plasma", f"{sim_obj.n_plasma:.2e} m^-3"),
            ("磁场 B0", f"{sim_obj.B0:.2f} T"),
            ("磁化强度 σ", f"{sim_obj.sigma:.3f}"),
            ("相对论热参数 θ", f"{sim_obj.theta:.3f}"),
            ("--- 数值参数 ---",),
            ("网格", f"{sim_obj.NX} x {sim_obj.NZ}"),
            ("模拟域 (d_e)", f"{sim_obj.LX:.1f} x {sim_obj.LZ:.1f}"),
            ("总步数", f"{sim_obj.total_steps}"),
            ("时间步长 (1/ω_pe)", f"{sim_obj.DT:.3f}"),
            ("每单元粒子数", f"{sim_obj.NPPC}"),
        ]
        row_labels, cell_text = [], []
        for item in params:
            if len(item) == 1:
                row_labels.append(item[0])
                cell_text.append([''])
            else:
                row_labels.append(f"  {item[0]}")
                cell_text.append([item[1]])
        return row_labels, cell_text

    row_labels, cell_text = prepare_table_data(sim)
    table = ax2.table(cellText=cell_text, rowLabels=row_labels,
                      colLabels=[f"值: {run_prefix}"],
                      loc='center', cellLoc='center')

    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2.0)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold', color='k')
            cell.set_facecolor('#B0C4DE')
        if col == -1:
            cell.set_text_props(ha='left')
            if "---" in cell.get_text().get_text():
                cell.set_text_props(weight='bold')
                cell.set_facecolor('#E0E0E0')

    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    output_path = os.path.join(output_folder, f"{run_prefix}_summary.png")
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  -> (2/2) 摘要图已保存到: {output_path}")


# =============================================================================
# 主执行函数
# =============================================================================
def main():
    """主函数：查找参数文件并启动分析。"""
    param_file = "sim_parameters.dpkl"

    print(f"正在当前目录中查找参数文件 '{param_file}'...")
    if not os.path.exists(param_file):
        print(f"\n错误：未找到 '{param_file}'。")
        print("请确保此脚本与模拟输出文件在同一个目录中。")
        return

    try:
        with open(param_file, "rb") as f:
            sim = dill.load(f)
        print(f"成功加载参数文件。")
    except Exception as e:
        print(f"\n错误：加载 '{param_file}' 时出错: {e}。")
        return

    # 调用核心绘图函数
    create_summary_plot(sim, source_folder='.')

    print("\n分析完成。")


if __name__ == "__main__":
    main()
