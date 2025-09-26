#!/usr/bin/env python3
#
# --- 单次模拟运行的简化分析脚本 (V2 - 修正版) ---
#
# V2 版本改动:
# 1. [修正] 自动从HDF5文件读取所有粒子物种名称，解决了硬编码导致的数据加载失败问题。
# 2. [修正] 调整了理论曲线的绘制逻辑，不再错误地将混合初始态与单一热分布对比。
# 3. [改进] 默认使用对数能量分箱(logarithmic binning)，能更清晰地展示高能粒子谱。
# 4. [改进] 动态设置Y轴下限，避免硬编码。
# 5. [改进] 增加了对复合初始分布的注释和说明。
#

import os
import glob
import dill
import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy import constants
from scipy.special import kv
import matplotlib.font_manager as fm


# ... (中文字体设置 和 理论分布函数 保持不变) ...
def setup_chinese_font():
    """自动查找并设置支持中文的字体。"""
    chinese_fonts_priority = ['WenQuanYi Micro Hei', 'WenQuanYi Micro Hei Mono', 'Noto Sans CJK SC',
                              'Source Han Sans SC', 'SimHei', 'Microsoft YaHei']
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


def get_maxwell_boltzmann_distribution(E_bins_J, T_J):
    if T_J <= 0: return np.zeros_like(E_bins_J)
    return (2.0 / np.sqrt(np.pi)) * (1.0 / T_J) ** 1.5 * np.sqrt(E_bins_J) * np.exp(-E_bins_J / T_J)


def get_maxwell_juttner_distribution(E_bins_J, T_J):
    m_e_c2 = constants.m_e * constants.c ** 2
    if T_J <= 0: return np.zeros_like(E_bins_J)
    theta = T_J / m_e_c2
    gamma = 1.0 + E_bins_J / m_e_c2
    pc = np.sqrt(E_bins_J * (E_bins_J + 2 * m_e_c2))
    normalization_factor = 1.0 / (m_e_c2 * theta * kv(2, 1.0 / theta))
    return normalization_factor * (pc / m_e_c2) * gamma * np.exp(-gamma / theta)


# =============================================================================
# 核心绘图函数 (V2)
# =============================================================================
def create_summary_plot(sim, source_folder='.'):
    """
    为单次模拟运行创建一张摘要图。(V2版)
    """
    print("  -> (1/2) 正在生成摘要图 (V2)...")
    output_folder = os.path.join(source_folder, 'analysis_plots')
    os.makedirs(output_folder, exist_ok=True)
    run_prefix = os.path.basename(os.path.abspath(source_folder))

    plt.rcParams.update({"font.size": 12})
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 14), gridspec_kw={'height_ratios': [2, 3]})

    fig.suptitle(f"最终状态摘要: {run_prefix}", fontsize=20, y=0.98)
    ax1.set_title('最终能谱 (所有带电粒子) 与理论分布对比')

    # --- [V2] 嵌套辅助函数，用于加载能谱 (已修正) ---
    def _load_spectrum_data(folder, file_index=-1):
        particle_files = sorted(glob.glob(os.path.join(folder, "diags/particle_states", "openpmd_*.h5")))
        if not particle_files or len(particle_files) <= abs(file_index):
            print(f"     -> 警告: 找不到足够的HDF5文件 (需要至少 {abs(file_index) + 1} 个)。")
            return None, None

        target_file = particle_files[file_index]
        all_energies_MeV, all_weights = [], []
        m_e_c2 = constants.m_e * constants.c ** 2

        try:
            with h5py.File(target_file, 'r') as f:
                step_str = list(f['data'].keys())[0]
                current_time_step = int(step_str)
                # <<< MODIFIED: 自动发现HDF5文件中的所有粒子物种
                species_in_file = list(f[f'data/{current_time_step}/particles'].keys())
                print(f"     -> 在文件 {os.path.basename(target_file)} 中发现物种: {species_in_file}")

                # <<< MODIFIED: 过滤掉光子等中性粒子
                charged_species = [s for s in species_in_file if 'photon' not in s]

                for species in charged_species:
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
            print(f"     -> 警告: 加载能谱数据 {os.path.basename(target_file)} 时出错: {e}")
            return None, None

    initial_energies, initial_weights = _load_spectrum_data(source_folder, file_index=0)
    final_energies, final_weights = _load_spectrum_data(source_folder, file_index=-1)
    # final_energies, final_weights = _load_spectrum_data(source_folder, file_index=1)

    all_energies_list = [e for e in [initial_energies, final_energies] if e is not None]

    if all_energies_list:
        combined_energies = np.concatenate(all_energies_list)
        positive_energies = combined_energies[combined_energies > 0]
        if positive_energies.size > 1:
            # <<< MODIFIED: 使用对数分箱 (Logarithmic Binning)
            num_bins = 150
            # 找到一个合理的能量范围，避免log(0)
            min_E = max(positive_energies.min() * 0.5, 1e-4)
            max_E = positive_energies.max() * 1.1
            common_bins_MeV = np.logspace(np.log10(min_E), np.log10(max_E), num_bins + 1)

            bin_centers_MeV = np.sqrt(common_bins_MeV[:-1] * common_bins_MeV[1:])
            bin_widths_MeV = common_bins_MeV[1:] - common_bins_MeV[:-1]

            if initial_energies is not None:
                initial_counts, _ = np.histogram(initial_energies, bins=common_bins_MeV, weights=initial_weights)
                dN_dE_initial = initial_counts / bin_widths_MeV
                ax1.plot(bin_centers_MeV, dN_dE_initial, ':', color='gray', label='模拟初始能谱 (热背景+束流)')

            if final_energies is not None:
                final_counts, _ = np.histogram(final_energies, bins=common_bins_MeV, weights=final_weights)
                dN_dE_final = final_counts / bin_widths_MeV
                ax1.plot(bin_centers_MeV, dN_dE_final, '-', color='crimson', lw=2, label='模拟最终能谱')

                N_total_final = np.sum(final_weights)
                if N_total_final > 0:
                    # T_plasma = sim.T_plasma
                    T_plasma = 1e3 * 150
                    T_plasma_J = T_plasma * constants.e
                    bin_centers_J = bin_centers_MeV * 1e6 * constants.e
                    J_per_MeV = constants.e * 1e6

                    # <<< MODIFIED: 仅为最终能谱绘制参考理论曲线
                    # 这是初始温度下的热平衡分布，作为一个参照基准
                    pdf_juttner_per_J = get_maxwell_juttner_distribution(bin_centers_J, T_plasma_J)
                    pdf_juttner_per_MeV = pdf_juttner_per_J * J_per_MeV

                    # <<< MODIFIED: 注意，这里的N_total需要乘以热组分的比例(1-beam_fraction)
                    # 因为理论分布只描述热背景部分
                    thermal_fraction = (1.0 - sim.beam_fraction)
                    N_total_thermal_approx = N_total_final # * thermal_fraction

                    dN_dE_juttner = N_total_thermal_approx * pdf_juttner_per_MeV
                    ax1.plot(bin_centers_MeV, dN_dE_juttner, '--', color='black',
                             label=f'初始热背景理论谱 (T={T_plasma / 1e3:.1f} keV, 参考)')
        else:
            ax1.text(0.5, 0.5, '无法加载能谱数据', ha='center', va='center', transform=ax1.transAxes, color='red')

    ax1.set_xlabel('动能 (MeV)')
    ax1.set_ylabel('粒子数谱密度 (dN/dE [MeV⁻¹])')
    ax1.set_xscale('log')  # <<< NEW: X轴也用对数坐标
    ax1.set_yscale('log')
    ax1.legend()
    ax1.grid(True, which="both", ls="--", alpha=0.5)

    # <<< MODIFIED: 动态设置Y轴下限
    if all_energies_list:
        all_dN_dE = np.concatenate([dN_dE_initial[dN_dE_initial > 0], dN_dE_final[dN_dE_final > 0]])
        if all_dN_dE.size > 0:
            ax1.set_ylim(bottom=all_dN_dE.min() * 0.1)

    # ... (参数表格部分保持不变，但增加束流参数) ...
    ax2.axis('off')
    ax2.set_title('模拟参数', y=0.95)

    def prepare_table_data(sim_obj):
        params = [
            ("--- 物理参数 ---",),
            ("温度 T", f"{sim_obj.T_plasma * 1e-6:.3f} MeV"),
            ("总数密度 n_plasma", f"{sim_obj.n_plasma:.2e} m^-3"),
            ("磁场 B0", f"{sim_obj.B0:.2f} T"),
            ("磁化强度 σ", f"{sim_obj.sigma:.3f}"),
            ("相对论热参数 θ", f"{sim_obj.theta:.3f}"),
            ("--- 束流参数 ---",),  # <<< NEW
            ("束流占比", f"{sim_obj.beam_fraction * 100:.0f} %"),
            ("束流动量 u_drift", f"{sim_obj.beam_u_drift:.2f}"),
            ("--- 数值参数 ---",),
            ("网格", f"{sim_obj.NX} x {sim_obj.NZ}"),
            ("模拟域 (d_e)", f"{sim_obj.LX:.1f} x {sim_obj.LZ:.1f}"),
            ("总步数", f"{sim_obj.total_steps}"),
            ("时间步长 (1/ω_pe)", f"{sim_obj.DT:.3f}"),
            ("每单元总粒子数", f"{sim_obj.NPPC}"),
        ]
        row_labels, cell_text = [], []
        for item in params:
            if len(item) == 1:
                row_labels.append(item[0]);
                cell_text.append([''])
            else:
                row_labels.append(f"  {item[0]}");
                cell_text.append([item[1]])
        return row_labels, cell_text

    row_labels, cell_text = prepare_table_data(sim)
    table = ax2.table(cellText=cell_text, rowLabels=row_labels, colLabels=[f"值: {run_prefix}"], loc='center',
                      cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2.0)
    for (row, col), cell in table.get_celld().items():
        if row == 0: cell.set_text_props(weight='bold', color='k'); cell.set_facecolor('#B0C4DE')
        if col == -1:
            cell.set_text_props(ha='left')
            if "---" in cell.get_text().get_text(): cell.set_text_props(weight='bold'); cell.set_facecolor('#E0E0E0')

    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    output_path = os.path.join(output_folder, f"{run_prefix}_summary_v2.png")
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  -> (2/2) 摘要图已保存到: {output_path}")


# ... (main 函数保持不变) ...
def main():
    param_file = "sim_parameters.dpkl"
    print(f"正在当前目录中查找参数文件 '{param_file}'...")
    if not os.path.exists(param_file):
        print(f"\n错误：未找到 '{param_file}'。\n请确保此脚本与模拟输出文件在同一个目录中。")
        return
    try:
        with open(param_file, "rb") as f:
            sim = dill.load(f)
        print(f"成功加载参数文件。")
    except Exception as e:
        print(f"\n错误：加载 '{param_file}' 时出错: {e}。")
        return
    create_summary_plot(sim, source_folder='.')
    print("\n分析完成。")


if __name__ == "__main__":
    main()