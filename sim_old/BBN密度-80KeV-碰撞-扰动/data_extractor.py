#
# --- 数据提取脚本 (用于与Mathematica联动) ---
#
import os
import glob
import dill
import h5py
import json
import numpy as np
from scipy import constants


def extract_data_for_export(source_folder='.'):
    """
    从模拟结果中提取能谱数据和关键参数，并返回一个字典。
    """
    print("  -> (1/2) 正在加载参数文件...")
    param_file = os.path.join(source_folder, "sim_parameters.dpkl")
    if not os.path.exists(param_file):
        raise FileNotFoundError(f"参数文件 '{param_file}' 未找到。")
    with open(param_file, "rb") as f:
        sim = dill.load(f)

    print("  -> (2/2) 正在提取能谱数据...")

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
                # 自动发现HDF5文件中的所有粒子物种
                species_in_file = list(f[f'data/{current_time_step}/particles'].keys())
                print(f"     -> 在文件 {os.path.basename(target_file)} 中发现物种: {species_in_file}")

                # 过滤掉光子等中性粒子
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

    # 加载数据
    initial_energies, initial_weights = _load_spectrum_data(source_folder, file_index=0)
    final_energies, final_weights = _load_spectrum_data(source_folder, file_index=-1)

    # 准备输出字典
    output_data = {
        "metadata": {
            "run_prefix": os.path.basename(os.path.abspath(source_folder)),
            "source_folder": os.path.abspath(source_folder)
        },
        "parameters": {
            "T_plasma_eV": sim.T_plasma,
            "n_plasma_m3": sim.n_plasma,
            "B0_T": sim.B0,
            "sigma": sim.sigma,
            "theta": sim.theta,
            "beam_fraction": sim.beam_fraction,
            "beam_u_drift": sim.beam_u_drift,
            "NX": sim.NX,
            "NZ": sim.NZ,
            "LX_de": sim.LX,
            "LZ_de": sim.LZ,
            "total_steps": sim.total_steps,
            "DT_omegap": sim.DT,
            "NPPC": sim.NPPC,
        },
        "spectrum_data": {}
    }

    # 计算能谱密度
    all_energies_list = [e for e in [initial_energies, final_energies] if e is not None]
    if not all_energies_list:
        return output_data  # 返回包含参数但没有能谱的数据

    combined_energies = np.concatenate(all_energies_list)
    positive_energies = combined_energies[combined_energies > 0]
    if positive_energies.size > 1:
        num_bins = 150
        min_E = max(positive_energies.min() * 0.5, 1e-4)
        max_E = positive_energies.max() * 1.1
        common_bins_MeV = np.logspace(np.log10(min_E), np.log10(max_E), num_bins + 1)
        bin_centers_MeV = np.sqrt(common_bins_MeV[:-1] * common_bins_MeV[1:])
        bin_widths_MeV = common_bins_MeV[1:] - common_bins_MeV[:-1]

        if initial_energies is not None:
            initial_counts, _ = np.histogram(initial_energies, bins=common_bins_MeV, weights=initial_weights)
            dN_dE_initial = initial_counts / bin_widths_MeV
            # 将 NumPy 数组转为 Python 列表以便 JSON 序列化
            output_data["spectrum_data"]["initial"] = {
                "bin_centers_MeV": bin_centers_MeV.tolist(),
                "dN_dE": dN_dE_initial.tolist()
            }
            output_data["parameters"]["N_total_initial"] = float(np.sum(initial_weights))

        if final_energies is not None:
            final_counts, _ = np.histogram(final_energies, bins=common_bins_MeV, weights=final_weights)
            dN_dE_final = final_counts / bin_widths_MeV
            output_data["spectrum_data"]["final"] = {
                "bin_centers_MeV": bin_centers_MeV.tolist(),
                "dN_dE": dN_dE_final.tolist()
            }
            output_data["parameters"]["N_total_final"] = float(np.sum(final_weights))

    return output_data


def main():
    run_folder = '.'
    output_filename = os.path.join(run_folder, "analysis_plots", "analysis_data.json")

    try:
        data_to_export = extract_data_for_export(run_folder)
        os.makedirs(os.path.dirname(output_filename), exist_ok=True)

        with open(output_filename, 'w') as f:
            json.dump(data_to_export, f, indent=4)

        print(f"\n数据提取完成，已保存到: {output_filename}")

    except Exception as e:
        print(f"\n错误：数据提取过程中发生错误: {e}")


if __name__ == "__main__":
    main()