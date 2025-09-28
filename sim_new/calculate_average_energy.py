#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# --- 粒子平均能量计算脚本 ---
#
# 功能:
# 1. 加载指定 WarpX 模拟的最后一个诊断步。
# 2. 计算所有带电粒子的加权平均动能。
# 3. 输出结果，供后续分析（如 Mathematica）使用。
#

import os
import glob
import h5py
import dill
import argparse
import numpy as np
from scipy import constants
from dataclasses import dataclass
from typing import Optional

# --- 数据结构定义 ---
@dataclass
class SpectrumData:
    """存放能谱数据"""
    energies_MeV: np.ndarray
    weights: np.ndarray

# --- 全局常量 ---
C = constants.c
M_E = constants.m_e
E = constants.e
J_PER_MEV = E * 1e6

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
                print(f"  [Warning] 在文件 {os.path.basename(h5_filepath)} 中未找到带电粒子。")
                return None

            for species in charged_species:
                base_path = f"data/{step_key}/particles/{species}/"
                px = f[base_path + 'momentum/x'][:]
                py = f[base_path + 'momentum/y'][:]
                pz = f[base_path + 'momentum/z'][:]
                weights = f[base_path + 'weighting'][:]

                if weights.size == 0:
                    continue

                p_sq = px ** 2 + py ** 2 + pz ** 2
                kinetic_energy_J = np.sqrt(p_sq * C ** 2 + m_e_c2_J ** 2) - m_e_c2_J
                all_energies_MeV.append(kinetic_energy_J / J_PER_MEV)
                all_weights.append(weights)

        if not all_energies_MeV:
            print(f"  [Warning] 文件 {os.path.basename(h5_filepath)} 中没有可用的粒子能谱数据。")
            return None

        return SpectrumData(np.concatenate(all_energies_MeV), np.concatenate(all_weights))

    except Exception as e:
        print(f"  [Error] 加载能谱 {os.path.basename(h5_filepath)} 时发生意外错误: {e}")
        return None

def main():
    """主执行函数"""
    parser = argparse.ArgumentParser(
        description="计算 WarpX 模拟中最终时刻粒子的加权平均动能。"
    )
    parser.add_argument(
        "sim_directory",
        type=str,
        help="包含 'diags/particle_states' 的模拟文件夹路径。"
    )
    args = parser.parse_args()

    dir_path = args.sim_directory
    if not os.path.isdir(dir_path):
        print(f"[Error] 目录不存在: {dir_path}")
        return

    print(f"--- 正在分析模拟: {os.path.basename(dir_path)} ---")

    # 找到最后一个 HDF5 文件
    particle_files = sorted(glob.glob(os.path.join(dir_path, "diags/particle_states", "openpmd_*.h5")))
    if not particle_files:
        print("[Error] 在 'diags/particle_states/' 目录下找不到任何 HDF5 文件。")
        return

    final_file = particle_files[-1]
    print(f"加载最终状态文件: {os.path.basename(final_file)}")

    final_spectrum = _load_spectrum_from_file(final_file)

    if not final_spectrum or final_spectrum.weights.size == 0:
        print("[Error] 未能从最终状态加载任何粒子数据。")
        return

    # 计算加权平均动能
    average_kinetic_energy_MeV = np.average(
        final_spectrum.energies_MeV,
        weights=final_spectrum.weights
    )

    # 计算总能量作为参考
    total_kinetic_energy_J = np.sum(
        final_spectrum.energies_MeV * J_PER_MEV * final_spectrum.weights
    )

    print("\n" + "="*40)
    print("           计算结果")
    print("="*40)
    print(f"总粒子数 (加权): {np.sum(final_spectrum.weights):.3e}")
    print(f"总动能 (焦耳):   {total_kinetic_energy_J:.3e} J")
    print(f"平均动能 (MeV):  {average_kinetic_energy_MeV:.6f} MeV")
    print("="*40)
    print("\n请将上面的 '平均动能 (MeV)' 值复制到您的 Mathematica 脚本中。")


if __name__ == "__main__":
    main()