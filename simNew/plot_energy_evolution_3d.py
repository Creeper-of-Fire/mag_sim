#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# --- 交互式场能与动能演化对比分析脚本 (3D 版本) ---
#
# 功能:
# 1. 对比分析磁场能量与粒子动能随时间的演化。
# 2. 从 diags/fields/ 加载 .npz 文件计算磁能。
# 3. 从 diags/particle_states/ 加载 .h5 文件计算动能。
# 4. 绘制能量密度分量图，分析能量各向异性。
# 5. 绘制总能量演化图，分析系统能量转换。
# 6. 为每次模拟生成独立的分析图，并附带参数表。
#
import glob
import os
from types import SimpleNamespace
from typing import List, Optional

import dill
import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import c, m_e, mu_0, e

from warpx_analysis_utils import (
    console,
    setup_chinese_font,
    select_directories,
    plot_parameter_table,
    SimulationRun,
    EnergyEvolutionData,
    SpectrumData
)


# =============================================================================
# 核心数据加载与计算函数
# =============================================================================

def _center_field(field: np.ndarray, target_shape: tuple) -> np.ndarray:
    """将一个在3D交错网格上的场分量插值到单元中心。"""
    if field.shape == target_shape:
        return field
    nx, ny, nz = target_shape
    if field.shape == (nx + 1, ny, nz): return 0.5 * (field[:-1, :, :] + field[1:, :, :])
    if field.shape == (nx, ny + 1, nz): return 0.5 * (field[:, :-1, :] + field[:, 1:, :])
    if field.shape == (nx, ny, nz + 1): return 0.5 * (field[:, :, :-1] + field[:, :, 1:])
    console.print(f"[yellow]警告: 场形状 {field.shape} 无法插值到 {target_shape}。尝试裁剪。[/yellow]")
    return field[:nx, :ny, :nz]


def _calculate_magnetic_energy(fpath: str, target_shape: tuple, cell_volume: float):
    """从单个 .npz 文件计算磁能密度和总磁能。"""
    with np.load(fpath) as data:
        B_norm = data['B_norm']
        Bx_s = data['Bx'] * B_norm
        By_s = data['By'] * B_norm
        Bz_s = data['Bz'] * B_norm

    Bx = _center_field(Bx_s, target_shape)
    By = _center_field(By_s, target_shape)
    Bz = _center_field(Bz_s, target_shape)

    # 能量密度 (J/m^3)
    energy_density_x = (Bx ** 2) / (2 * mu_0)
    energy_density_y = (By ** 2) / (2 * mu_0)
    energy_density_z = (Bz ** 2) / (2 * mu_0)
    energy_density_total = energy_density_x + energy_density_y + energy_density_z

    # 空间平均能量密度
    mean_edens_x = np.mean(energy_density_x)
    mean_edens_y = np.mean(energy_density_y)
    mean_edens_z = np.mean(energy_density_z)
    mean_edens_total = np.mean(energy_density_total)

    # 盒子内总能量 (J)
    total_energy = np.sum(energy_density_total) * cell_volume

    return mean_edens_x, mean_edens_y, mean_edens_z, mean_edens_total, total_energy


def _calculate_kinetic_energy(fpath: str, sim_volume: float):
    """从单个 .h5 文件计算总动能和平均动能密度。"""
    total_kinetic_energy = 0.0
    m_e_c2_J = m_e * c ** 2

    try:
        with h5py.File(fpath, 'r') as f:
            step_key = list(f['data'].keys())[0]
            particles_group = f[f'data/{step_key}/particles']

            for species in particles_group.keys():
                if 'photon' in species: continue  # 跳过光子

                px = particles_group[f'{species}/momentum/x'][:]
                py = particles_group[f'{species}/momentum/y'][:]
                pz = particles_group[f'{species}/momentum/z'][:]
                weights = particles_group[f'{species}/weighting'][:]

                if weights.size == 0: continue

                p_sq = px ** 2 + py ** 2 + pz ** 2
                kinetic_energy_J = np.sqrt(p_sq * c ** 2 + m_e_c2_J ** 2) - m_e_c2_J
                total_kinetic_energy += np.sum(kinetic_energy_J * weights)

    except Exception as e:
        console.print(f"  [red]✗ 处理粒子文件 {os.path.basename(fpath)} 时出错: {e}[/red]")
        return None, None

    mean_energy_density = total_kinetic_energy / sim_volume if sim_volume > 0 else 0
    return total_kinetic_energy, mean_energy_density


def _get_step_from_filename(filename: str) -> Optional[int]:
    """从 WarpX 诊断文件名中稳健地提取步数。"""
    try:
        # e.g., 'fields_000100.npz' -> '000100' -> 100
        # e.g., 'openpmd_000100.h5' -> '000100' -> 100
        base_name = os.path.basename(filename)
        step_str = base_name.split('_')[-1].split('.')[0]
        return int(step_str)
    except (IndexError, ValueError):
        # 如果文件名格式不符合预期，则忽略
        console.print(f"[yellow]警告: 无法从文件名 '{filename}' 中解析步数。[/yellow]")
        return None

def _load_spectrum_from_file(h5_filepath: str) -> Optional[SpectrumData]:
    """从单个 HDF5 文件中加载所有带电粒子的能谱。"""
    all_energies_MeV, all_weights = [], []
    m_e_c2_J = m_e * c**2
    J_PER_MEV = e * 1e6

    try:
        with h5py.File(h5_filepath, 'r') as f:
            step_key = list(f['data'].keys())[0]
            particles_group = f[f'data/{step_key}/particles']
            for species in particles_group.keys():
                if 'photon' in species: continue
                px = particles_group[f'{species}/momentum/x'][:]
                py = particles_group[f'{species}/momentum/y'][:]
                pz = particles_group[f'{species}/momentum/z'][:]
                weights = particles_group[f'{species}/weighting'][:]
                if weights.size == 0: continue
                p_sq = px**2 + py**2 + pz**2
                kinetic_energy_J = np.sqrt(p_sq * c**2 + m_e_c2_J**2) - m_e_c2_J
                all_energies_MeV.append(kinetic_energy_J / J_PER_MEV)
                all_weights.append(weights)
        if not all_energies_MeV: return None
        return SpectrumData(np.concatenate(all_energies_MeV), np.concatenate(all_weights))
    except Exception as err:
        console.print(f"  [red]  -> ✗ 加载能谱 {os.path.basename(h5_filepath)} 时出错: {err}[/red]")
        return None

def load_energy_evolution_data(dir_path: str, sim_obj: object) -> Optional[EnergyEvolutionData]:
    """
    从 .npz 和 .h5 文件序列中加载磁能和动能演化数据。
    """
    field_dir = os.path.join(dir_path, "diags/fields")
    particle_dir = os.path.join(dir_path, "diags/particle_states")

    if not os.path.isdir(field_dir):
        console.print(f"  [yellow]⚠ 找不到磁场诊断目录 '{field_dir}'。[/yellow]")
        return None
    if not os.path.isdir(particle_dir):
        console.print(f"  [yellow]⚠ 找不到粒子诊断目录 '{particle_dir}'。[/yellow]")
        return None

    field_files = {step: f for f in glob.glob(os.path.join(field_dir, "fields_*.npz")) if (step := _get_step_from_filename(f)) is not None}
    particle_files = {step: f for f in glob.glob(os.path.join(particle_dir, "openpmd_*.h5")) if (step := _get_step_from_filename(f)) is not None}

    # 找到共有的时间步
    common_steps = sorted(list(set(field_files.keys()) & set(particle_files.keys())))

    if not common_steps:
        console.print(f"  [yellow]⚠ 找不到任何匹配的磁场和粒子诊断文件。[/yellow]")
        return None

    console.print(f"  [white]正在处理 {len(common_steps)} 个匹配的时间步...[/white]")

    # --- 初始化数据列表 ---
    times = []
    mag_edens_x, mag_edens_y, mag_edens_z, mag_edens_tot = [], [], [], []
    kin_edens = []
    total_mag_E, total_kin_E = [], []

    # --- 模拟参数 ---
    target_shape = (sim_obj.NX, sim_obj.NY, sim_obj.NZ)

    if not all(hasattr(sim_obj, attr) for attr in ['dx', 'dy', 'dz']):
        console.print("  [yellow]警告: 模拟参数对象中缺少 dx, dy, dz。将从 Lx/NX 等计算。[/yellow]")
        dx = sim_obj.Lx / sim_obj.NX
        dy = sim_obj.Ly / sim_obj.NY
        dz = sim_obj.Lz / sim_obj.NZ
    else:
        dx, dy, dz = sim_obj.dx, sim_obj.dy, sim_obj.dz
    cell_volume = dx * dy * dz

    sim_volume = cell_volume * sim_obj.NX * sim_obj.NY * sim_obj.NZ

    for step in common_steps:
        times.append(step * sim_obj.dt)

        # 1. 计算磁能
        fpath_field = field_files[step]
        m_ed_x, m_ed_y, m_ed_z, m_ed_tot, tot_m_E = _calculate_magnetic_energy(fpath_field, target_shape, cell_volume)
        mag_edens_x.append(m_ed_x)
        mag_edens_y.append(m_ed_y)
        mag_edens_z.append(m_ed_z)
        mag_edens_tot.append(m_ed_tot)
        total_mag_E.append(tot_m_E)

        # 2. 计算动能
        fpath_particle = particle_files[step]
        tot_k_E, k_ed = _calculate_kinetic_energy(fpath_particle, sim_volume)
        if tot_k_E is None: continue  # 如果读取失败则跳过此步
        total_kin_E.append(tot_k_E)
        kin_edens.append(k_ed)

    if not times:
        return None

    return EnergyEvolutionData(
        time=np.array(times),
        mean_mag_energy_density_x=np.array(mag_edens_x),
        mean_mag_energy_density_y=np.array(mag_edens_y),
        mean_mag_energy_density_z=np.array(mag_edens_z),
        mean_mag_energy_density_total=np.array(mag_edens_tot),
        mean_kin_energy_density=np.array(kin_edens),
        total_magnetic_energy=np.array(total_mag_E),
        total_kinetic_energy=np.array(total_kin_E)
    )


# =============================================================================
# 绘图函数
# =============================================================================

def generate_energy_evolution_plot(runs: List['SimulationRun']):
    """为每个选定的模拟生成一张独立的能量演化分析图。"""
    console.print("\n[bold magenta]正在为每个模拟生成独立的能量演化图...[/bold magenta]")

    J_PER_EV = e  # 焦耳与eV的转换因子

    for i, run in enumerate(runs):
        output_name = f"energy_evolution_3d_{run.name}.png"
        console.print(f"\n--- ({i + 1}/{len(runs)}) 正在处理 [bold]{run.name}[/bold] ---")

        if not run.energy_data:
            console.print(f"  [yellow]⚠ 警告: 模拟 '{run.name}' 缺少能量数据，已跳过。[/yellow]")
            continue

        data = run.energy_data

        kin_density_ev = data.mean_kin_energy_density / J_PER_EV
        mag_density_total_ev = data.mean_mag_energy_density_total / J_PER_EV
        mag_density_x_ev = data.mean_mag_energy_density_x / J_PER_EV
        mag_density_y_ev = data.mean_mag_energy_density_y / J_PER_EV
        mag_density_z_ev = data.mean_mag_energy_density_z / J_PER_EV

        total_kin_ev = data.total_kinetic_energy / J_PER_EV
        total_mag_ev = data.total_magnetic_energy / J_PER_EV

        fig, (ax_density, ax_total, ax_table) = plt.subplots(
            3, 1,
            figsize=(12, 18),
            gridspec_kw={'height_ratios': [4, 4, 3]},
            constrained_layout=True
        )
        fig.suptitle(f"能量演化分析: {run.name}", fontsize=18, y=1.02)

        # --- 子图1: 平均能量密度演化 ---
        ax_density.set_title('平均能量密度演化', fontsize=14)
        ax_density.plot(data.time, kin_density_ev, '-', color='black', lw=2.5, label=r'$\langle \epsilon_K \rangle$ (动能)')
        ax_density.plot(data.time, mag_density_total_ev, '--', color='purple', lw=2.5, label=r'$\langle \epsilon_B \rangle$ (总磁能)')
        ax_density.plot(data.time, mag_density_x_ev, ':', color='red', lw=1.5, label=r'$\langle \epsilon_{B,x} \rangle$')
        ax_density.plot(data.time, mag_density_y_ev, ':', color='green', lw=1.5, label=r'$\langle \epsilon_{B,y} \rangle$')
        ax_density.plot(data.time, mag_density_z_ev, ':', color='blue', lw=1.5, label=r'$\langle \epsilon_{B,z} \rangle$')

        ax_density.set_ylabel(r'平均能量密度 (eV/m$^3$)', fontsize=12)
        ax_density.set_yscale('log')
        ax_density.grid(True, which="both", ls="--", alpha=0.6)
        ax_density.legend(fontsize=11)

        # --- 子图2: 总能量演化 ---
        ax_total.set_title('盒子内总能量演化', fontsize=14)
        ax_total.plot(data.time, total_kin_ev, '-', color='black', lw=2.5, label=r'$E_{K, tot}$ (总动能)')
        ax_total.plot(data.time, total_mag_ev, '--', color='purple', lw=2.5, label=r'$E_{B, tot}$ (总磁能)')

        # 绘制总能量
        total_energy_ev = total_kin_ev + total_mag_ev
        ax_total.plot(data.time, total_energy_ev, '-', color='orange', lw=2, alpha=0.8, label=r'$E_{K, tot} + E_{B, tot}$')

        ax_total.set_xlabel('时间 (s)', fontsize=12)
        ax_total.set_ylabel('总能量 (eV)', fontsize=12)
        ax_total.set_yscale('log')
        ax_total.grid(True, which="both", ls="--", alpha=0.6)
        ax_total.legend(fontsize=11)

        # --- 子图3: 参数表 ---
        plot_parameter_table(ax_table, run)

        plt.savefig(output_name, dpi=200, bbox_inches='tight')
        plt.close(fig)
        console.print(f"[bold green]✔ 能量演化图已成功保存到: {output_name}[/bold green]")


# =============================================================================
# 主交互流程
# =============================================================================
def main():
    """主执行函数"""
    console.print("[bold inverse] WarpX 3D 能量演化对比分析器 [/bold inverse]")
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
                # 加载字典，并将其转换为一个方便访问的对象
                param_dict = dill.load(f)
                sim_obj = SimpleNamespace(**param_dict)
            console.print("  [green]✔ 成功加载参数文件。[/green]")

            # 加载能量演化数据
            energy_data = load_energy_evolution_data(dir_path, sim_obj)

            # 加载初始能谱数据 (为了参数表)
            initial_spectrum = None
            particle_files = sorted(glob.glob(os.path.join(dir_path, "diags/particle_states", "openpmd_*.h5")))
            if particle_files:
                console.print("  [white]正在加载初始能谱 (用于计算总粒子数)...[/white]")
                initial_spectrum = _load_spectrum_from_file(particle_files[0])
            else:
                console.print("  [yellow]⚠ 警告: 找不到粒子文件，无法计算总粒子数。[/yellow]")

            run_instance = SimulationRun(
                path=dir_path,
                name=os.path.basename(dir_path),
                sim=sim_obj,
                energy_data=energy_data,
                initial_spectrum=initial_spectrum
            )
            loaded_runs.append(run_instance)
        except Exception as e:
            console.print(f"  [red]✗ 加载模拟 {os.path.basename(dir_path)} 失败: {e}[/red]")
            continue

    valid_runs = [run for run in loaded_runs if run.energy_data]
    if not valid_runs:
        console.print("\n[red]未能成功加载任何能量演化数据，无法生成图像。[/red]")
        return

    generate_energy_evolution_plot(valid_runs)

    console.print("\n[bold]分析完成。[/bold]")


if __name__ == "__main__":
    main()
