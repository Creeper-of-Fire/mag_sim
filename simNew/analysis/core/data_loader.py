# core/data_loader.py

# !/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# --- 统一数据加载模块 ---
#
# 包含所有从磁盘文件读取和计算物理量的函数。
# 提供一个高级接口 `load_run_data` 来按需加载数据。
#
import glob
import os
from types import SimpleNamespace
from typing import Set, Optional, Tuple

import dill
import h5py
import numpy as np
from scipy.constants import c, m_e, mu_0, e, epsilon_0

from .simulation import (SimulationRun, EnergyEvolutionData, FieldEvolutionData, SpectrumData)
from .utils import console


# =============================================================================
# 1. 底层计算和辅助函数
# =============================================================================

def _get_step_from_filename(filename: str) -> Optional[int]:
    """从 WarpX 诊断文件名中稳健地提取步数。"""
    try:
        base_name = os.path.basename(filename)
        step_str = base_name.split('_')[-1].split('.')[0]
        return int(step_str)
    except (IndexError, ValueError):
        return None


def _center_field_3d(field: np.ndarray, target_shape: tuple) -> np.ndarray:
    """将3D交错网格场分量稳健地插值到单元中心。"""
    if field.shape == target_shape:
        return field
    nx, ny, nz = target_shape
    if field.shape == (nx + 1, ny, nz): return 0.5 * (field[:-1, :, :] + field[1:, :, :])
    if field.shape == (nx, ny + 1, nz): return 0.5 * (field[:, :-1, :] + field[:, 1:, :])
    if field.shape == (nx, ny, nz + 1): return 0.5 * (field[:, :, :-1] + field[:, :, 1:])
    if field.shape == (nx, ny + 1, nz + 1):
        field_y_avg = 0.5 * (field[:, :-1, :] + field[:, 1:, :])
        return 0.5 * (field_y_avg[:, :, :-1] + field_y_avg[:, :, 1:])
    if field.shape == (nx + 1, ny, nz + 1):
        field_x_avg = 0.5 * (field[:-1, :, :] + field[1:, :, :])
        return 0.5 * (field_x_avg[:, :, :-1] + field_x_avg[:, :, 1:])
    if field.shape == (nx + 1, ny + 1, nz):
        field_x_avg = 0.5 * (field[:-1, :, :] + field[1:, :, :])
        return 0.5 * (field_x_avg[:, :-1, :] + field_x_avg[:, 1:, :])
    console.print(f"[red]错误: 无法处理的场形状 {field.shape}，目标为 {target_shape}。将进行裁剪。[/red]")
    return field[:nx, :ny, :nz]


# =============================================================================
# 2. 模块化数据加载函数 (按需调用)
# =============================================================================

def _load_spectrum_data(h5_filepath: str) -> Optional[SpectrumData]:
    """从单个 HDF5 文件中加载所有带电粒子的能谱。"""
    all_energies_MeV, all_weights = [], []
    m_e_c2_J = m_e * c ** 2
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
                p_sq = px ** 2 + py ** 2 + pz ** 2
                kinetic_energy_J = np.sqrt(p_sq * c ** 2 + m_e_c2_J ** 2) - m_e_c2_J
                all_energies_MeV.append(kinetic_energy_J / J_PER_MEV)
                all_weights.append(weights)
        if not all_energies_MeV: return None
        return SpectrumData(np.concatenate(all_energies_MeV), np.concatenate(all_weights))
    except Exception as err:
        console.print(f"  [red] -> ✗ 加载能谱 {os.path.basename(h5_filepath)} 时出错: {err}[/red]")
        return None


def _load_field_evolution_data(dir_path: str, sim_obj: object) -> Optional[FieldEvolutionData]:
    """从 .npz 文件序列中加载磁场演化数据 (3D版本)。"""
    field_files = sorted(glob.glob(os.path.join(dir_path, "diags/fields", "fields_*.npz")))
    if not field_files:
        console.print(f"  [yellow]⚠ 在 'diags/fields/' 目录下找不到任何 .npz 文件。[/yellow]")
        return None

    times, b_max_vals = [], []
    b_mean_x_vals, b_mean_y_vals, b_mean_z_vals = [], [], []
    b_rms_x_vals, b_rms_y_vals, b_rms_z_vals = [], [], []
    b_mean_abs_vals = []
    target_shape = (sim_obj.NX, sim_obj.NY, sim_obj.NZ)

    for fpath in field_files:
        try:
            step = _get_step_from_filename(fpath)
            if step is None: continue

            with np.load(fpath) as data:
                Bx_s, By_s, Bz_s = data['Bx'], data['By'], data['Bz']

            Bx = _center_field_3d(Bx_s, target_shape)
            By = _center_field_3d(By_s, target_shape)
            Bz = _center_field_3d(Bz_s, target_shape)

            b_mean_x_vals.append(np.mean(Bx))
            b_mean_y_vals.append(np.mean(By))
            b_mean_z_vals.append(np.mean(Bz))
            b_rms_x_vals.append(np.sqrt(np.mean(Bx ** 2)))
            b_rms_y_vals.append(np.sqrt(np.mean(By ** 2)))
            b_rms_z_vals.append(np.sqrt(np.mean(Bz ** 2)))
            b_magnitude = np.sqrt(Bx ** 2 + By ** 2 + Bz ** 2)
            b_max_vals.append(np.max(b_magnitude))
            b_mean_abs_vals.append(np.mean(b_magnitude))
            times.append(step * sim_obj.dt)
        except Exception as e:
            console.print(f"  [red]✗ 处理场文件 {os.path.basename(fpath)} 时出错: {e}[/red]")
            continue

    if not times: return None
    return FieldEvolutionData(
        time=np.array(times), b_max_normalized=np.array(b_max_vals),
        b_mean_x_normalized=np.array(b_mean_x_vals), b_mean_y_normalized=np.array(b_mean_y_vals),
        b_mean_z_normalized=np.array(b_mean_z_vals), b_mean_abs_normalized=np.array(b_mean_abs_vals),
        b_rms_x_normalized=np.array(b_rms_x_vals), b_rms_y_normalized=np.array(b_rms_y_vals),
        b_rms_z_normalized=np.array(b_rms_z_vals)
    )


def _load_energy_evolution_data(dir_path: str, sim_obj: object) -> Optional[EnergyEvolutionData]:
    """从 .npz 和 .h5 文件序列中加载磁能、电能和动能演化数据。"""
    field_dir = os.path.join(dir_path, "diags/fields")
    particle_dir = os.path.join(dir_path, "diags/particle_states")
    field_files = {s: f for f in glob.glob(os.path.join(field_dir, "*.npz")) if (s := _get_step_from_filename(f)) is not None}
    particle_files = {s: f for f in glob.glob(os.path.join(particle_dir, "*.h5")) if (s := _get_step_from_filename(f)) is not None}

    common_steps = sorted(list(set(field_files.keys()) & set(particle_files.keys())))
    if not common_steps: return None

    times = []
    mag_edens_x, mag_edens_y, mag_edens_z, mag_edens_tot = [], [], [], []
    elec_edens_x, elec_edens_y, elec_edens_z, elec_edens_tot = [], [], [], []
    kin_edens = []
    total_mag_E, total_elec_E, total_kin_E = [], [], []

    target_shape = (sim_obj.NX, sim_obj.NY, sim_obj.NZ)
    dx = sim_obj.Lx / sim_obj.NX
    dy = sim_obj.Ly / sim_obj.NY
    dz = sim_obj.Lz / sim_obj.NZ
    cell_volume = dx * dy * dz
    sim_volume = cell_volume * np.prod(target_shape)

    for step in common_steps:
        times.append(step * sim_obj.dt)

        # 场能
        with np.load(field_files[step]) as data:
            B_norm = data['B_norm']
            Bx = _center_field_3d(data['Bx'] * B_norm, target_shape)
            By = _center_field_3d(data['By'] * B_norm, target_shape)
            Bz = _center_field_3d(data['Bz'] * B_norm, target_shape)
            Ex = _center_field_3d(data['Ex'], target_shape)
            Ey = _center_field_3d(data['Ey'], target_shape)
            Ez = _center_field_3d(data['Ez'], target_shape)

        mag_ed_x, mag_ed_y, mag_ed_z = (Bx ** 2) / (2 * mu_0), (By ** 2) / (2 * mu_0), (Bz ** 2) / (2 * mu_0)
        elec_ed_x, elec_ed_y, elec_ed_z = (epsilon_0 * Ex ** 2) / 2, (epsilon_0 * Ey ** 2) / 2, (epsilon_0 * Ez ** 2) / 2

        mag_edens_x.append(np.mean(mag_ed_x));
        mag_edens_y.append(np.mean(mag_ed_y));
        mag_edens_z.append(np.mean(mag_ed_z))
        mag_edens_tot.append(np.mean(mag_ed_x + mag_ed_y + mag_ed_z))
        total_mag_E.append(np.sum(mag_ed_x + mag_ed_y + mag_ed_z) * cell_volume)

        elec_edens_x.append(np.mean(elec_ed_x));
        elec_edens_y.append(np.mean(elec_ed_y));
        elec_edens_z.append(np.mean(elec_ed_z))
        elec_edens_tot.append(np.mean(elec_ed_x + elec_ed_y + elec_ed_z))
        total_elec_E.append(np.sum(elec_ed_x + elec_ed_y + elec_ed_z) * cell_volume)

        # 动能
        tot_k_E, k_ed = 0.0, 0.0
        m_e_c2_J = m_e * c ** 2
        with h5py.File(particle_files[step], 'r') as f:
            step_key = list(f['data'].keys())[0]
            particles_group = f[f'data/{step_key}/particles']
            for species in particles_group.keys():
                if 'photon' in species: continue
                px, py, pz, weights = [particles_group[f'{species}/{k}'][:] for k in ['momentum/x', 'momentum/y', 'momentum/z', 'weighting']]
                if weights.size == 0: continue
                p_sq = px ** 2 + py ** 2 + pz ** 2
                kinetic_energy_J = np.sqrt(p_sq * c ** 2 + m_e_c2_J ** 2) - m_e_c2_J
                tot_k_E += np.sum(kinetic_energy_J * weights)
        k_ed = tot_k_E / sim_volume if sim_volume > 0 else 0
        total_kin_E.append(tot_k_E)
        kin_edens.append(k_ed)

    if not times: return None
    return EnergyEvolutionData(
        time=np.array(times), mean_mag_energy_density_x=np.array(mag_edens_x),
        mean_mag_energy_density_y=np.array(mag_edens_y), mean_mag_energy_density_z=np.array(mag_edens_z),
        mean_mag_energy_density_total=np.array(mag_edens_tot),
        mean_elec_energy_density_x=np.array(elec_edens_x), mean_elec_energy_density_y=np.array(elec_edens_y),
        mean_elec_energy_density_z=np.array(elec_edens_z), mean_elec_energy_density_total=np.array(elec_edens_tot),
        mean_kin_energy_density=np.array(kin_edens), total_magnetic_energy=np.array(total_mag_E),
        total_electric_energy=np.array(total_elec_E), total_kinetic_energy=np.array(total_kin_E)
    )


# =============================================================================
# 3. 高级统一加载接口
# =============================================================================

def load_run_data(dir_path: str, required_data: Set[str]) -> Optional[SimulationRun]:
    """
    为单个模拟目录加载所有需要的数据。

    Args:
        dir_path (str): 模拟数据文件夹路径。
        required_data (Set[str]): 一个包含所需数据类型的集合,
            例如 {'energy', 'initial_spectrum', 'field'}.

    Returns:
        Optional[SimulationRun]: 一个填充了所需数据的 SimulationRun 对象，
                                 如果加载失败则返回 None。
    """
    console.print(f"\n[bold cyan]正在加载模拟: {os.path.basename(dir_path)}[/bold cyan]")
    param_file = os.path.join(dir_path, "sim_parameters.dpkl")
    if not os.path.exists(param_file):
        console.print(f"  [red]✗ 错误: 找不到参数文件 '{param_file}'。[/red]")
        return None

    try:
        with open(param_file, "rb") as f:
            sim_obj = SimpleNamespace(**dill.load(f))
        console.print("  [green]✔ 成功加载参数文件。[/green]")
    except Exception as e:
        console.print(f"  [red]✗ 加载参数文件失败: {e}[/red]")
        return None

    run = SimulationRun(path=dir_path, name=os.path.basename(dir_path), sim=sim_obj)

    # --- 按需加载数据 ---
    particle_files = sorted(glob.glob(os.path.join(dir_path, "diags/particle_states", "openpmd_*.h5")))
    run.particle_files = particle_files

    if 'initial_spectrum' in required_data and particle_files:
        console.print("  [white]  -> 正在加载初始能谱...[/white]")
        run.initial_spectrum = _load_spectrum_data(particle_files[0])

    if 'final_spectrum' in required_data and particle_files:
        if len(particle_files) > 1:
            console.print("  [white]  -> 正在加载最终能谱...[/white]")
            run.final_spectrum = _load_spectrum_data(particle_files[-1])
        else:
            console.print("  [yellow]  -> 只有一个粒子文件，最终能谱将与初始能谱相同。[/yellow]")
            run.final_spectrum = run.initial_spectrum

    if 'energy' in required_data:
        console.print("  [white]  -> 正在加载能量演化数据...[/white]")
        run.energy_data = _load_energy_evolution_data(dir_path, sim_obj)
        if run.energy_data is None:
            console.print("  [yellow]  -> 未能加载能量演化数据。[/yellow]")

    if 'field' in required_data:
        console.print("  [white]  -> 正在加载场演化数据...[/white]")
        run.field_data = _load_field_evolution_data(dir_path, sim_obj)
        if run.field_data is None:
            console.print("  [yellow]  -> 未能加载场演化数据。[/yellow]")

    # 视频模块可能只需要文件列表
    if 'field_files' in required_data:
        run.field_files = sorted(glob.glob(os.path.join(dir_path, "diags/fields", "fields_*.npz")))

    return run