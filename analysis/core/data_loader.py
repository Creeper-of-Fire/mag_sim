# core/data_loader.py

# !/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# --- 统一数据加载模块 ---
#
# 包含所有从磁盘文件读取和计算物理量的函数。
# 提供一个高级接口 `load_run_data` 来按需加载数据。
#
import os
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Set, Optional, List, Any, Tuple

import dill
import h5py
import numpy as np
from scipy.constants import c, m_e, mu_0, e, epsilon_0
from tqdm import tqdm

from .simulation import SimulationRun
from .utils import console

# --- 各种分析所需的数据容器 ---

@dataclass
class FieldEvolutionData:
    """存放磁场演化数据"""
    time: np.ndarray
    b_mean_abs_normalized: np.ndarray
    b_max_normalized: np.ndarray
    b_mean_x_normalized: np.ndarray
    b_mean_y_normalized: np.ndarray
    b_mean_z_normalized: np.ndarray
    b_rms_x_normalized: np.ndarray
    b_rms_y_normalized: np.ndarray
    b_rms_z_normalized: np.ndarray


@dataclass
class EnergyEvolutionData:
    """存储能量随时间演化的数据"""
    time: np.ndarray

    # 平均磁场能量密度 (J/m^3)
    mean_mag_energy_density_x: Optional[np.ndarray] = field(default=None)
    mean_mag_energy_density_y: Optional[np.ndarray] = field(default=None)
    mean_mag_energy_density_z: Optional[np.ndarray] = field(default=None)
    mean_mag_energy_density_total: Optional[np.ndarray] = field(default=None)

    # 平均电场能量密度 (J/m^3)
    mean_elec_energy_density_x: Optional[np.ndarray] = field(default=None)
    mean_elec_energy_density_y: Optional[np.ndarray] = field(default=None)
    mean_elec_energy_density_z: Optional[np.ndarray] = field(default=None)
    mean_elec_energy_density_total: Optional[np.ndarray] = field(default=None)

    # 平均动能密度 (J/m^3)
    mean_kin_energy_density: Optional[np.ndarray] = field(default=None)

    # 盒子内的总能量 (J)
    total_magnetic_energy: Optional[np.ndarray] = field(default=None)
    total_electric_energy: Optional[np.ndarray] = field(default=None)
    total_kinetic_energy: Optional[np.ndarray] = field(default=None)


@dataclass
class SpectrumData:
    """存放能谱数据"""
    energies_MeV: np.ndarray
    weights: np.ndarray


# =============================================================================
# 1. 底层计算和辅助函数
# =============================================================================

def _get_h5_dataset(h5_item: h5py.Group, component_path: str) -> np.ndarray:
    """
    稳健地读取 HDF5 数据集，处理扁平、嵌套和单成员组等多种结构。
    """
    item = h5_item[component_path]
    if isinstance(item, h5py.Dataset):
        # 情况1：路径直接指向一个数据集 (最简单的情况)
        return item[:]
    elif isinstance(item, h5py.Group):
        # 情况2：路径指向一个组 (需要进一步探查)

        # 如果组是空的，说明这个物种在该时刻没有粒子，返回空数组
        if not item:  # 在 h5py 中，空组的布尔值为 False
            return np.array([])

        if len(item) <= 0:
            return np.array([])

        # 稳健策略：如果组内只有一个成员，且是数据集，则假定它就是目标数据。
        # 这能处理 '.../momentum/x' (Group) -> 'momentum/x' (Dataset) 的情况。
        if len(item) == 1:
            first_key = list(item.keys())[0]
            if isinstance(item[first_key], h5py.Dataset):
                return item[first_key][:]

        # 后备策略：尝试使用路径的最后一部分作为数据集名称 (例如 'weighting' -> 'weighting')
        dataset_name = component_path.split('/')[-1]
        if dataset_name in item and isinstance(item[dataset_name], h5py.Dataset):
            return item[dataset_name][:]

        # 如果以上策略都失败，抛出明确的错误
        raise KeyError(f"在组 '{item.name}' 中无法找到预期的数据集。组内成员: {list(item.keys())}")
    else:
        raise TypeError(f"HDF5 对象 '{item.name}' 的类型无法识别。")


def _get_step_from_filename(filename: str) -> Optional[int]:
    """从 WarpX 诊断文件名中稳健地提取步数。"""
    try:
        base_name = os.path.basename(filename)
        step_str = base_name.split('_')[-1].split('.')[0]
        return int(step_str)
    except (IndexError, ValueError):
        return None


# =============================================================================
# 2. 纯计算函数 (Pure Compute Functions)
# =============================================================================

def read_field_slice(fpath: str, axis: str = 'z', slice_idx: Optional[int] = None) -> Optional[np.ndarray]:
    """
    [纯函数] 从 HDF5 文件中读取磁场强度 |B| 的切片。
    这是典型的“中间变量”提取。

    Args:
        fpath: HDF5 文件路径
        axis: 切片法向 ('x', 'y', 'z')
        slice_idx: 切片索引。如果为 None，自动取中心。
    """
    try:
        step = _get_step_from_filename(fpath)
        with h5py.File(fpath, 'r') as f:
            base_path = f"/data/{step}/fields/"
            # 尝试读取
            if f"{base_path}B/x" not in f:
                return None

            # 读取 Bx, By, Bz
            # 注意：WarpX 数据通常是 (x, y, z) 或 (x, z)
            # 但 h5py 读取出来通常是 (Nx, Ny, Nz) 的 F-order 或者 C-order，取决于具体实现
            # 这里假设读取出来是 Numpy 默认顺序
            Bx_ds = f[base_path + 'B/x']
            By_ds = f[base_path + 'B/y']
            Bz_ds = f[base_path + 'B/z']

            shape = Bx_ds.shape
            ndim = len(shape)

            # 2D 模拟处理
            if ndim == 2:
                # 2D 模拟通常是 X-Z 平面，数据本身就是切片
                Bx = Bx_ds[:]
                By = By_ds[:]
                Bz = Bz_ds[:]
                return np.sqrt(Bx ** 2 + By ** 2 + Bz ** 2)

            # 3D 模拟处理
            if ndim == 3:
                # 确定切片索引
                target_axis_idx = {'x': 0, 'y': 1, 'z': 2}.get(axis, 2)

                if slice_idx is None:
                    slice_idx = shape[target_axis_idx] // 2

                # 使用 numpy 的切片语法只读取需要的数据层，减少 I/O
                slicer = [slice(None)] * 3
                slicer[target_axis_idx] = slice_idx
                slicer = tuple(slicer)

                Bx = Bx_ds[slicer]
                By = By_ds[slicer]
                Bz = Bz_ds[slicer]

                return np.sqrt(Bx ** 2 + By ** 2 + Bz ** 2)

    except Exception as e:
        console.print(f"[yellow]⚠ 读取场切片失败 {os.path.basename(fpath)}: {e}[/yellow]")
        return None

    return None

def compute_single_spectrum(h5_filepath: str) -> Optional[SpectrumData]:
    """
    [纯函数] 读取单个 HDF5 文件并返回 SpectrumData。
    """
    all_energies_MeV, all_weights = [], []
    m_e_c2_J = m_e * c ** 2
    J_PER_MEV = e * 1e6

    try:
        with h5py.File(h5_filepath, 'r') as f:
            # WarpX / OpenPMD 标准路径
            if 'data' not in f: return None
            step_key = list(f['data'].keys())[0]
            particles_group = f[f'data/{step_key}/particles']

            for species_name in particles_group.keys():
                if 'photon' in species_name: continue  # 跳过光子

                species_group = particles_group[species_name]
                try:
                    px = _get_h5_dataset(species_group, 'momentum/x')
                    py = _get_h5_dataset(species_group, 'momentum/y')
                    pz = _get_h5_dataset(species_group, 'momentum/z')
                    weights = _get_h5_dataset(species_group, 'weighting')
                except KeyError:
                    continue

                if weights.size == 0: continue

                p_sq = px ** 2 + py ** 2 + pz ** 2
                kinetic_energy_J = np.sqrt(p_sq * c ** 2 + m_e_c2_J ** 2) - m_e_c2_J
                all_energies_MeV.append(kinetic_energy_J / J_PER_MEV)
                all_weights.append(weights)

        if not all_energies_MeV: return None
        return SpectrumData(np.concatenate(all_energies_MeV), np.concatenate(all_weights))

    except Exception as err:
        console.print(f"[yellow]⚠ 读取能谱失败 {os.path.basename(h5_filepath)}: {err}[/yellow]")
        return None


def compute_field_evolution(field_files: List[str], sim_obj: Any) -> Optional[FieldEvolutionData]:
    """
    [纯函数] 遍历场文件列表，计算场演化统计量。
    """
    if not field_files: return None

    times, b_max_vals = [], []
    b_mean_x, b_mean_y, b_mean_z = [], [], []
    b_rms_x, b_rms_y, b_rms_z = [], [], []
    b_mean_abs = []

    # 简单抽样以显示进度条
    for fpath in tqdm(field_files, desc="  计算场演化", unit="file", leave=False):
        step = _get_step_from_filename(fpath)
        if step is None: continue

        try:
            with h5py.File(fpath, 'r') as f:
                base_path = f"/data/{step}/fields/"
                # 直接读取
                Bx = f[base_path + 'B/x'][:]
                By = f[base_path + 'B/y'][:]
                Bz = f[base_path + 'B/z'][:]

            # 归一化
            Bx /= sim_obj.B_norm
            By /= sim_obj.B_norm
            Bz /= sim_obj.B_norm

            b_mean_x.append(np.mean(Bx))
            b_mean_y.append(np.mean(By))
            b_mean_z.append(np.mean(Bz))
            b_rms_x.append(np.sqrt(np.mean(Bx ** 2)))
            b_rms_y.append(np.sqrt(np.mean(By ** 2)))
            b_rms_z.append(np.sqrt(np.mean(Bz ** 2)))

            b_mag = np.sqrt(Bx ** 2 + By ** 2 + Bz ** 2)
            b_max_vals.append(np.max(b_mag))
            b_mean_abs.append(np.mean(b_mag))

            times.append(step * sim_obj.dt)
        except Exception:
            continue

    if not times: return None

    return FieldEvolutionData(
        time=np.array(times),
        b_max_normalized=np.array(b_max_vals),
        b_mean_x_normalized=np.array(b_mean_x),
        b_mean_y_normalized=np.array(b_mean_y),
        b_mean_z_normalized=np.array(b_mean_z),
        b_mean_abs_normalized=np.array(b_mean_abs),
        b_rms_x_normalized=np.array(b_rms_x),
        b_rms_y_normalized=np.array(b_rms_y),
        b_rms_z_normalized=np.array(b_rms_z)
    )


def compute_energy_evolution(field_files: List[str], particle_files: List[str], sim_obj: Any) -> Optional[EnergyEvolutionData]:
    """
    [纯函数] 综合计算场能和动能的演化。
    需要对齐场文件和粒子文件的时间步。
    """
    # 建立 {step: filepath} 映射
    f_map = {s: f for f in field_files if (s := _get_step_from_filename(f)) is not None}
    p_map = {s: f for f in particle_files if (s := _get_step_from_filename(f)) is not None}

    # 找交集
    common_steps = sorted(list(set(f_map.keys()) & set(p_map.keys())))
    if not common_steps:
        console.print("[yellow]⚠ 未找到时间步匹配的场文件和粒子文件，能量计算跳过。[/yellow]")
        return None

    times = []
    # 数据容器
    res = {k: [] for k in [
        'mag_x', 'mag_y', 'mag_z', 'mag_tot',
        'elec_x', 'elec_y', 'elec_z', 'elec_tot',
        'kin_dens', 'tot_mag', 'tot_elec', 'tot_kin'
    ]}

    # 计算体积元
    cell_vol = (sim_obj.Lx / sim_obj.NX) * (sim_obj.Ly / sim_obj.NY) * (sim_obj.Lz / sim_obj.NZ)
    sim_vol = cell_vol * sim_obj.NX * sim_obj.NY * sim_obj.NZ
    m_e_c2_J = m_e * c ** 2

    for step in tqdm(common_steps, desc="  计算能量演化", unit="step", leave=False):
        times.append(step * sim_obj.dt)

        # --- 1. 场能 ---
        try:
            with h5py.File(f_map[step], 'r') as f:
                bp = f"/data/{step}/fields/"
                Bx = f[bp + 'B/x'][:]
                By = f[bp + 'B/y'][:]
                Bz = f[bp + 'B/z'][:]
                Ex = f[bp + 'E/x'][:]
                Ey = f[bp + 'E/y'][:]
                Ez = f[bp + 'E/z'][:]

            # 磁能密度 (J/m^3)
            md_x, md_y, md_z = (Bx ** 2) / (2 * mu_0), (By ** 2) / (2 * mu_0), (Bz ** 2) / (2 * mu_0)
            res['mag_x'].append(np.mean(md_x))
            res['mag_y'].append(np.mean(md_y))
            res['mag_z'].append(np.mean(md_z))
            res['mag_tot'].append(np.mean(md_x + md_y + md_z))
            res['tot_mag'].append(np.sum(md_x + md_y + md_z) * cell_vol)

            # 电能密度
            ed_x, ed_y, ed_z = (epsilon_0 * Ex ** 2) / 2, (epsilon_0 * Ey ** 2) / 2, (epsilon_0 * Ez ** 2) / 2
            res['elec_x'].append(np.mean(ed_x))
            res['elec_y'].append(np.mean(ed_y))
            res['elec_z'].append(np.mean(ed_z))
            res['elec_tot'].append(np.mean(ed_x + ed_y + ed_z))
            res['tot_elec'].append(np.sum(ed_x + ed_y + ed_z) * cell_vol)

        except Exception as e:
            console.print(f"[red]Error reading fields at step {step}: {e}[/red]")
            return None

        # --- 2. 动能 ---
        try:
            tot_k_E = 0.0
            with h5py.File(p_map[step], 'r') as f:
                sk = list(f['data'].keys())[0]
                pg = f[f'data/{sk}/particles']
                for sp in pg.keys():
                    if 'photon' in sp: continue
                    sg = pg[sp]
                    try:
                        px = _get_h5_dataset(sg, 'momentum/x')
                        py = _get_h5_dataset(sg, 'momentum/y')
                        pz = _get_h5_dataset(sg, 'momentum/z')
                        w = _get_h5_dataset(sg, 'weighting')
                    except KeyError:
                        continue

                    if w.size == 0: continue
                    p2 = px ** 2 + py ** 2 + pz ** 2
                    k_E = np.sqrt(p2 * c ** 2 + m_e_c2_J ** 2) - m_e_c2_J
                    tot_k_E += np.sum(k_E * w)

            res['tot_kin'].append(tot_k_E)
            res['kin_dens'].append(tot_k_E / sim_vol if sim_vol > 0 else 0)

        except Exception as e:
            console.print(f"[red]Error reading particles at step {step}: {e}[/red]")
            return None

    return EnergyEvolutionData(
        time=np.array(times),
        mean_mag_energy_density_x=np.array(res['mag_x']),
        mean_mag_energy_density_y=np.array(res['mag_y']),
        mean_mag_energy_density_z=np.array(res['mag_z']),
        mean_mag_energy_density_total=np.array(res['mag_tot']),
        mean_elec_energy_density_x=np.array(res['elec_x']),
        mean_elec_energy_density_y=np.array(res['elec_y']),
        mean_elec_energy_density_z=np.array(res['elec_z']),
        mean_elec_energy_density_total=np.array(res['elec_tot']),
        mean_kin_energy_density=np.array(res['kin_dens']),
        total_magnetic_energy=np.array(res['tot_mag']),
        total_electric_energy=np.array(res['tot_elec']),
        total_kinetic_energy=np.array(res['tot_kin'])
    )


def compute_spectrum_evolution_matrix(
        particle_files: List[str],
        sim_obj: Any,
        n_bins: int = 200,
        log_scale: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    [纯函数] 计算能谱随时间的演化矩阵 (Waterfall Data)。

    Returns:
        (times, energy_bin_centers, matrix_2d)
        matrix_2d shape: (n_time_steps, n_bins)
    """
    if not particle_files:
        return np.array([]), np.array([]), np.array([])

    # 1. 预扫描: 确定全局能量范围
    # 为了速度，只采样首尾和中间
    sample_indices = np.unique(np.linspace(0, len(particle_files) - 1, min(10, len(particle_files))).astype(int))
    sampled_files = [particle_files[i] for i in sample_indices]

    global_min, global_max = 1e9, -1e9

    for fpath in sampled_files:
        spec = compute_single_spectrum(fpath)
        if spec and spec.energies_MeV.size > 0:
            pos_E = spec.energies_MeV[spec.energies_MeV > 0]
            if pos_E.size > 0:
                global_min = min(global_min, pos_E.min())
                global_max = max(global_max, pos_E.max())

    if global_max < 0:  # 没找到有效数据
        return np.array([]), np.array([]), np.array([])

    global_min = max(global_min * 0.9, 1e-4)
    global_max = global_max * 1.2

    # 2. 创建 Bins
    if log_scale:
        bins = np.logspace(np.log10(global_min), np.log10(global_max), n_bins + 1)
    else:
        bins = np.linspace(global_min, global_max, n_bins + 1)

    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    bin_widths = np.diff(bins)

    times = []
    matrix_rows = []

    # 3. 遍历所有文件计算直方图
    for fpath in tqdm(particle_files, desc="  计算能谱矩阵", unit="step", leave=False):
        step = _get_step_from_filename(fpath)
        if step is None: continue

        spec = compute_single_spectrum(fpath)

        if spec and spec.energies_MeV.size > 0:
            counts, _ = np.histogram(spec.energies_MeV, bins=bins, weights=spec.weights)
            # 归一化为 dN/dE
            dNdE = counts / bin_widths
        else:
            dNdE = np.zeros(n_bins)

        matrix_rows.append(dNdE)
        times.append(step * sim_obj.dt)

    return np.array(times), bin_centers, np.array(matrix_rows)


# =============================================================================
# 3. 高级统一加载接口
# =============================================================================

def load_run_data(dir_path: str, required_data: Set[str] = None) -> Optional[SimulationRun]:
    """
    [工厂函数] 为单个模拟目录创建一个 SimulationRun 实例。
    """
    console.print(f"\n[bold cyan]正在初始化模拟: {os.path.basename(dir_path)}[/bold cyan]")

    param_file = os.path.join(dir_path, "sim_parameters.dpkl")
    if not os.path.exists(param_file):
        console.print(f"  [red]✗ 错误: 找不到参数文件 '{param_file}'。[/red]")
        return None

    try:
        with open(param_file, "rb") as f:
            sim_obj = SimpleNamespace(**dill.load(f))

        # 创建实例，它会自动建立索引
        run = SimulationRun(path=dir_path, name=os.path.basename(dir_path), sim=sim_obj)
        console.print("  [green]✔ 索引建立完成。[/green]")
        return run

    except Exception as e:
        console.print(f"  [red]✗ 加载失败: {e}[/red]")
        return None
