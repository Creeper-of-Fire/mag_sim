# core/simulation.py

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# --- 核心数据结构模块 ---
#
# 定义 SimulationRun 和其他用于在分析模块之间传递数据的标准数据类。
#

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List
import numpy as np

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


# --- 核心的模拟运行数据容器 ---

@dataclass
class SimulationRun:
    """
    存放一次模拟运行的所有相关数据。
    这是一个合并后的综合版本，是框架中数据传递的核心。
    """
    path: str
    name: str
    sim: object  # 加载自 dill 的模拟参数对象

    # --- 按需加载的数据 ---
    # 粒子数据
    initial_spectrum: Optional[SpectrumData] = field(default=None)
    final_spectrum: Optional[SpectrumData] = field(default=None)
    user_T_keV: Optional[float] = field(default=None)  # 用于能谱分析

    # 场演化数据
    field_data: Optional[FieldEvolutionData] = field(default=None)

    # 能量演化数据
    energy_data: Optional[EnergyEvolutionData] = field(default=None)

    # 视频生成模块可能需要原始文件列表
    field_files: Optional[List[str]] = field(default=None)
    particle_files: Optional[List[str]] = field(default=None)

    @property
    def job_name(self) -> str:
        """
        根据 path 解析所属的 Job 名称。
        假设结构为: JobDir/sim_results/TaskDir 或 JobDir/TaskDir
        """
        p = Path(self.path)
        # 如果父目录是 sim_results，则 JobName 是再上一级
        if p.parent.name == 'sim_results':
            return p.parent.parent.name
        # 否则父目录就是 JobName
        return p.parent.name
    
    @property
    def job_path(self) -> Path:
        """
        根据 path 解析所属的 Job 的完整路径 (Path 对象)。
        """
        p = Path(self.path)
        # resolve() 可以将可能存在的 ".." 或 "." 符号解析为绝对路径
        if p.parent.name == 'sim_results':
            return p.parent.parent.resolve()
        return p.parent.resolve()