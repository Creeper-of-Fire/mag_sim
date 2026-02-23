# core/simulation.py

# !/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# --- 核心数据结构模块 ---
#
# 定义 SimulationRun 和其他用于在分析模块之间传递数据的标准数据类。
#

import typing
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import numpy as np

from .data_loader import FieldEvolutionData, EnergyEvolutionData, SpectrumData


# --- 核心的模拟运行数据容器 ---

# --- 抽象基类 ---

class SimulationRun(ABC):
    """
    模拟运行的抽象基类。
    定义了单次运行(Single)和运行组(Group)必须实现的通用接口。
    """

    name: str
    sim: object  # sim_parameters 对象
    path: str  # 物理路径或逻辑路径

    @property
    def job_name(self) -> str:
        """
        根据 path 解析所属的 Job 名称。
        """
        p = Path(self.path)
        # 如果父目录是 sim_results，则 JobName 是再上一级
        if p.parent.name == 'sim_results':
            return p.parent.parent.name
        return p.parent.name

    @property
    def job_path(self) -> Path:
        """
        根据 path 解析所属的 Job 的完整路径。
        """
        p = Path(self.path)
        if p.parent.name == 'sim_results':
            return p.parent.parent.resolve()
        return p.parent.resolve()

    # --- 必须实现的抽象接口 ---

    @property
    @abstractmethod
    def energy_data(self) -> Optional['EnergyEvolutionData']:
        """获取能量演化数据 (可能包含 _std 统计信息)"""
        pass

    @property
    @abstractmethod
    def field_data(self) -> Optional['FieldEvolutionData']:
        """获取场演化数据"""
        pass

    @abstractmethod
    def get_spectrum(self, step_index: int = -1) -> Optional['SpectrumData']:
        """获取特定时间步的能谱"""
        pass

    @abstractmethod
    def get_field_slice(self, step_index: int = -1, axis: str = 'z') -> Optional[np.ndarray]:
        """获取特定时间步的场切片"""
        pass

    # --- 兼容性属性 ---

    @property
    def initial_spectrum(self):
        return self.get_spectrum(0)

    @property
    def final_spectrum(self):
        return self.get_spectrum(-1)



