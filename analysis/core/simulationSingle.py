import glob
import os
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np

from analysis.core.cache import SmartCache, cached_op
from analysis.core.simulation import SimulationRun

from analysis.core.data_loader import EnergyEvolutionData, FieldEvolutionData, SpectrumData

@dataclass
class SimulationRunSingle(SimulationRun):
    """
    SimulationRun 现在是一个智能的数据访问门面 (Facade)。
    它负责管理文件索引和缓存，按需调用 loader。
    """
    path: str
    name: str
    sim: object  # sim_parameters 对象

    # 内部组件
    _cache: SmartCache = field(init=False, repr=False)

    # 文件索引 (初始化时扫描)
    _particle_files: List[str] = field(default_factory=list, repr=False)
    _field_files: List[str] = field(default_factory=list, repr=False)
    _param_file: str = field(init=False, repr=False)

    # 运行时状态
    # TODO 这不是个好设计，之后应该会去掉
    user_T_keV: Optional[float] = field(default=None)

    def __post_init__(self):
        """初始化后自动建立索引和缓存管理器"""
        self.path = os.path.abspath(self.path)

        # 独立的缓存目录，避免污染源目录太多文件
        cache_dir = Path(self.path) / ".analysis_v2_cache"
        self._cache = SmartCache(cache_dir)

        # 1. 定位参数文件
        self._param_file = os.path.join(self.path, "sim_parameters.dpkl")

        # 2. 建立文件索引 (glob非常快)
        # 确保排序，因为文件列表的顺序会影响缓存指纹
        self._particle_files = sorted(glob.glob(os.path.join(self.path, "diags/particle_states", "openpmd_*.h5")))
        self._field_files = sorted(glob.glob(os.path.join(self.path, "diags/field_states", "*.h5")))

    # --- 基础属性 ---

    @property
    def particle_files(self) -> List[str]:
        return self._particle_files

    @property
    def field_files(self) -> List[str]:
        return self._field_files

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

    # --- 核心数据访问 (Lazy Loading) ---

    @property
    @cached_op(file_dep="all")
    def energy_data(self) -> Optional['EnergyEvolutionData']:
        """获取能量演化数据 (Cached)。"""
        from .data_loader import compute_energy_evolution

        return compute_energy_evolution(self._field_files, self._particle_files, sim_obj=self.sim)

    @property
    @cached_op(file_dep="field")
    def field_data(self) -> Optional['FieldEvolutionData']:
        """
        获取场演化数据 (Cached)。
        """
        from .data_loader import compute_field_evolution

        return compute_field_evolution(field_files=self._field_files, sim_obj=self.sim)

    # 原始粒子读取非常快（HDF5自带切片能力），不需要缓存，且占用内存巨大。
    def get_spectrum_from_path(self, fpath: str) -> Optional['SpectrumData']:
        """
        这个方法是“自动导航”的：
        装饰器识别到参数中的 fpath，会自动将其作为单文件依赖。
        """
        from .data_loader import compute_single_spectrum
        return compute_single_spectrum(fpath)

    def get_spectrum(self, step_index: int = -1) -> Optional['SpectrumData']:
        """
        获取特定时间步的能谱 (Cached)。

        Args:
            step_index: 帧索引。-1 表示最后一帧，0 表示第一帧。
        """

        # --- 索引解析逻辑 (只在 SimulationRun 内部知道) ---
        files = self._particle_files
        idx = step_index if step_index >= 0 else len(files) + step_index

        if not (0 <= idx < len(files)):
            return None

        target_file = files[idx]

        # --- 调用缓存层 ---
        # 此时传给装饰器的是具体的文件路径
        return self.get_spectrum_from_path(target_file)

    @cached_op(file_dep="particle")
    def get_spectrum_evolution_matrix(self, n_bins: int = 200, log_scale: bool = True):
        """
        获取能谱随时间演化的矩阵 (Waterfall data)。
        """
        from .data_loader import compute_spectrum_evolution_matrix

        return compute_spectrum_evolution_matrix(self._particle_files, self.sim, n_bins, log_scale)

    @cached_op(file_dep="auto")
    def get_field_slice_from_path(self, fpath: str, axis: str = 'z') -> Optional[np.ndarray]:
        """
        单帧场切片读取 (Cached)。
        """
        from .data_loader import read_field_slice
        # 读取原始数据
        data = read_field_slice(fpath, axis=axis)
        # 在这里做归一化，保证出来的中间变量是物理上有意义的数值或者归一化数值
        if data is not None and hasattr(self.sim, 'B_norm') and self.sim.B_norm > 0:
            return data / self.sim.B_norm
        return data

    def get_field_slice(self, step_index: int = -1, axis: str = 'z') -> Optional[np.ndarray]:
        """
        获取特定时间步的磁场强度切片 (Cached Intermediate Variable)。

        Returns:
            np.ndarray: 2D array of |B| / B_norm
        """
        files = self._field_files
        if not files: return None
        idx = step_index if step_index >= 0 else len(files) + step_index
        if not (0 <= idx < len(files)):
            return None

        return self.get_field_slice_from_path(files[idx], axis=axis)