import typing
from dataclasses import fields
from typing import List, Optional

import numpy as np

from analysis.core.utils import console
from analysis.modules.utils.comparison_utils import create_common_energy_bins
from .simulation import SimulationRun

if typing.TYPE_CHECKING:
    from .data_loader import FieldEvolutionData, EnergyEvolutionData, SpectrumData

class SimulationRunGroup:
    """
    代表一组统计性重复的 SimulationRun。
    负责对其数据进行平均和统计。
    """

    def __init__(self, runs: List[SimulationRun]):
        if not runs:
            raise ValueError("SimulationRunGroup 必须包含至少一个 run。")
        self.runs: List[SimulationRun] = runs
        self.name = f"{runs[0].name}_group(N={len(runs)})"
        # 假设所有 runs 的 sim 对象参数一致
        self.sim = runs[0].sim
        # 用于参数表和路径决策
        self.job_path = runs[0].job_path

    @property
    def final_spectrum(self) -> Optional[SpectrumData]:
        """返回平均后的最终能谱，包含均值和标准差。"""
        try:
            # 1. 创建公共分箱
            bins, centers, widths = create_common_energy_bins(self.runs)

            # 2. 计算每个run在公共分箱上的 dN/dE
            dNdE_list = []
            for run in self.runs:
                spec = run.final_spectrum
                if spec is None: continue
                counts, _ = np.histogram(spec.energies_MeV, bins=bins, weights=spec.weights)
                dNdE_list.append(counts / widths)

            if not dNdE_list: return None

            # 3. 计算均值和标准差
            dNdE_stack = np.vstack(dNdE_list)
            mean_dNdE = np.mean(dNdE_stack, axis=0)
            std_dNdE = np.std(dNdE_stack, axis=0)

            # 4. 返回一个特殊的数据结构，或者直接在 SpectrumData 中增加字段
            #    为了简单起见，我们可以在返回的 SpectrumData 对象上动态附加属性

            # 我们用均值反算出等效的 weights 来构建一个 SpectrumData 对象
            # 注意：这里的 energies_MeV 和 weights 仅用于结构兼容，真正有用的是附加的属性
            avg_spec = SpectrumData(energies_MeV=centers, weights=mean_dNdE * widths)
            avg_spec.mean_dNdE = mean_dNdE
            avg_spec.std_dNdE = std_dNdE
            avg_spec.energy_centers = centers  # 绘图时需要
            return avg_spec

        except Exception as e:
            console.print(f"[red]为 {self.name} 计算平均能谱失败: {e}[/red]")
            return None

    @property
    def energy_data(self) -> Optional[EnergyEvolutionData]:
        """返回平均后的能量演化数据。"""
        all_data = [run.energy_data for run in self.runs if run.energy_data]
        if not all_data: return None

        # 假设所有模拟的时间步完全一致
        time_axis = all_data[0].time

        # 对 EnergyEvolutionData 的每个字段进行平均
        averaged_fields = {}
        for field_info in fields(EnergyEvolutionData):
            field_name = field_info.name
            if field_name == 'time':
                continue

            # 收集所有 runs 的该字段数据
            field_arrays = [getattr(data, field_name) for data in all_data if hasattr(data, field_name) and getattr(data, field_name) is not None]
            if not field_arrays: continue

            # 堆叠并计算均值和标准差
            stacked_arrays = np.vstack(field_arrays)
            mean_array = np.mean(stacked_arrays, axis=0)
            std_array = np.std(stacked_arrays, axis=0)

            # 存入字典
            averaged_fields[field_name] = mean_array
            averaged_fields[f"{field_name}_std"] = std_array  # 附加标准差

        return EnergyEvolutionData(time=time_axis, **averaged_fields)

    # 为其他数据属性（如 field_data）实现类似的平均逻辑 ...