import typing
from dataclasses import fields
from typing import List, Optional

import numpy as np

from .simulation import SimulationRun

from .data_loader import FieldEvolutionData, EnergyEvolutionData, SpectrumData


class SimulationRunGroup(SimulationRun):
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
        self._sim_ref = runs[0].sim
        # 用于参数表和路径决策
        self.path = runs[0].path

    @property
    def sim(self) -> object:
        return self._sim_ref

    @property
    def energy_data(self) -> Optional['EnergyEvolutionData']:
        """
        返回平均后的能量演化数据。

        返回的对象在原有的 EnergyEvolutionData 字段（平均值）基础上，
        动态附加了 `_std` 后缀的字段（标准差）。
        """
        from .data_loader import EnergyEvolutionData

        # 收集所有有效的 energy_data
        all_data = [run.energy_data for run in self.runs if run.energy_data]
        if not all_data: return None

        # 假设时间轴一致，取第一个
        time_axis = all_data[0].time
        averaged_fields = {}

        # 动态遍历 dataclass 字段进行平均
        for field_info in fields(EnergyEvolutionData):
            field_name = field_info.name
            if field_name == 'time': continue

            # 收集该字段的数据数组
            field_arrays = []
            for d in all_data:
                val = getattr(d, field_name, None)
                if val is not None:
                    field_arrays.append(val)

            if not field_arrays: continue

            try:
                stacked = np.vstack(field_arrays)
                mean_arr = np.mean(stacked, axis=0)
                std_arr = np.std(stacked, axis=0)

                averaged_fields[field_name] = mean_arr
                averaged_fields[f"{field_name}_std"] = std_arr  # 动态附加标准差
            except Exception:
                # 如果形状不匹配（如某些 run 提前终止），则跳过该字段或后续处理
                continue

        # 创建基本对象
        result = EnergyEvolutionData(time=time_axis, **{k: v for k, v in averaged_fields.items() if k in [f.name for f in fields(EnergyEvolutionData)]})

        # 重新附加非标准字段 (_std)
        for k, v in averaged_fields.items():
            if k.endswith('_std'):
                setattr(result, k, v)

        return result

    @property
    def field_data(self) -> Optional['FieldEvolutionData']:
        """
        [暂未完全实现] 对 FieldEvolutionData 的平均。
        目前返回 None，直到明确 max/min 值的统计意义。
        """
        return None

    def get_spectrum(self, step_index: int = -1) -> Optional['SpectrumData']:
        """
        计算指定帧的平均能谱。

        逻辑：
        1. 收集组内所有 run 在该时刻的能谱。
        2. 使用 utils 生成公共分箱。
        3. 重新分箱并计算平均 dN/dE。
        """
        # --- Local Import 防止循环依赖 ---
        # comparison_utils 依赖 SimulationRun，而 SimulationRunGroup 定义在 SimulationRun 所在文件
        from .data_loader import SpectrumData

        try:
            # 1. 收集所有 run 在该时刻的能谱 (懒加载)
            # 注意：这里我们临时把 self.runs 视为一组独立的 run 来处理
            valid_runs = []
            valid_specs = []

            for run in self.runs:
                spec = run.get_spectrum(step_index)
                if spec is not None:
                    valid_runs.append(run)
                    valid_specs.append(spec)

            if not valid_specs: return None

            # 2. 如果只有一个有效数据，直接返回
            if len(valid_specs) == 1:
                return valid_specs[0]

            # 3. 创建公共分箱
            # create_common_energy_bins 需要传入 SimulationRun 对象列表来扫描范围
            # 但它默认看 initial/final，这里我们需要它针对当前帧做分箱？
            # 实际上 create_common_energy_bins 设计是针对全局对比的。
            # 为了更精确，这里我们手动针对 *当前帧* 做一个简单的并集分箱逻辑，
            # 或者复用 create_common_energy_bins 但只针对这一帧的数据。

            # 简化的本地逻辑：基于当前帧的能量范围
            all_energies = np.concatenate([s.energies_MeV for s in valid_specs])
            min_e, max_e = np.min(all_energies), np.max(all_energies)
            if min_e <= 0: min_e = 1e-4

            # 使用第一个谱的分箱数量作为参考
            n_bins = len(valid_specs[0].energies_MeV)
            bins = np.logspace(np.log10(min_e), np.log10(max_e), n_bins + 1)
            widths = np.diff(bins)
            centers = (bins[:-1] + bins[1:]) / 2

            # 4. 重新分箱并计算 dN/dE
            dNdE_list = []
            for spec in valid_specs:
                counts, _ = np.histogram(spec.energies_MeV, bins=bins, weights=spec.weights)
                dNdE_list.append(counts / widths)

            dNdE_stack = np.vstack(dNdE_list)
            mean_dNdE = np.mean(dNdE_stack, axis=0)
            std_dNdE = np.std(dNdE_stack, axis=0)

            # 5. 构建结果对象
            # 为了兼容性，weights = dN/dE * width
            avg_weights = mean_dNdE * widths

            res = SpectrumData(energies_MeV=centers, weights=avg_weights)
            # 附加统计信息供高级绘图使用
            res.mean_dNdE = mean_dNdE
            res.std_dNdE = std_dNdE
            res.energy_centers = centers

            return res

        except Exception as e:
            # 简单的错误打印，实际生产中应使用 logging
            print(f"[Warn] 计算组 {self.name} 的平均能谱失败: {e}")
            return None

    def get_field_slice(self, step_index: int = -1, axis: str = 'z') -> Optional[np.ndarray]:
        """
        获取场切片的平均值。
        前提：所有 run 的空间网格大小必须完全一致。
        """
        slices = []
        for run in self.runs:
            d = run.get_field_slice(step_index, axis)
            if d is not None:
                slices.append(d)

        if not slices: return None

        try:
            stack = np.array(slices)
            return np.mean(stack, axis=0)
        except ValueError:
            # 形状不匹配 (grid size 不同)
            return None
