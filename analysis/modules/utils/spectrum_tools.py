from typing import List

from analysis.core.simulation import SimulationRun


def filter_valid_runs(
        runs: List['SimulationRun'],
        require_particles: bool = False,
        min_particle_files: int = 2,
        require_fields: bool = False,
        min_field_files: int = 2
) -> List['SimulationRun']:
    """
    极速过滤无效的模拟 Run (不触发任何 HDF5 数据读取)。
    只通过检查底层文件索引的数量来判断模拟是否完整。
    """
    from analysis.core.simulationGroup import SimulationRunGroup
    from analysis.core.simulationSingle import SimulationRunSingle

    valid_runs = []
    for run in runs:
        is_valid = True

        if isinstance(run, SimulationRunGroup):
            # 如果是 Group，检查其内部的所有 run
            for single_run in run.runs:
                if require_particles and len(getattr(single_run, 'particle_files', [])) < min_particle_files:
                    is_valid = False
                    break
                if require_fields and len(getattr(single_run, 'field_files', [])) < min_field_files:
                    is_valid = False
                    break

        elif isinstance(run, SimulationRunSingle):
            # 如果是单个 Run
            if require_particles and len(run.particle_files) < min_particle_files:
                is_valid = False
            if require_fields and len(run.field_files) < min_field_files:
                is_valid = False

        else:
            # 防御性编程：对于未知类型的 Run，保守跳过或直接放行
            pass

        if is_valid:
            valid_runs.append(run)

    return valid_runs
