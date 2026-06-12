# analysis/modules/single/energy_moments_timeseries.py

import gc
from typing import List, Dict

import numpy as np
from tqdm import tqdm

from analysis.core.async_utils import asyncify
from analysis.core.cache import cached_op
from analysis.core.data_loader import _get_step_from_filename
from analysis.core.simulation import SimulationRun
from analysis.core.utils import console
from analysis.modules.abstract.base_module import BaseAnalysisModule
from analysis.modules.utils.spectrum_tools import filter_valid_runs
from analysis.physics.moments import compute_run_moments
from analysis.plotting.data_layout import DataLayout


@cached_op(file_dep="all")
def compute_moments_timeseries(run: SimulationRun) -> Dict[str, np.ndarray]:
    """遍历所有粒子文件，计算每一帧的能谱偏度、峰度和中心矩。"""
    files = run.particle_files
    if not files:
        return {}

    console.print(f"  [cyan]计算能谱高阶矩时序 (共 {len(files)} 帧)...[/cyan]")

    times, skewness, kurtosis, m3_vals, m4_vals = [], [], [], [], []

    async_compute = asyncify(compute_run_moments)

    for fpath in tqdm(files, desc="  能谱矩", unit="file", leave=False):
        result = compute_run_moments(run, fpath=fpath)
        step = _get_step_from_filename(fpath)
        time = step * run.sim.dt

        times.append(time)
        skewness.append(result.skewness)
        kurtosis.append(result.kurtosis)
        m3_vals.append(result.moment_3)
        m4_vals.append(result.moment_4)

        del result
        gc.collect()

    if not times:
        return {}

    return {
        "time": np.array(times),
        "skewness": np.array(skewness),
        "kurtosis": np.array(kurtosis),
        "moment_3": np.array(m3_vals),
        "moment_4": np.array(m4_vals),
    }


class EnergyMomentsTimeSeriesModule(BaseAnalysisModule):
    @property
    def name(self) -> str:
        return "能谱高阶矩时序分析"

    @property
    def description(self) -> str:
        return (
            "计算每个时间步的能谱偏度、峰度(超额)和 3/4 阶中心矩，"
            "绘制时序演化图并导出 CSV。"
        )

    def run(self, loaded_runs: List[SimulationRun]):
        console.print("\n[bold magenta]执行: 能谱高阶矩时序分析...[/bold magenta]")
        valid_runs = filter_valid_runs(loaded_runs, require_particles=True, min_particle_files=2)
        if not valid_runs:
            console.print("[yellow]警告: 没有找到有效的粒子诊断文件，跳过此分析。[/yellow]")
            return

        for run in valid_runs:
            self._analyze(run)

    def _analyze(self, run: SimulationRun):
        console.print(f"\n[bold]分析模拟: {run.name}[/bold]")

        data = compute_moments_timeseries(run)
        if not data:
            console.print("[red]  错误: 未能生成能谱矩时序数据。[/red]")
            return

        t = data["time"]

        with DataLayout(run, "energy_moments_timeseries", ncols=2, shared_xlabel="时间 (s)") as layout:
            ax_sk = layout.request_axes()
            ax_kt = layout.request_axes()
            ax_m3 = layout.request_axes()
            ax_m4 = layout.request_axes()

            # 偏度
            ax_sk.plot(t, data["skewness"], "o-", color="darkorange", ms=3, label="偏度 (skewness)")
            ax_sk.axhline(0, color="gray", ls="--", lw=1, label="Gaussian 参考线")
            ax_sk.set_ylabel("偏度 $\\gamma_1$")
            ax_sk.set_title("能谱偏度演化 (Skewness)")
            ax_sk.legend()
            ax_sk.grid(True, alpha=0.3)

            # 超额峰度
            ax_kt.plot(t, data["kurtosis"], "s-", color="crimson", ms=3, label="超额峰度 (excess kurtosis)")
            ax_kt.axhline(0, color="gray", ls="--", lw=1, label="Gaussian 参考线")
            ax_kt.set_ylabel("超额峰度 $\\gamma_2$")
            ax_kt.set_title("能谱峰度演化 (Excess Kurtosis)")
            ax_kt.legend()
            ax_kt.grid(True, alpha=0.3)

            # 3阶中心矩
            ax_m3.plot(t, data["moment_3"], "o-", color="royalblue", ms=3, label="$\\mu_3$")
            ax_m3.axhline(0, color="gray", ls="--", lw=1)
            ax_m3.set_ylabel("$\\mu_3$ (MeV$^3$)")
            ax_m3.set_title("3 阶中心矩演化")
            ax_m3.legend()
            ax_m3.grid(True, alpha=0.3)

            # 4阶中心矩
            ax_m4.plot(t, data["moment_4"], "s-", color="seagreen", ms=3, label="$\\mu_4$")
            ax_m4.axhline(0, color="gray", ls="--", lw=1)
            ax_m4.set_ylabel("$\\mu_4$ (MeV$^4$)")
            ax_m4.set_title("4 阶中心矩演化")
            ax_m4.legend()
            ax_m4.grid(True, alpha=0.3)

        console.print(f"  [green]✔ {run.name}: 偏度 {data['skewness'][-1]:.3f}, "
                       f"峰度 {data['kurtosis'][-1]:.3f}[/green]")
