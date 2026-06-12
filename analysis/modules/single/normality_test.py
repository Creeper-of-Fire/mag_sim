# analysis/modules/single/normality_test.py

import gc
from typing import List, Dict

import numpy as np
from scipy.stats import shapiro, jarque_bera, chi2
from tqdm import tqdm

from analysis.core.cache import cached_op
from analysis.core.data_loader import _get_step_from_filename
from analysis.core.simulation import SimulationRun
from analysis.core.utils import console
from analysis.modules.abstract.base_module import BaseAnalysisModule
from analysis.modules.utils.spectrum_tools import filter_valid_runs
from analysis.plotting.data_layout import DataLayout

_SW_MAX_N = 5000


@cached_op(file_dep="all")
def compute_normality_timeseries(run: SimulationRun) -> Dict[str, np.ndarray]:
    """
    遍历所有粒子文件，对每帧能谱执行 Jarque-Bera 和 Shapiro-Wilk 正态性检验。

    JB 使用加权偏度/峰度 + N_eff，无样本量限制。
    SW 需要随机子采样至 <=5000。
    """
    files = run.particle_files
    if not files:
        return {}

    console.print(f"  [cyan]计算正态性检验时序 (共 {len(files)} 帧)...[/cyan]")

    times = []
    jb_stats, jb_pvals = [], []
    sw_stats, sw_pvals = [], []

    for fpath in tqdm(files, desc="  正态性", unit="file", leave=False):
        spec = run.get_spectrum_from_path(fpath)
        if spec is None or spec.weights.size == 0:
            continue

        valid = spec.energies_MeV > 0
        E = spec.energies_MeV[valid]
        W = spec.weights[valid]

        if E.size < 3:
            continue

        step = _get_step_from_filename(fpath)
        time = step * run.sim.dt

        w_sum = np.sum(W)
        if w_sum <= 0:
            continue

        # 加权统计量
        mu = np.sum(W * E) / w_sum
        d = E - mu
        m2 = np.sum(W * d ** 2) / w_sum
        m3 = np.sum(W * d ** 3) / w_sum
        m4 = np.sum(W * d ** 4) / w_sum

        if m2 <= 0:
            continue

        S = m3 / m2 ** 1.5       # 加权偏度
        K = m4 / m2 ** 2         # 加权峰度（非超额）
        N_eff = w_sum ** 2 / np.sum(W ** 2)

        # Jarque-Bera（加权版）
        jb = (N_eff / 6.0) * (S ** 2 + (K - 3.0) ** 2 / 4.0)
        jb_p = float(chi2.sf(jb, df=2))

        # Shapiro-Wilk（随机子采样）
        if E.size > _SW_MAX_N:
            idx = np.random.choice(E.size, _SW_MAX_N, replace=False)
            sw_W, sw_p = shapiro(E[idx])
        else:
            sw_W, sw_p = shapiro(E)

        times.append(time)
        jb_stats.append(jb)
        jb_pvals.append(jb_p)
        sw_stats.append(sw_W)
        sw_pvals.append(sw_p)

        del spec
        gc.collect()

    if not times:
        return {}

    return {
        "time": np.array(times),
        "JB_stat": np.array(jb_stats),
        "JB_p": np.array(jb_pvals),
        "SW_W": np.array(sw_stats),
        "SW_p": np.array(sw_pvals),
    }


class NormalityTestModule(BaseAnalysisModule):
    @property
    def name(self) -> str:
        return "能谱正态性检验时序 (Jarque-Bera & Shapiro-Wilk)"

    @property
    def description(self) -> str:
        return (
            "对每个时间步的粒子能谱执行 Jarque-Bera 和 Shapiro-Wilk 正态性检验，"
            "绘制检验统计量和 P 值随时间的演化，导出 CSV。"
        )

    def run(self, loaded_runs: List[SimulationRun]):
        console.print("\n[bold magenta]执行: 能谱正态性检验时序...[/bold magenta]")
        valid_runs = filter_valid_runs(loaded_runs, require_particles=True, min_particle_files=2)
        if not valid_runs:
            console.print("[yellow]警告: 没有找到有效的粒子诊断文件，跳过此分析。[/yellow]")
            return

        for run in valid_runs:
            self._analyze(run)

    def _analyze(self, run: SimulationRun):
        console.print(f"\n[bold]分析模拟: {run.name}[/bold]")

        data = compute_normality_timeseries(run)
        if not data:
            console.print("[red]  错误: 未能生成正态性检验数据。[/red]")
            return

        t = data["time"]

        with DataLayout(run, "normality_test_timeseries", ncols=2, shared_xlabel="时间 (s)") as layout:
            ax_jb = layout.request_axes()
            ax_jb_p = layout.request_axes()
            ax_sw = layout.request_axes()
            ax_sw_p = layout.request_axes()

            # JB 统计量
            ax_jb.plot(t, data["JB_stat"], "o-", color="crimson", ms=3, label="JB 统计量")
            ax_jb.set_ylabel("Jarque-Bera 统计量")
            ax_jb.set_title("Jarque-Bera 检验统计量")
            ax_jb.set_yscale("log")
            ax_jb.legend()
            ax_jb.grid(True, alpha=0.3)

            # JB P 值
            safe_jb_p = np.clip(data["JB_p"], 1e-15, 1.0)
            ax_jb_p.plot(t, safe_jb_p, "s-", color="darkorange", ms=3, label="JB P 值")
            ax_jb_p.axhline(0.05, color="red", ls=":", lw=1.5, label=r"$\\alpha = 0.05$")
            ax_jb_p.set_ylabel("P 值")
            ax_jb_p.set_title("Jarque-Bera P 值")
            ax_jb_p.set_yscale("log")
            ax_jb_p.legend()
            ax_jb_p.grid(True, alpha=0.3)

            # SW W 统计量
            ax_sw.plot(t, data["SW_W"], "o-", color="royalblue", ms=3, label="SW W 统计量")
            ax_sw.axhline(1.0, color="gray", ls="--", lw=1, label="W=1 (完全正态)")
            ax_sw.set_ylabel("Shapiro-Wilk W")
            ax_sw.set_title("Shapiro-Wilk 检验统计量")
            ax_sw.set_ylim(0, 1.05)
            ax_sw.legend()
            ax_sw.grid(True, alpha=0.3)

            # SW P 值
            safe_sw_p = np.clip(data["SW_p"], 1e-15, 1.0)
            ax_sw_p.plot(t, safe_sw_p, "s-", color="seagreen", ms=3, label="SW P 值")
            ax_sw_p.axhline(0.05, color="red", ls=":", lw=1.5, label=r"$\\alpha = 0.05$")
            ax_sw_p.set_ylabel("P 值")
            ax_sw_p.set_title("Shapiro-Wilk P 值")
            ax_sw_p.set_yscale("log")
            ax_sw_p.legend()
            ax_sw_p.grid(True, alpha=0.3)

        console.print(f"  [green]✔ {run.name}:[/green]")
        console.print(f"    最终 JB: {data['JB_stat'][-1]:.2f} (P={data['JB_p'][-1]:.2e})")
        console.print(f"    最终 SW: W={data['SW_W'][-1]:.4f} (P={data['SW_p'][-1]:.2e})")
