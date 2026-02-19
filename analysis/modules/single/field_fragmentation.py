# analysis/modules/field_fragmentation.py

import os
from typing import List, Tuple, Dict, Any

import numpy as np
from scipy.stats import kurtosis
from tqdm import tqdm

from analysis.core.simulation import SimulationRun
from analysis.core.utils import console
from analysis.modules.abstract.base_module import BaseAnalysisModule
from analysis.plotting.layout import create_analysis_figure


def _compute_spectrum_1d(field_2d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算 2D 场的径向积分功率谱 (Shell-averaged Power Spectrum)。
    返回: (k_bins, E_k)
    """
    nx, ny = field_2d.shape

    # 1. 做 2D FFT
    # 减去平均值以去除直流分量(k=0)，关注结构变化
    field_fluctuation = field_2d - np.mean(field_2d)
    fft_val = np.fft.fft2(field_fluctuation)
    fft_shift = np.fft.fftshift(fft_val)
    power_spectrum_2d = np.abs(fft_shift) ** 2

    # 2. 构建 k 空间坐标
    kx = np.fft.fftshift(np.fft.fftfreq(nx))
    ky = np.fft.fftshift(np.fft.fftfreq(ny))
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    K_R = np.sqrt(KX ** 2 + KY ** 2)

    # 3. 径向积分 (Binning)
    nbins = min(nx, ny) // 2
    k_bins = np.linspace(0, 0.5, nbins)  # Nyquist frequency is 0.5

    # 使用直方图统计来做径向平均
    # E(k) * dk = Sum of power in ring
    energy_hist, _ = np.histogram(K_R, bins=k_bins, weights=power_spectrum_2d)

    # 获取bin中心
    k_centers = 0.5 * (k_bins[1:] + k_bins[:-1])

    # 归一化：除以环内的像素数（可选，得到平均能量密度），或者直接保留总能量
    # 这里为了计算平均尺度，使用总能量即可

    return k_centers, energy_hist


def compute_fragmentation_metrics(run: SimulationRun, max_samples: int = 50) -> Dict[str, Any]:
    """
    计算场碎裂指标。
    参数:
        max_samples: 为了防止缓存生成太慢，限制采样的帧数。
    """
    files = run.field_files
    if not files:
        return {}

    # 采样逻辑
    if len(files) > max_samples:
        indices = np.linspace(0, len(files) - 1, max_samples, dtype=int)
        selected_files = [files[i] for i in indices]
    else:
        selected_files = files

    times = []
    scales = []
    kurtosis_vals = []

    # 我们只缓存首尾的能谱用于对比绘图
    initial_spec = None
    final_spec = None

    console.print(f"  [cyan]计算场结构碎裂指标 ({len(selected_files)} 帧)...[/cyan]")

    for i, fpath in enumerate(tqdm(selected_files, desc="  分析场结构", unit="file", leave=False)):
        try:
            B_slice = run.get_field_slice_from_path(fpath, axis='z')
            step = int(os.path.basename(fpath).split('_')[-1].split('.')[0])
            time = step * run.sim.dt
        except Exception:
            continue

        # 1. 计算能谱
        k, E_k = _compute_spectrum_1d(B_slice)

        # 2. 计算特征尺度 (Mean weighted wavelength)
        total_energy = np.sum(E_k)
        lam = (1.0 / (np.sum(k * E_k) / total_energy)) if total_energy > 0 and np.sum(k * E_k) > 0 else 0

        # 3. 计算峰度
        k_val = kurtosis(B_slice.flatten(), fisher=True)

        times.append(time)
        scales.append(lam)
        kurtosis_vals.append(k_val)

        if i == 0:
            initial_spec = (k, E_k)
        if i == len(selected_files) - 1:
            final_spec = (k, E_k)

    return {
        "time": np.array(times),
        "characteristic_scales": np.array(scales),
        "kurtosis": np.array(kurtosis_vals),
        "initial_spectrum": initial_spec,
        "final_spectrum": final_spec
    }


class FieldFragmentationModule(BaseAnalysisModule):
    @property
    def name(self) -> str:
        return "磁场碎裂与湍流度分析"

    @property
    def description(self) -> str:
        return "通过FFT能谱和统计峰度，评估磁场结构从大尺度向小尺度碎裂（湍流化）的程度。"

    def _analyze_single_run(self, run: SimulationRun):
        # 直接调用外部函数进行计算
        metrics = compute_fragmentation_metrics(run, max_samples=20)

        if not metrics:
            return

        times = metrics["time"]
        characteristic_scales = metrics["characteristic_scales"]
        kurtosis_vals = metrics["kurtosis"]
        initial_spectrum = metrics["initial_spectrum"]
        final_spectrum = metrics["final_spectrum"]

        # --- 绘图 ---
        # 3个子图：尺度演化、峰度演化、能谱对比
        filename_override = f"{run.name}_analysis_fragmentation"

        with create_analysis_figure(run, "analysis_fragmentation", num_plots=3, figsize=(10, 12), override_filename=filename_override) as (fig, axes):
            ax_scale, ax_kurt, ax_spec = axes

            # Plot 1: 特征尺度演化
            ax_scale.plot(times, characteristic_scales, 'o-', color='teal', lw=2)
            ax_scale.set_title("磁场特征尺度演化 (Characteristic Scale)", fontsize=14)
            ax_scale.set_ylabel(r"特征长度 $\lambda \propto \langle k \rangle^{-1}$ (grid units)")
            ax_scale.set_xlabel("时间 (s)")
            ax_scale.grid(True, alpha=0.3)

            # 标注趋势
            if len(characteristic_scales) > 1:
                delta = characteristic_scales[-1] - characteristic_scales[0]
                pct = (delta / characteristic_scales[0]) * 100 if characteristic_scales[0] != 0 else 0
                color = 'red' if delta < 0 else 'green'
                ax_scale.text(0.05, 0.9, f"变化率: {pct:.1f}%", transform=ax_scale.transAxes, color=color, fontweight='bold')

            # Plot 2: 峰度演化
            ax_kurt.plot(times, kurtosis_vals, 's-', color='darkorange', lw=2)
            ax_kurt.set_title("磁场结构间歇性 (Kurtosis)", fontsize=14)
            ax_kurt.set_ylabel("峰度 (Gaussian=0)")
            ax_kurt.set_xlabel("时间 (s)")
            ax_kurt.grid(True, alpha=0.3)
            ax_kurt.text(0.05, 0.85, "高峰度 $\\rightarrow$ 强局域结构/细丝", transform=ax_kurt.transAxes, fontsize=10, color='gray')

            # Plot 3: 初始 vs 最终 能谱
            if initial_spectrum and final_spectrum:
                k_i, E_i = initial_spectrum
                k_f, E_f = final_spectrum

                # 绘制
                ax_spec.loglog(k_i, E_i, label='Initial (t=0)', color='gray', linestyle='--')
                ax_spec.loglog(k_f, E_f, label=f'Final (t={times[-1]:.2e})', color='crimson', lw=2)

                ax_spec.set_title("磁场湍流功率谱 (Power Spectrum)", fontsize=14)
                ax_spec.set_xlabel(r"波数 $k$ (1/grid)")
                ax_spec.set_ylabel(r"$E_B(k)$")
                ax_spec.legend()
                ax_spec.grid(True, which='both', alpha=0.2)

                # 标注高频部分
                ax_spec.text(0.95, 0.05, "高频能量增加 $\\rightarrow$ 碎裂", transform=ax_spec.transAxes, ha='right', va='bottom', color='crimson')

        console.print(f"  [green]✔ 分析完成: {run.name}[/green]")
        console.print(f"    最终特征尺度: {characteristic_scales[-1]:.2f} (初始: {characteristic_scales[0]:.2f})")
        console.print(f"    最终峰度: {kurtosis_vals[-1]:.2f}")

    def run(self, loaded_runs: List[SimulationRun]):
        console.print("\n[bold magenta]执行: 磁场碎裂与湍流度分析...[/bold magenta]")

        valid_runs = [r for r in loaded_runs if r.field_files]
        if not valid_runs:
            console.print("[yellow]警告: 没有找到有效的场文件 (hdf5)，跳过此分析。[/yellow]")
            return

        for run in valid_runs:
            console.print(f"\n[bold]分析模拟: {run.name}[/bold]")
            self._analyze_single_run(run)
