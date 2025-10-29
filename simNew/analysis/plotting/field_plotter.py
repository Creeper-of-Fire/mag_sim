# plotting/field_plotter.py
from typing import Optional

from matplotlib.axes import Axes

from .base_plotter import BasePlotter
from ..core.simulation import SimulationRun


class FieldRmsPlotter(BasePlotter):
    """绘制磁场分量 RMS 值的演化。"""

    def plot(self, ax: Axes, run: SimulationRun, label: str, color: Optional[str] = None, **kwargs):
        data = run.field_data
        label_suffix = f" ({label})"

        ax.plot(data.time, data.b_rms_x_normalized, '-', label='RMS(Bx)' + label_suffix, **kwargs)
        ax.plot(data.time, data.b_rms_y_normalized, '--', label='RMS(By)' + label_suffix, **kwargs)
        ax.plot(data.time, data.b_rms_z_normalized, ':', label='RMS(Bz)' + label_suffix, **kwargs)

    def setup_axes(self, ax: Axes):
        ax.set_title('磁场分量RMS值演化 (湍流各向异性分析)', fontsize=14)
        ax.set_ylabel('分量RMS值 / B_norm')
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True, which="both", ls="--", alpha=0.5)


class FieldMeanPlotter(BasePlotter):
    """绘制磁场分量平均值的演化。"""

    def plot(self, ax: Axes, run: SimulationRun, label: str, color: Optional[str] = None, **kwargs):
        data = run.field_data
        label_suffix = f" ({label})"

        ax.plot(data.time, data.b_mean_x_normalized, '-', label='<Bx>' + label_suffix, **kwargs)
        ax.plot(data.time, data.b_mean_y_normalized, '--', label='<By>' + label_suffix, **kwargs)
        ax.plot(data.time, data.b_mean_z_normalized, ':', label='<Bz>' + label_suffix, **kwargs)
        ax.axhline(0.0, color='black', linestyle='-', linewidth=1, alpha=0.7)

    def setup_axes(self, ax: Axes):
        ax.set_title('磁场分量平均值 <B> 演化 (宏观各向异性分析)', fontsize=14)
        ax.set_ylabel('平均磁场分量 <B_i> / B_norm')
        ax.legend()
        ax.grid(True, which="both", ls="--", alpha=0.5)


class FieldMagnitudePlotter(BasePlotter):
    """绘制磁场强度（平均值和最大值）的演化。"""

    def plot(self, ax: Axes, run: SimulationRun, label: str, color: Optional[str] = None, **kwargs):
        data = run.field_data
        label_suffix = f" ({label})"

        ax.plot(data.time, data.b_mean_abs_normalized, '-', label='<|B|>' + label_suffix, **kwargs)
        ax.plot(data.time, data.b_max_normalized, '--', alpha=0.9, label='Max|B|' + label_suffix, **kwargs)

    def setup_axes(self, ax: Axes):
        ax.set_title('磁场强度演化 (增长与饱和分析)', fontsize=14)
        ax.set_xlabel('时间 (s)', fontsize=12)
        ax.set_ylabel('磁场强度 |B| / B_norm')
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True, which="both", ls="--", alpha=0.5)