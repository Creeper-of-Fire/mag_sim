# plotting/layout.py

from contextlib import contextmanager
from typing import List, Generator, Optional
from warnings import deprecated

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .styles import get_style
from ..core.param_table import plot_parameter_table, plot_comparison_parameter_table
from ..core.simulation import SimulationRun
from ..core.utils import save_figure


def _get_table_actual_height_inch(
        run_or_runs: SimulationRun | list[SimulationRun],
        figure_width_inch: float
) -> float:
    """
    通过在临时画布上预渲染表格来精确计算其物理高度。

    Args:
        run_or_runs: 用于生成表格的 SimulationRun 对象或列表。
        figure_width_inch (float): 最终图形的宽度，因为表格列宽可能受其影响。

    Returns:
        float: 表格实际占用的高度（英寸）。
    """
    # 创建一个临时的、不可见的 figure
    # 宽度必须与最终的图匹配，高度不重要，它会被内容撑开
    temp_fig = plt.figure(figsize=(figure_width_inch, 1))
    temp_ax = temp_fig.add_subplot(111)

    # 绘制表格
    is_comparison = isinstance(run_or_runs, list)
    if is_comparison:
        table = plot_comparison_parameter_table(temp_ax, run_or_runs)
    else:
        table = plot_parameter_table(temp_ax, run_or_runs)

    # 如果没有生成表格（例如，数据为空），则返回一个很小的高度
    if table is None:
        plt.close(temp_fig)
        return 0.1

    # 强制 matplotlib 计算所有元素的位置和大小
    temp_fig.canvas.draw()

    # 获取包含表格的轴的边界框（以像素为单位）
    # get_tightbbox 是最可靠的方法，因为它会考虑所有装饰器（标题等）
    bbox = temp_ax.get_tightbbox(temp_fig.canvas.get_renderer())

    # 将像素高度转换为英寸
    table_height = bbox.height / temp_fig.dpi

    # 增加一点点边距，以防万一
    table_height += 0.25

    # 清理临时 figure
    plt.close(temp_fig)

    return table_height


class AnalysisLayout:
    """
    动态分析图布局管理器。

    使用方法:
        with AnalysisLayout(run, "spectrum") as layout:
            ax1 = layout.add_axes(ratio=2)   # 高度比例2
            ax2 = layout.add_axes()          # 默认比例1
            ax1.plot(...)
            ax2.plot(...)
        # 退出时自动绘制参数表，保存带表/不带表两个版本
    """

    def __init__(
            self,
            run_or_runs: SimulationRun | list[SimulationRun],
            base_filename: str,
            plot_ratio: Optional[tuple[float, float]] = None,  # 单个绘图区的宽高比
            override_filename: Optional[str] = None,
    ):
        self.run_or_runs = run_or_runs
        self.base_filename = base_filename
        self.override_filename = override_filename

        # --- 标题和输出文件名 ---
        if self.override_filename:
            self.output_name = f"{self.override_filename}.png"
        else:
            is_comp = isinstance(run_or_runs, list)
            if is_comp:
                if not run_or_runs:
                    raise ValueError("对比分析至少需要一个 SimulationRun 实例。")
                names = tuple(sorted(r.name for r in run_or_runs))
                short = hex(abs(hash(names)))[2:8]
                self.output_name = f"{short}_{base_filename}.png"
            else:
                self.output_name = f"{run_or_runs.name}_{base_filename}.png"

        # --- 画布宽度 ---
        style = get_style()
        base_width, base_height = style.figsize
        self.plot_width = base_width
        if plot_ratio is None:
            prop_w, prop_h = base_width, base_height
        else:
            prop_w, prop_h = plot_ratio

        # 单个绘图单元的高度 = 宽度按比例换算
        self._unit_height = base_width * (prop_h / prop_w)

        # --- 预计算表格高度 ---
        self.table_height_inch = _get_table_actual_height_inch(run_or_runs, self.plot_width)

        # --- 动态状态 ---
        self.fig: Figure = plt.figure(figsize=(self.plot_width, 0.1))  # 临时高度
        self.plot_axes: List[Axes] = []  # 按添加顺序
        self.plot_heights: List[float] = []  # 英寸
        self.table_ax: Optional[Axes] = None

    def request_axes(self, ratio: float = 1.0) -> Axes:
        """
        申请一个绘图轴并立即返回。

        Args:
            ratio: 相对于 unit_plot_height 的高度比例，越大轴越高。
        Returns:
            Axes 对象，可直接在其上绘图。
        """
        height = ratio * self._unit_height
        self.plot_heights.append(height)

        # 扩展画布高度
        total = sum(self.plot_heights) + self.table_height_inch
        self.fig.set_size_inches(self.plot_width, total)

        # 创建新轴（占位坐标，待统一重排）
        ax = self.fig.add_axes((0, 0, 1, 0))
        self.plot_axes.append(ax)
        self._reposition_all()
        return ax

    def _reposition_all(self):
        """根据当前绘图轴列表和表格高度，重新计算所有轴的归一化位置。"""
        total_h = sum(self.plot_heights) + self.table_height_inch
        # 表格始终位于底部
        table_bottom = 0.0
        table_height_frac = self.table_height_inch / total_h

        # 如果表格轴还不存在，创建它
        if self.table_ax is None:
            self.table_ax = self.fig.add_axes(
                (0.1, table_bottom, 0.8, table_height_frac)
            )
        else:
            self.table_ax.set_position((0.1, table_bottom, 0.8, table_height_frac))

        # 从表格上方开始依次放置绘图轴
        cur_bottom = table_height_frac
        for ax, h_inch in zip(self.plot_axes, self.plot_heights):
            h_frac = h_inch / total_h
            ax.set_position((0.1, cur_bottom, 0.8, h_frac))
            cur_bottom += h_frac

    def __enter__(self) -> "AnalysisLayout":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            plt.close(self.fig)
            return False  # 传播异常

        # --- 绘制参数表 ---
        is_comp = isinstance(self.run_or_runs, list)
        if is_comp:
            plot_comparison_parameter_table(self.table_ax, self.run_or_runs)
        else:
            plot_parameter_table(self.table_ax, self.run_or_runs)

        # --- 保存带表格版本 ---
        save_figure(self.fig, self.output_name, run_or_runs=self.run_or_runs, subfolder="with_table")

        # --- 保存不带表格版本 ---
        self.table_ax.set_visible(False)
        save_figure(self.fig, self.output_name, run_or_runs=self.run_or_runs, subfolder="without_table")

        plt.close(self.fig)
        return False  # 不抑制异常


@deprecated("请改用 AnalysisLayout 上下文管理器")
@contextmanager
def create_analysis_figure(
        run_or_runs: SimulationRun | list[SimulationRun],
        base_filename: str,
        num_plots: int,
        plot_ratios: List[int] = None,
        figsize: Optional[tuple[float, float]] = None,
        override_filename: Optional[str] = None,
) -> Generator[tuple[Figure, Axes] | tuple[Figure, tuple[Axes, ...]], None, None]:
    """
    传统固定数量的上下文管理器，内部委托给 AnalysisLayout。
    用于向后兼容，不推荐新代码使用。
    """
    if plot_ratios is None:
        plot_ratios = [1] * num_plots

    layout = AnalysisLayout(
        run_or_runs,
        base_filename,
        plot_ratio=figsize,
        override_filename=override_filename,
    )
    with layout:
        axes = [layout.request_axes(ratio=r) for r in plot_ratios]
        if num_plots == 1:
            yield layout.fig, axes[0]
        else:
            yield layout.fig, tuple(axes)
