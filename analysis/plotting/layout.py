# plotting/layout.py

from contextlib import contextmanager
from typing import List, Tuple, Any, Generator, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.transforms import Bbox

from .styles import get_style
from ..core.simulation import SimulationRun
from ..core.utils import save_figure
from ..core.param_table import plot_parameter_table, plot_comparison_parameter_table


def _get_table_actual_height_inch(
        run_or_runs: Union[SimulationRun, List[SimulationRun]],
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


@contextmanager
def create_analysis_figure(
        run_or_runs: SimulationRun | List[SimulationRun],
        base_filename: str,
        num_plots: int,
        plot_ratios: List[int] = None,
        figsize: Optional[Tuple[float, float]] = None,
        override_filename: Optional[str] = None
) -> Generator[tuple[Any, Any] | tuple[Any, tuple[Any, ...]], Any, None]:
    """
    一个用于标准分析图的上下文管理器，自动处理布局、双版本保存和资源清理。

    它会创建一个包含N个数据图和一个参数表区域的画布。在退出时，
    它会自动保存两个版本的图：一个带参数表，一个不带。

    用法:
        with create_analysis_figure(run, "spectrum", num_plots=1) as (fig, axes):
            ax_plot, ax_table = axes
            # ... 在 ax_plot 和 ax_table 上绘图 ...
        # 此处不需要调用 save_figure 或 plt.close

    Args:
        run_or_runs: 单个 SimulationRun 对象或一个列表。用于标题、文件名和参数表。
        base_filename (str): 输出文件的基础名称 (例如 "analysis_spectrum")。
        num_plots (int): 需要的数据绘图区域数量。
        plot_ratios (List[int], optional): 各个数据图的高度比例。默认为等分。
        table_ratio (int, optional): 参数表的高度比例。默认为 3。
        figsize (Tuple[float, float], optional): 画布尺寸。

    Yields:
        Tuple[Figure, Tuple[Axes, ...]]: Matplotlib Figure 对象和所有 Axes 对象的元组。
                                          最后一个 Axes 始终是为参数表准备的。
    """
    # --- 文件名生成逻辑 ---
    if override_filename:
        # 如果提供了覆盖文件名，则优先使用它
        output_name = f"{override_filename}.png"
        # 标题仍然可以自动生成
        is_comparison = isinstance(run_or_runs, list)
        if is_comparison:
            title = f"对比分析: {base_filename.replace('_', ' ').title()}"
        else:
            title = f"分析: {base_filename.replace('_', ' ').title()} for: {run_or_runs.name}"

    else:
        # 否则，使用现有的自动命名逻辑
        is_comparison = isinstance(run_or_runs, list)
        if is_comparison:
            if not run_or_runs:
                raise ValueError("对比分析至少需要一个 SimulationRun 实例。")
            run_names = tuple(sorted([r.name for r in run_or_runs]))
            short_hash = hex(abs(hash(run_names)))[2:8]
            output_name = f"{base_filename}_{short_hash}.png"
            title = f"对比分析: {base_filename.replace('_', ' ').title()}"
        else:
            output_name = f"{base_filename}_{run_or_runs.name}.png"
            title = f"分析: {base_filename.replace('_', ' ').title()} for: {run_or_runs.name}"

    # --- 根据样式和传入的比例计算最终尺寸 ---

    style = get_style()
    base_width, base_height = style.figsize  # 获取样式的边界框

    if figsize is None:
        # 如果用户未提供figsize，则直接使用样式定义的标准尺寸
        plot_figsize = (base_width, base_height)
    else:
        # --- 取最大缩放比例 (Cover) ---

        req_w, req_h = figsize

        # 计算两个维度的缩放因子
        # 例如：基准是 10x6
        # 情况1：请求 12x15。
        #       scale_w = 10/12 ≈ 0.83
        #       scale_h = 6/15 = 0.4
        #       我们应该选 0.83 (让宽度撑满10，高度随之变成 12.5)，而不是选 0.4 (让高度缩到6，宽度变成4.8)

        scale_w = base_width / req_w
        scale_h = base_height / req_h

        # 取 max，保证图像足够大，至少有一边能撑满基准尺寸
        scale = max(scale_w, scale_h)

        final_plot_width = req_w * scale
        final_plot_height = req_h * scale

        plot_figsize = (final_plot_width, final_plot_height)

    # --- 1. 精确计算布局 ---
    plot_width, plot_height = plot_figsize

    # a. 通过预渲染精确计算表格的实际高度
    actual_table_height = _get_table_actual_height_inch(run_or_runs, plot_width)

    # b. 计算带表格时的总画布高度
    total_height_with_table = plot_height + actual_table_height

    # c. 计算 gridspec 的高度比例，现在基于精确的物理尺寸
    if plot_ratios is None:
        plot_ratios = [1] * num_plots
    total_plot_ratio_units = sum(plot_ratios)

    # 将物理尺寸转换为比例
    final_plot_ratios = [(r / total_plot_ratio_units) * plot_height for r in plot_ratios]
    height_ratios = final_plot_ratios + [actual_table_height]

    # --- 2. 创建主画布 ---
    fig, axes_array = plt.subplots(
        num_plots + 1, 1,
        figsize=(plot_width, total_height_with_table),
        gridspec_kw={'height_ratios': height_ratios},
        constrained_layout=True
    )
    if num_plots > 0 and not isinstance(axes_array, (list, np.ndarray)):
        axes_array = [axes_array]

    fig.suptitle(title, fontsize=18)

    plot_axes = axes_array[:-1] if num_plots > 0 else []
    table_ax = axes_array[-1]

    try:
        # --- 3. 将控制权和【仅绘图区】交给调用者 ---
        if num_plots == 1:
            yield fig, plot_axes[0]
        else:
            yield fig, tuple(plot_axes)

        # --- 4. 调用者完成绘图，收回控制权，绘制参数表 ---
        if is_comparison:
            plot_comparison_parameter_table(table_ax, run_or_runs)
        else:
            plot_parameter_table(table_ax, run_or_runs)

        # --- 5. 保存带表格的版本 ---
        # 此时布局已经最终确定
        save_figure(fig, output_name, subfolder="with_table")

        # --- 6. 准备并保存不带表格的版本 ---
        table_ax.set_visible(False)

        # `save_figure` 函数中的 `bbox_inches='tight'` 会自动裁剪掉
        # 因表格不可见而产生的底部空白区域，同时保持绘图区的几何形状不变。
        save_figure(fig, output_name, subfolder="without_table")


    finally:
        # --- 6. 清理资源 ---
        plt.close(fig)
