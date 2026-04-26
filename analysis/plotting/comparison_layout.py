# plotting/layout.py 新增
from typing import Optional, Any, NamedTuple

from numpy import ndarray

from .layout import AnalysisLayout
from ..core.param_display_names import get_param_display
from ..core.parameter_selector import ParameterSelector
from ..core.simulation import SimulationRun
from ..modules.utils.spectrum_tools import filter_valid_runs


class Unpacked(NamedTuple):
    runs: list[SimulationRun]
    x_scaled: ndarray


class ComparisonContext:
    """
    对比分析的数据准备器。

    负责筛选有效模拟、选择 X 轴参数、计算缩放后的坐标值和标签。
    使用后通过 .runs, .x_scaled, .x_label 等属性获取准备好的数据，
    或调用 .unpack 解包出最常用的 (runs, x_scaled)。

    用法:
        try:
            ctx = ComparisonContext(loaded_runs, "multiband_tail")
        except ComparisonContext.InsufficientDataError as e:
            console.print(f"[red]{e}[/red]")
            return

        runs, x_scaled = ctx.unpack   # 解包常用属性
        # 或按需访问:
        x_raw, x_scaled, x_label = ctx.x
        runs = ctx.runs
    """

    class InsufficientDataError(Exception):
        """有效模拟数量不足时抛出"""

        def __init__(self, required, actual):
            super().__init__(f"需要至少 {required} 个有效模拟，实际只有 {actual}")

    def __init__(
            self,
            loaded_runs: list[SimulationRun],
            base_filename: str,
            min_runs: int = 1,
    ):
        """
        Args:
            loaded_runs: 所有加载的模拟运行列表。
            base_filename: 输出文件的基础名称（如 "multiband_tail"）。
            min_runs: 最低有效模拟数量要求，不足时抛出 InsufficientDataError。
        """
        valid_runs = filter_valid_runs(loaded_runs, require_particles=True, min_particle_files=2)
        if len(valid_runs) < min_runs:
            raise self.InsufficientDataError(min_runs, len(valid_runs))

        selector = ParameterSelector(valid_runs)
        x_label_key, x_vals, sorted_runs = selector.select()
        final_filename = selector.generate_filename(x_label_key, sorted_runs, prefix=base_filename)

        x_axis_info = get_param_display(x_label_key)

        x_scaled, final_x_label, is_num = x_axis_info.prepare_axis(x_vals)
        # --- 公开属性 ---
        self.runs: list[SimulationRun] = sorted_runs
        """排序后的有效模拟列表"""

        self.x_raw: list[str] = x_vals
        """X 轴原始值（字符串列表）"""

        self.x_scaled: ndarray = x_scaled
        """X 轴缩放后的数值（用于绘图定位）"""

        self.x_label: str = final_x_label
        """X 轴标签（已包含单位与数量级）"""

        self.is_num: bool = is_num
        """X 轴是否为数值型"""

        self.output_filename: str = final_filename
        """完整的输出文件名（含哈希前缀），传给 AnalysisLayout"""

    @property
    def x(self) -> tuple[list[Any], ndarray, str]:
        """一次获取 X 轴全部信息: (x_raw, x_scaled, x_label)"""
        return self.x_raw, self.x_scaled, self.x_label

    @property
    def unpack(self) -> tuple[list[SimulationRun], ndarray]:
        """解包最常用的两个属性: (runs, x_scaled)"""
        return self.runs, self.x_scaled


class ComparisonLayout(AnalysisLayout):
    """
    对比分析专用布局管理器，继承自 AnalysisLayout。

    在 AnalysisLayout 的基础上增加了绘图结束时的自动收尾：
    - 隐藏除最后一个轴以外的 X 轴刻度标签
    - 为最后一个轴设置 X 轴标签

    用法:
        with ComparisonLayout(ctx) as layout:
            ax1 = layout.request_axes()
            ax1.plot(ctx.x_scaled, data1)
            ax2 = layout.request_axes()
            ax2.plot(ctx.x_scaled, data2)
        # 退出时自动收尾并保存
    """

    def __init__(
            self,
            ctx: ComparisonContext,
            plot_ratio: Optional[tuple[float, float]] = None,
    ):
        super().__init__(
            ctx.runs,
            base_filename="",  # 不使用 AnalysisLayout 的默认文件名逻辑
            plot_ratio=plot_ratio,
            override_filename=ctx.output_filename,
        )
        self._ctx = ctx

    def __exit__(self, exc_type, exc_val, exc_tb):
        """绘图结束后统一收尾，然后委托给 AnalysisLayout 保存并清理。"""
        if exc_type is not None:
            return super().__exit__(exc_type, exc_val, exc_tb)

        # 隐藏除最后一个轴以外的 X 轴刻度标签，保持图表整洁
        for ax in self.plot_axes[:-1]:
            ax.set_xticklabels([])
            ax.set_xlabel("")

        # 仅为最后一个轴设置 X 轴标签
        if self.plot_axes:
            self.plot_axes[-1].set_xlabel(self._ctx.x_label)

        # 委托父类完成参数表绘制、双版本保存和资源清理
        return super().__exit__(None, None, None)
