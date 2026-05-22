# plotting/comparison_layout.py
from typing import Optional, Any, NamedTuple

import numpy as np

from .data_layout import DataLayout
from ..core.param_display_names import get_param_display, ParamInfo
from ..core.parameter_selector import ParameterSelector
from ..core.simulation import SimulationRun
from ..modules.utils.spectrum_tools import filter_valid_runs


class Unpacked(NamedTuple):
    runs: list[SimulationRun]
    x_scaled: np.ndarray


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

        x_label_info = get_param_display(x_label_key)

        x_scaled, final_x_label, is_num = x_label_info.prepare_axis(x_vals)
        # --- 公开属性 ---
        self.runs: list[SimulationRun] = sorted_runs
        """排序后的有效模拟列表"""

        self.x_raw: list[str] = x_vals
        """X 轴原始值（字符串列表）"""

        self.x_scaled: np.ndarray = x_scaled
        """X 轴缩放后的数值（用于绘图定位）"""

        self.x_label_key: str = x_label_key
        """X 轴参数的原始键名（如 "target_sigma", "B0"）"""

        self.x_label_info: ParamInfo = x_label_info
        """X 轴标签的完整 ParamInfo 对象，可访问 .name_cn, .symbol, .unit 等"""

        self.x_label_str: str = final_x_label
        """X 轴标签（已包含单位与数量级）"""

        self.is_num: bool = is_num
        """X 轴是否为数值型"""

        self.output_filename: str = final_filename
        """完整的输出文件名（含哈希前缀），传给 AnalysisLayout"""

    @property
    def x(self) -> tuple[list[Any], np.ndarray]:
        """一次获取 X 轴全部信息: (x_raw, x_scaled)"""
        return self.x_raw, self.x_scaled

    @property
    def x_label(self) -> tuple[str, str, ParamInfo, bool]:
        """X 轴标签信息: (x_label_str, x_label_key, x_label_info, is_num)"""
        return self.x_label_str, self.x_label_key, self.x_label_info, self.is_num


    @property
    def unpack(self) -> tuple[list[SimulationRun], np.ndarray]:
        """解包最常用的两个属性: (runs, x_scaled)"""
        return self.runs, self.x_scaled


class ComparisonLayout(DataLayout):
    """
    对比分析专用布局管理器，DataLayout 的薄包装。

    从 ComparisonContext 提取 xlabel 和 xtick 配置，委托给 DataLayout 处理
    共享 xlabel、CSV 导出和图片保存。

    用法:
        with ComparisonLayout(ctx, suffix="excess") as layout:
            ax = layout.request_axes()
            ax.plot(ctx.x_scaled, data)
        # 退出时自动收尾并保存
    """

    def __init__(
            self,
            ctx: ComparisonContext,
            plot_ratio: Optional[tuple[float, float]] = None,
            suffix: Optional[str] = None,
            ncols: int = 1,
    ):
        filename = ctx.output_filename
        if suffix:
            filename = f"{filename}_{suffix}"
        super().__init__(
            ctx.runs,
            base_filename="",
            plot_ratio=plot_ratio,
            override_filename=filename,
            ncols=ncols,
            shared_xlabel=ctx.x_label_str,
            xtick_labels=ctx.x_raw if not ctx.is_num else None,
        )
        self._ctx = ctx
