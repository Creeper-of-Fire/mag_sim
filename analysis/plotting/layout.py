# plotting/layout.py

from typing import List, Optional

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .styles import get_style
from ..core.simulation import SimulationRun
from ..core.utils import save_figure


class AnalysisLayout:
    """
    动态分析图布局管理器。

    使用方法:
        with AnalysisLayout(run, "spectrum") as layout:
            ax1 = layout.add_axes(ratio=2)   # 高度比例2
            ax2 = layout.add_axes()          # 默认比例1
            ax1.plot(...)
            ax2.plot(...)
        # 退出时自动保存图像
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

        # --- 动态状态 ---
        self.fig: Figure = plt.figure(figsize=(self.plot_width, 0.1))  # 临时高度
        self.plot_axes: List[Axes] = []  # [最上面的轴, ..., 最下面的轴(最新)]
        self.plot_ratios: List[float] = []  # 英寸
        self._gs = None

    def request_axes(self, ratio: float = 1.0) -> Axes:
        """
        申请一个绘图轴并立即返回。

        Args:
            ratio: 相对于 unit_plot_height 的高度比例，越大轴越高。
        Returns:
            Axes 对象，可直接在其上绘图。
        """
        self.plot_ratios.append(ratio)

        # 计算所有行的高度比例
        heights = [r * self._unit_height for r in self.plot_ratios]

        # 更新画布总高度
        total_height = sum(heights)
        self.fig.set_size_inches(self.plot_width, total_height)

        # 创建新的GridSpec
        n_rows = len(heights)
        new_gs = self.fig.add_gridspec(n_rows, 1, height_ratios=heights,
                                       hspace=0.15, top=0.95, bottom=0.05)

        # 移动现有绘图轴到新位置（从上到下）
        for i, ax in enumerate(self.plot_axes):
            if ax.get_subplotspec() is not None:
                # 清除旧gridspec关联，但保留内容
                ax.set_subplotspec(new_gs[i, 0])

        # 创建新的绘图轴（倒数第二行，紧挨表格上方）
        new_idx = len(self.plot_axes)
        new_ax = self.fig.add_subplot(new_gs[new_idx, 0])
        self.plot_axes.append(new_ax)

        # 更新GridSpec引用
        self._gs = new_gs

        return new_ax

    def __enter__(self) -> "AnalysisLayout":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            plt.close(self.fig)
            return False  # 传播异常

        # 强制渲染，确保所有set_subplotspec生效
        self.fig.canvas.draw()

        save_figure(self.fig, self.output_name, run_or_runs=self.run_or_runs)

        plt.close(self.fig)
        return False  # 不抑制异常
