# plotting/base_plotter.py

from abc import ABC, abstractmethod
from matplotlib.axes import Axes

# 导入类型提示，避免循环导入
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.simulation import SimulationRun


class BasePlotter(ABC):
    """
    所有绘图器的抽象基类。
    定义了绘图器的标准接口。
    """

    @abstractmethod
    def plot(self, ax: Axes, run: 'SimulationRun', **kwargs):
        """
        在给定的 Matplotlib Axes 对象上绘制单个模拟运行的数据。

        Args:
            ax (Axes): 用于绘图的 Matplotlib Axes 对象。
            run (SimulationRun): 包含数据的模拟运行对象。
            **kwargs: 传递给绘图函数的额外参数 (例如 label, color, linestyle)。
                      这对于实现对比图至关重要。
        """
        pass

    @abstractmethod
    def setup_axes(self, ax: Axes):
        """
        配置 Axes 的外观，如标题、坐标轴标签、刻度等。

        Args:
            ax (Axes): 需要配置的 Matplotlib Axes 对象。
        """
        pass