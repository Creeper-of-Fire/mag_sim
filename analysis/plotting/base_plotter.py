# plotting/base_plotter.py

from abc import ABC, abstractmethod
from matplotlib.axes import Axes

# 导入类型提示，避免循环导入
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from ..core.simulation import SimulationRun


class BasePlotter(ABC):
    """
    所有绘图器的抽象基类。
    定义了绘图器的标准接口。
    """

    @abstractmethod
    def plot(self, ax: Axes, run: 'SimulationRun', label: str, color: Optional[str] = None, **kwargs):
        """
        在给定的 Matplotlib Axes 对象上绘制单个模拟运行的数据。

        Args:
            ax (Axes): 用于绘图的 Matplotlib Axes 对象。
            run (SimulationRun): 包含数据的模拟运行对象。
            label (str): 用于图例的标签，此为必需参数。
            color (Optional[str]): 绘图所用的颜色。如果未提供，通常将使用 matplotlib 的默认颜色循环。
            **kwargs: 传递给底层绘图函数的其他参数 (例如 linestyle, lw)。
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