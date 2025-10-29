# modules/base_module.py

from abc import ABC, abstractmethod
from typing import List, Set

# 导入类型提示，避免循环导入
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..core.simulation import SimulationRun

class BaseAnalysisModule(ABC):
    """
    所有分析模块的抽象基类。
    它定义了所有分析模块必须提供的标准接口。
    """
    @property
    @abstractmethod
    def name(self) -> str:
        """
        模块的名称，用于在交互式菜单中向用户显示。
        例如: "能量演化分析 (场能 vs 动能)"
        """
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """
        模块功能的简短描述，可以在菜单中作为帮助信息显示。
        例如: "绘制总能量、场能和动能随时间的演化图。"
        """
        pass

    @property
    @abstractmethod
    def required_data(self) -> Set[str]:
        """
        声明此模块运行所必需的数据类型。
        主程序将根据这个集合，调用 data_loader 来加载数据。
        可用键: 'energy', 'field', 'initial_spectrum', 'final_spectrum', 'field_files'
        """
        pass

    @abstractmethod
    def run(self, loaded_runs: List['SimulationRun']):
        """
        执行分析和绘图的核心方法。
        主程序会调用此方法，并传入一个已经加载好所需数据的 SimulationRun 列表。
        """
        pass

class BaseComparisonModule(BaseAnalysisModule, ABC):
    """
    所有【对比】分析模块的抽象基类。
    它的 run 方法应该将所有模拟数据作为一个整体来处理，通常生成一张包含所有模拟对比结果的图。
    通过继承 BaseAnalysisModule，它自动获得了统一的接口。
    """
    pass

class BaseVideoModule(BaseAnalysisModule, ABC):
    """
    所有【视频生成】模块的抽象基类。
    通过继承 BaseAnalysisModule，它自动获得了统一的接口，并通过类本身进行区分。
    """
    pass