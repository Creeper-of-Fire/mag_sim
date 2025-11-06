# plotting/style_manager.py

from dataclasses import dataclass
from enum import Enum
from typing import Tuple
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class PlotStyle:
    """
    一个用于存储绘图样式配置的数据类。
    这里不仅包含matplotlib的参数，还包含了我们自己定义的“语义化”样式。
    """
    lw_base: float

    # --- 语义化样式 (我们在代码中直接调用这些) ---
    ls_primary: str  # 主要线条的样式
    ls_secondary: str  # 次要线条的样式
    ls_tertiary: str  # 第三线条的样式 (例如虚线)

    color_baseline_primary: str
    color_baseline_secondary: str
    color_comparison_primary: str
    color_comparison_secondary: str

    # --- Matplotlib 全局参数 (由 apply() 方法设置) ---
    figsize: Tuple[float, float]
    font_size_base: int
    font_size_title: int
    font_size_label: int
    font_size_tick: int
    font_size_legend: int
    savefig_dpi: int
    savefig_format: str

    def apply(self):
        """将此样式配置应用到 matplotlib.rcParams"""
        plt.rcParams.update({
            'figure.figsize': self.figsize,
            'font.size': self.font_size_base,
            'axes.titlesize': self.font_size_title,
            'axes.labelsize': self.font_size_label,
            'xtick.labelsize': self.font_size_tick,
            'ytick.labelsize': self.font_size_tick,
            'legend.fontsize': self.font_size_legend,
            'figure.titlesize': self.font_size_title,
            'savefig.dpi': self.savefig_dpi,
            'savefig.format': self.savefig_format,
            'lines.linewidth': self.lw_base,  # 将次要线宽设为默认
        })


# --- 定义具体的样式实例 ---

PaperStyle = PlotStyle(
    lw_base=1.0,
    # 语义化样式
    ls_primary='-',
    ls_secondary='--',
    ls_tertiary=':',
    color_baseline_primary='black',
    color_baseline_secondary='gray',
    color_comparison_primary='C0',
    color_comparison_secondary='C1',
    # Matplotlib参数
    figsize=(7.0, 4.33),
    font_size_base=10,
    font_size_title=12,
    font_size_label=10,
    font_size_tick=8,
    font_size_legend=8,
    savefig_dpi=300,
    savefig_format='pdf'
)

PresentationStyle = PlotStyle(
    lw_base=2.5,
    # 语义化样式
    ls_primary='-',
    ls_secondary='--',
    ls_tertiary=':',
    color_baseline_primary='black',
    color_baseline_secondary='gray',
    color_comparison_primary='C0',  # 可以为PPT换更亮的颜色
    color_comparison_secondary='C3',
    # Matplotlib参数
    figsize=(10, 6.0),
    font_size_base=18,
    font_size_title=24,
    font_size_label=22,
    font_size_tick=18,
    font_size_legend=18,
    savefig_dpi=150,
    savefig_format='png'
)


class StyleTheme(Enum):
    """使用枚举来选择样式，提供类型安全和自动补全"""
    PAPER = PaperStyle
    PRESENTATION = PresentationStyle


# --- 全局状态管理器 ---
_current_style: PlotStyle = PresentationStyle  # 默认为PPT样式


def set_style(theme: StyleTheme):
    """在程序开始时设置全局绘图样式"""
    global _current_style
    _current_style = theme.value
    _current_style.apply()
    print(f"绘图样式已设置为: {theme.name}")


def get_style() -> PlotStyle:
    """在任何绘图模块中获取当前激活的样式对象"""
    return _current_style