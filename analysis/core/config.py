# analysis/core/config.py
#
# --- 全局配置模块 ---
#
# 这个模块提供了一个全局共享的配置对象。
# 主程序 (analyze.py) 会在启动时设置这些值，
# 然后任何分析模块都可以导入并使用它们。
#
from dataclasses import dataclass


@dataclass
class Config:
    """
    一个用于存放全局配置的数据类。

    属性:
        output_dir (str): 所有输出文件（图片、视频等）应保存到的目录路径。
    """
    output_dir: str = "analysis_results1"

    # 未来可以轻松扩展，并提供类型安全：
    # plot_dpi: int = 200
    # default_cmap: str = "viridis"


# 创建并导出一个全局唯一的配置实例。
# 应用程序的其他部分将导入并修改这个实例。
config = Config()
