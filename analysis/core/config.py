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
    # 跨 Job 对比时的兜底输出目录 (当无法确定唯一的 JobDir 时使用)
    global_output_dir: str = "analysis_results_global"

    # 相对路径配置 (相对于 JobDir)
    analysis_folder_name: str = "analysis"           # 一级目录: JobDir/analysis
    
    # 子目录名称
    single_analysis_subfolder: str = "single_runs"   # 单次分析: JobDir/analysis/single_runs
    comparison_subfolder: str = "comparisons"        # 对比分析: JobDir/analysis/comparisons
    video_subfolder: str = "videos"                  # 视频文件: JobDir/analysis/videos

config = Config()
