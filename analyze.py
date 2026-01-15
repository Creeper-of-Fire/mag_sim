#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# --- WarpX 交互式分析框架主入口 ---
#
# 功能:
# 1. 自动扫描并加载 `modules/` 目录下的所有分析模块。
# 2. 提供一个交互式菜单让用户选择要执行的分析。
# 3. 统一处理模拟目录的选择。
# 4. 根据所选分析的需求，按需加载数据。
# 5. 依次执行所选的分析模块。
#
import argparse
import glob
import importlib
import os
import shutil
from pathlib import Path
from typing import List, Dict, Union

from rich.prompt import Prompt, Confirm
from tqdm import tqdm

from analysis.core.config import config
from analysis.core.data_loader import load_run_data
# --- 导入核心库组件 ---
from analysis.core.utils import console, select_directories
from analysis.utils import setup_chinese_font
from analysis.modules.base_module import BaseAnalysisModule, BaseComparisonModule, BaseVideoModule
from analysis.plotting.styles import StyleTheme, set_style

# 定义模块类型别名
AnyModule = Union[BaseAnalysisModule, BaseComparisonModule, BaseVideoModule]


def discover_modules() -> tuple[dict[str, BaseAnalysisModule], dict[str, BaseComparisonModule], dict[str, BaseVideoModule]]:
    """
    动态扫描 `modules` 文件夹，加载并区分单个分析模块和对比分析模块。
    """
    individual_modules = {}
    comparison_modules = {}
    video_modules = {}
    modules_dir = Path(__file__).resolve().parent / "analysis/modules"

    for f in modules_dir.glob("*.py"):
        if f.name.startswith(('_', 'base_')):
            continue

        module_name = f"analysis.modules.{f.stem}"
        try:
            module = importlib.import_module(module_name)
            for item_name in dir(module):
                item = getattr(module, item_name)

                # 避免加载基类自身
                if not isinstance(item, type) or item.__module__ != module_name:
                    continue

                if not issubclass(item, BaseAnalysisModule) or item in [BaseAnalysisModule, BaseComparisonModule, BaseVideoModule]:
                    continue

                instance = item()
                if isinstance(instance, BaseVideoModule):
                    video_modules[instance.name] = instance
                elif isinstance(instance, BaseComparisonModule):
                    comparison_modules[instance.name] = instance
                elif isinstance(instance, BaseAnalysisModule):
                    individual_modules[instance.name] = instance

        except Exception as e:
            console.print(f"[red]加载模块 {module_name} 失败: {e}[/red]")

    return dict(sorted(individual_modules.items())), dict(sorted(comparison_modules.items())), dict(sorted(video_modules.items()))


def _select_modules_from_list(module_dict: Dict[str, AnyModule]) -> List[AnyModule]:
    """通用模块选择交互函数"""
    if not module_dict:
        return []

    module_list = list(module_dict.values())
    console.print("\n[bold underline]请选择要执行的分析模块：[/bold underline]\n")
    for i, mod in enumerate(module_list):
        console.print(f"[[cyan]{i}[/cyan]] [magenta]{mod.name}[/magenta]")
        console.print(f"    ↳ [white]{mod.description}[/white]")
    console.print("")

    while True:
        try:
            prompt_text = "[bold]请输入索引 (用逗号/空格分隔, [cyan]回车全选[/cyan]): [/bold]"
            choice_str = Prompt.ask(prompt_text, default="all")

            if choice_str.strip().lower() == "all":
                console.print(f"[green]✔ 已选择全部 {len(module_list)} 个模块。[/green]")
                return module_list

            indices_str = choice_str.replace(',', ' ').split()
            if not indices_str: continue

            indices = [int(i) for i in indices_str]
            if all(0 <= i < len(module_list) for i in indices):
                return [module_list[i] for i in indices]
            else:
                console.print("[yellow]警告: 输入的索引超出范围，请重试。[/yellow]")
        except ValueError:
            console.print("[red]错误: 无效输入，请输入数字索引。[/red]")


def _run_analysis_workflow(selected_modules: List[AnyModule], selected_dirs: List[str]):
    """统一的数据加载和模块执行流程"""
    if not selected_modules or not selected_dirs:
        return

    all_required_data = set().union(*(mod.required_data for mod in selected_modules))
    console.print(f"\n[bold]将为分析加载以下数据类型: {all_required_data}[/bold]")

    loaded_runs = [run for dir_path in selected_dirs if (run := load_run_data(dir_path, all_required_data))]
    if not loaded_runs:
        console.print("\n[red]未能成功加载任何模拟数据，无法继续分析。[/red]")
        return

    console.print("\n" + "=" * 50)
    console.print("[bold green]      数据加载完成，开始执行分析模块[/bold green]")
    console.print("=" * 50)
    for mod in selected_modules:
        try:
            mod.run(loaded_runs)
        except Exception as e:
            console.print(f"[bold red]✗ 执行模块 '{mod.name}' 时发生严重错误: {e}[/bold red]")
            import traceback
            console.print(traceback.format_exc())


def _run_tool_workflow(tool_name: str, selected_dirs: List[str]):
    """执行预处理工具的工作流。"""
    if not selected_dirs:
        return

    if tool_name == 'slimmer':
        try:
            # 动态导入，避免如果不使用工具时的额外开销
            from analysis.tools import slimmer
        except ImportError:
            console.print("[red]错误: 无法导入 'slimmer' 工具。请确保 'analysis/tools/slimmer.py' 文件存在。[/red]")
            return

        # 所有的逻辑（UI、并行、确认、移动）都封装在这里面
        slimmer.run_interactive_workflow(selected_dirs)

    else:
        console.print(f"[red]错误: 未知的工具名称 '{tool_name}'[/red]")

def main():
    """主执行函数"""
    console.print("[bold inverse] WarpX 可扩展交互式分析框架 [/bold inverse]")
    setup_chinese_font()

    # 发现所有模块
    individual_modules, comparison_modules, video_modules = discover_modules()

    if not individual_modules and not comparison_modules:
        console.print("[red]错误: 在 'analysis/modules/' 目录下未找到任何有效的分析模块。程序退出。[/red]")
        return

    # 设置并解析命令行参数
    parser = argparse.ArgumentParser(description="WarpX 交互式分析框架")
    parser.add_argument(
        '--style',
        type=str,
        default=StyleTheme.PRESENTATION.name, # 默认值
        choices=[theme.name for theme in StyleTheme], # 从枚举自动生成选项
        help="选择绘图样式。"
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default='analysis_results',
        help="指定保存分析结果（图片、视频等）的目录。"
    )

    # --- 模式选择：使用互斥组确保一次只运行一种模式 ---
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        '-c', '--compare',
        action='store_true',
        help="进入对比分析模式。"
    )
    mode_group.add_argument(
        '-v', '--video',
        action='store_true',
        help="进入视频生成模式。"
    )
    mode_group.add_argument(
        '-t', '--tool',
        type=str,
        choices=['slimmer'],
        help="进入工具模式。可用工具: 'slimmer' (粒子数据瘦身)。"
    )

    args = parser.parse_args()

    # --- 应用选择的样式 ---
    selected_theme = StyleTheme[args.style] # 将字符串转换为枚举成员
    set_style(selected_theme)

    config.output_dir = args.output

    os.makedirs(args.output, exist_ok=True)
    console.print(f"[green]✔ 所有分析结果将保存到 '{args.output}/' 目录。[/green]")

    # 根据模式选择工作流
    if args.tool:
        console.print(f"\n[bold]--- 运行在 [magenta]工具[/magenta] 模式 ({args.tool}) ---[/bold]")
        selected_dirs = select_directories()
        _run_tool_workflow(args.tool, selected_dirs)
        # 工具模式下，工作流已完成，直接退出
        console.print("\n[bold]工具任务已完成。[/bold]")
        return

    elif args.video:
        console.print("\n[bold]--- 运行在 [magenta]视频生成[/magenta] 模式 ---[/bold]")
        if not video_modules:
            console.print("[red]错误: 没有可用的视频生成模块。[/red]")
            return

        selected_dirs = select_directories()
        if not selected_dirs: return
        selected_mods = _select_modules_from_list(video_modules)

    elif args.compare:
        console.print("\n[bold]--- 运行在 [magenta]对比分析[/magenta] 模式 ---[/bold]")
        if not comparison_modules:
            console.print("[red]错误: 没有可用的对比分析模块。请先创建对比模块。[/red]")
            return

        selected_dirs = select_directories()
        if not selected_dirs:
            return
        if len(selected_dirs) < 2:
            console.print("[yellow]警告: 对比分析至少需要选择两个模拟目录。操作已取消。[/yellow]")
            return

        selected_mods = _select_modules_from_list(comparison_modules)

    else:  # 默认进入单个分析模式
        console.print("\n[bold]--- 运行在 [magenta]独立分析[/magenta] 模式 (默认) ---[/bold]")
        if not individual_modules:
            console.print("[red]错误: 没有可用的独立分析模块。[/red]")
            if comparison_modules: console.print("[cyan]提示: 检测到对比模块，可使用 `-c` 标志运行。[/cyan]")
            if video_modules: console.print("[cyan]提示: 检测到视频模块，可使用 `-v` 标志运行。[/cyan]")
            return

        selected_dirs = select_directories()
        if not selected_dirs:
            return
        selected_mods = _select_modules_from_list(individual_modules)

    # 4. 执行工作流
    _run_analysis_workflow(selected_mods, selected_dirs)

    console.print("\n[bold]所有选定的分析任务已完成。[/bold]")


if __name__ == "__main__":
    main()
