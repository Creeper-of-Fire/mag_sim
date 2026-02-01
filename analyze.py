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
import importlib
import os
from pathlib import Path
from typing import List, Dict, Union

from rich.prompt import Prompt

from analysis.core.config import config
from analysis.core.data_loader import load_run_data
from analysis.core.selector import SimpleTableSelector
# --- 导入核心库组件 ---
from analysis.core.utils import console, select_directories
from analysis.utils import setup_chinese_font
from analysis.modules.abstract.base_module import BaseAnalysisModule, BaseComparisonModule, BaseVideoModule
from analysis.plotting.styles import StyleTheme, set_style

# 定义模块类型别名
AnyModule = Union[BaseAnalysisModule, BaseComparisonModule, BaseVideoModule]


def discover_modules() -> tuple[dict[str, BaseAnalysisModule], dict[str, BaseComparisonModule], dict[str, BaseVideoModule]]:
    """
    递归扫描 `modules` 文件夹及其子文件夹，加载并区分模块。
    """
    individual_modules = {}
    comparison_modules = {}
    video_modules = {}

    # 基础路径
    base_dir = Path(__file__).resolve().parent

    modules_dir = base_dir / "analysis/modules"

    # 使用 rglob("*.py") 进行递归查找
    for f in modules_dir.rglob("*.py"):
        # 过滤掉 __init__.py, base_ 开头的文件以及隐藏文件
        if f.name.startswith(('_', 'base_')):
            continue

        # 计算相对路径并转换为 python 模块路径格式
        # 例如: analysis/modules/subfolder/my_mod.py -> analysis.modules.subfolder.my_mod
        relative_path = f.relative_to(base_dir)
        module_name = ".".join(relative_path.with_suffix("").parts)

        try:
            module = importlib.import_module(module_name)
            for item_name in dir(module):
                item = getattr(module, item_name)

                # 避免加载基类自身
                if not isinstance(item, type) or item.__module__ != module_name:
                    continue

                # 检查是否是对应的基类子类
                if not issubclass(item, BaseAnalysisModule) or item in [BaseAnalysisModule, BaseComparisonModule, BaseVideoModule]:
                    continue

                instance = item()
                # 根据实例类型归类
                if isinstance(instance, BaseVideoModule):
                    video_modules[instance.name] = instance
                elif isinstance(instance, BaseComparisonModule):
                    comparison_modules[instance.name] = instance
                elif isinstance(instance, BaseAnalysisModule):
                    individual_modules[instance.name] = instance

        except Exception as e:
            # 这里的 console 是你原代码中引用的，请确保作用域内可用
            console.print(f"[red]加载模块 {module_name} 失败: {e}[/red]")

    # 返回排序后的字典
    return (
        dict(sorted(individual_modules.items())),
        dict(sorted(comparison_modules.items())),
        dict(sorted(video_modules.items()))
    )


def _select_modules_from_list(module_dict: Dict[str, AnyModule]) -> List[AnyModule]:
    """模块选择交互函数"""
    if not module_dict:
        return []

    # 将字典的值转为列表
    module_list = list(module_dict.values())

    # 定义显示格式：显示名称和描述
    def module_formatter(mod: AnyModule) -> str:
        return f"[magenta]{mod.name}[/magenta]\n    ↳ [dim]{mod.description}[/dim]"

    selector = SimpleTableSelector(
        items=module_list,
        columns=["模块名称", "描述"],
        row_converter=lambda mod: [
            f"[magenta]{mod.name}[/magenta]",
            f"↳ [dim]{mod.description}[/dim]"
        ],
        title="请选择分析模块"
    )

    return selector.select(default="all")

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


def _run_tool_workflow(tool_mode: str, selected_dirs: List[str]):
    """
    执行预处理工具的工作流。
    tool_mode: 可能是 'interactive', 'slimmer', 或 'pruner'
    """
    if not selected_dirs:
        return

    # --- 动态导入工具 ---
    tools_available = {}

    # 尝试导入 Slimmer
    try:
        from analysis.tools import slimmer
        tools_available['slimmer'] = slimmer.run_interactive_workflow
    except ImportError:
        pass  # 忽略缺失的工具

    # 尝试导入 Pruner
    try:
        from analysis.tools import pruner
        tools_available['pruner'] = pruner.run_pruner_interactive
    except ImportError:
        pass

    if not tools_available:
        console.print("[red]错误: 未找到任何可用工具 (slimmer/pruner)。请检查 analysis/tools 目录。[/red]")
        return

    # --- 交互式选择 (如果用户只输入了 -t 而没有指定名称) ---
    target_tool = tool_mode
    if tool_mode == 'interactive':
        console.print("\n[bold underline]可用工具列表：[/bold underline]")
        tool_names = list(tools_available.keys())

        # 显示菜单
        console.print(f"[[cyan]s[/cyan]] [magenta]slimmer[/magenta] (粒子数据压缩/瘦身)")
        console.print(f"[[cyan]p[/cyan]] [magenta]pruner[/magenta]  (仅保留首/中/尾时刻，删除其余)")
        console.print("")

        choice = Prompt.ask(
            "[bold]请选择工具[/bold]",
            choices=tool_names + ['s', 'p'],
            default='slimmer'
        )

        # 映射快捷键
        if choice == 's':
            target_tool = 'slimmer'
        elif choice == 'p':
            target_tool = 'pruner'
        else:
            target_tool = choice

    # --- 执行工具 ---
    if target_tool in tools_available:
        console.print(f"\n[bold green]>>> 正在启动 {target_tool} 工具...[/bold green]")
        tools_available[target_tool](selected_dirs)
    else:
        console.print(f"[red]错误: 未知的工具名称 '{target_tool}'[/red]")

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
        nargs='?',             # 表示参数是可选的 (0个或1个)
        const='interactive',   # 如果有 -t 但没给值，args.tool = 'interactive'
        type=str,
        help="进入工具模式。指定 'slimmer' 或 'pruner' 可直接运行，留空则进入交互菜单。"
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
