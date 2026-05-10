"""工具工作流 — 交互式工具选择与执行。"""

from typing import List

from rich.prompt import Prompt

from analysis.core.utils import console


def run_tool_workflow(tool_mode: str, selected_dirs: List[str]):
    """
    执行预处理工具的工作流。
    tool_mode: 可能是 'interactive', 'slimmer', 'pruner', 'step_filter'
    """
    if not selected_dirs:
        return

    tools_available = {}

    try:
        from analysis.tools import slicer
        tools_available['slicer'] = slicer.run_interactive_workflow
    except ImportError:
        pass

    try:
        from analysis.tools import slimmer
        tools_available['slimmer'] = slimmer.run_interactive_workflow
    except ImportError:
        pass

    try:
        from analysis.tools import pruner
        tools_available['pruner'] = pruner.run_pruner_interactive
    except ImportError:
        pass

    try:
        from analysis.tools import step_filter
        tools_available['step_filter'] = step_filter.run_interactive_workflow
    except ImportError:
        pass

    if not tools_available:
        console.print("[red]错误: 未找到任何可用工具。请检查 analysis/tools 目录。[/red]")
        return

    target_tool = tool_mode
    if tool_mode == 'interactive':
        console.print("\n[bold underline]可用工具列表：[/bold underline]")
        console.print(f"[[cyan]s[/cyan]] [magenta]slicer[/magenta]  (虚拟切片：将一个模拟的各时刻拆分为多重模拟，零存储开销)")
        console.print(f"[[cyan]m[/cyan]] [magenta]slimmer[/magenta] (粒子数据压缩/瘦身)")
        console.print(f"[[cyan]p[/cyan]] [magenta]pruner[/magenta]  (仅保留首/中/尾时刻，删除其余)")
        console.print(f"[[cyan]f[/cyan]] [magenta]step_filter[/magenta] (非破坏性时间步过滤：禁用/恢复指定时间步)")
        console.print("")

        choice = Prompt.ask(
            "[bold]请选择工具[/bold]",
            choices=list(tools_available.keys()) + ['s', 'm', 'p', 'f'],
            default='slimmer'
        )

        shortcut_map = {'s': 'slicer', 'm': 'slimmer', 'p': 'pruner', 'f': 'step_filter'}
        target_tool = shortcut_map.get(choice, choice)

    if target_tool in tools_available:
        console.print(f"\n[bold green]>>> 正在启动 {target_tool} 工具...[/bold green]")
        tools_available[target_tool](selected_dirs)
    else:
        console.print(f"[red]错误: 未知的工具名称 '{target_tool}'[/red]")
