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
import importlib
from pathlib import Path
from typing import Dict

from rich.prompt import Prompt

# --- 导入核心库组件 ---
from simNew.analysis.core.utils import console, setup_chinese_font, select_directories
from simNew.analysis.core.data_loader import load_run_data
from simNew.analysis.modules.base_module import BaseAnalysisModule


def discover_modules() -> Dict[str, BaseAnalysisModule]:
    """
    动态扫描 `modules` 文件夹，加载所有合法的分析模块。
    """
    modules = {}
    modules_dir = Path(__file__).parent / "modules"

    for f in modules_dir.glob("*.py"):
        if f.name.startswith(('_', 'base_')):
            continue

        module_name = f"modules.{f.stem}"
        try:
            module = importlib.import_module(module_name)
            for item_name in dir(module):
                item = getattr(module, item_name)
                if isinstance(item, type) and issubclass(item, BaseAnalysisModule) and item is not BaseAnalysisModule:
                    # 找到了一个模块类，实例化它
                    instance = item()
                    # 使用模块名称作为键，避免重名
                    modules[instance.name] = instance
        except Exception as e:
            console.print(f"[red]加载模块 {module_name} 失败: {e}[/red]")

    # 按名称排序，保证菜单顺序稳定
    return dict(sorted(modules.items()))


def main():
    """主执行函数"""
    console.print("[bold inverse] WarpX 可扩展交互式分析框架 [/bold inverse]")
    setup_chinese_font()

    # 1. 发现并展示可用的分析模块
    analysis_modules = discover_modules()
    if not analysis_modules:
        console.print("[red]错误: 在 'modules/' 目录下未找到任何有效的分析模块。[/red]")
        return

    module_list = list(analysis_modules.values())
    # 使用简单的打印代替表格，以获得更好的兼容性
    console.print("\n[bold underline]可用的分析模块：[/bold underline]\n")
    for i, mod in enumerate(module_list):
        # 打印 " [索引] 模块名称 "
        console.print(f"[[cyan]{i}[/cyan]] [magenta]{mod.name}[/magenta]")
        # 缩进打印 "   ↳ 功能描述 "
        console.print(f"    ↳ [white]{mod.description}[/white]")
    console.print("")

    # 2. 让用户选择要执行的分析
    selected_modules = []
    while True:
        try:
            prompt_text = "[bold]请输入要执行的分析索引 (用逗号/空格分隔, [cyan]直接回车则全选[/cyan]): [/bold]"
            choice_str = Prompt.ask(prompt_text, default="all")

            if choice_str.strip().lower() == "all":
                selected_modules = module_list
                console.print(f"[green]✔ 已选择全部 {len(selected_modules)} 个分析模块。[/green]")
                break

            # 如果不是 'all'，则处理输入的索引
            indices_str = choice_str.replace(',', ' ').split()
            if not indices_str:  # 如果用户输入了空格然后回车
                continue

            indices = [int(i) for i in indices_str]
            if all(0 <= i < len(module_list) for i in indices):
                selected_modules = [module_list[i] for i in indices]
                break
            else:
                console.print("[yellow]警告: 输入的索引超出范围，请重试。[/yellow]")
        except ValueError:
            console.print("[red]错误: 无效输入，请输入数字索引。[/red]")

    # 3. 选择要分析的模拟目录
    selected_dirs = select_directories()
    if not selected_dirs:
        console.print("\n[yellow]未选择任何目录，程序退出。[/yellow]")
        return

    # 4. 智能数据加载
    # 汇总所有需要的数据类型
    all_required_data = set()
    for mod in selected_modules:
        all_required_data.update(mod.required_data)

    console.print(f"\n[bold]将为分析加载以下数据类型: {all_required_data}[/bold]")

    loaded_runs = []
    for dir_path in selected_dirs:
        run_data = load_run_data(dir_path, all_required_data)
        if run_data:
            loaded_runs.append(run_data)

    if not loaded_runs:
        console.print("\n[red]未能成功加载任何模拟数据，无法继续分析。[/red]")
        return

    # 5. 依次执行所选的分析模块
    console.print("\n" + "=" * 50)
    console.print("[bold green]      数据加载完成，开始执行分析模块[/bold green]")
    console.print("=" * 50)
    for mod in selected_modules:
        try:
            mod.run(loaded_runs)
        except Exception as e:
            console.print(f"[bold red]✗ 执行模块 '{mod.name}' 时发生严重错误: {e}[/bold red]")
            # 可以在这里添加更详细的错误追溯信息
            import traceback
            console.print(traceback.format_exc())

    console.print("\n[bold]所有选定的分析任务已完成。[/bold]")


if __name__ == "__main__":
    main()