"""模块发现 — 递归扫描 analysis/modules/ 下的所有分析模块。"""

import importlib
import inspect
from pathlib import Path
from typing import Union

from analysis.core.utils import console
from analysis.modules.abstract.base_module import BaseAnalysisModule, BaseComparisonModule, BaseVideoModule

AnyModule = Union[BaseAnalysisModule, BaseComparisonModule, BaseVideoModule]


def discover_modules() -> tuple[dict[str, BaseAnalysisModule], dict[str, BaseComparisonModule], dict[str, BaseVideoModule]]:
    """递归扫描 `modules` 文件夹及其子文件夹，加载并区分模块。"""
    individual_modules = {}
    comparison_modules = {}
    video_modules = {}

    base_dir = Path(__file__).resolve().parent.parent  # analysis_cli/ -> 项目根
    modules_dir = base_dir / "analysis" / "modules"

    base_types = (BaseAnalysisModule, BaseComparisonModule, BaseVideoModule)

    for f in modules_dir.rglob("*.py"):
        if f.name.startswith(('_', 'base_')):
            continue

        relative_path = f.relative_to(base_dir)
        module_name = ".".join(relative_path.with_suffix("").parts)

        try:
            module = importlib.import_module(module_name)

            for item_name, item in inspect.getmembers(module, inspect.isclass):
                if item.__module__ != module_name:
                    continue
                if item in base_types:
                    continue

                if issubclass(item, BaseVideoModule):
                    instance = item()
                    video_modules[instance.name] = instance
                elif issubclass(item, BaseComparisonModule):
                    instance = item()
                    comparison_modules[instance.name] = instance
                elif issubclass(item, BaseAnalysisModule):
                    instance = item()
                    individual_modules[instance.name] = instance

        except Exception as e:
            console.print(f"[red]加载模块 {module_name} 失败: {e}[/red]")

    console.print("已加载的模块：")

    return (
        dict(sorted(individual_modules.items())),
        dict(sorted(comparison_modules.items())),
        dict(sorted(video_modules.items()))
    )
