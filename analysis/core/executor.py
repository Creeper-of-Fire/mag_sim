"""
分析执行器。

纯执行逻辑，不涉及模块发现、数据加载或交互式选择。
调用者负责传入已加载好的模块实例和 SimulationRun 引用。
"""
from typing import Callable, List, TYPE_CHECKING, Union

if TYPE_CHECKING:
    from analysis.modules.abstract.base_module import (
        BaseAnalysisModule,
        BaseComparisonModule,
        BaseVideoModule,
    )
    from analysis.core.simulation import SimulationRun

AnyModule = Union["BaseAnalysisModule", "BaseComparisonModule", "BaseVideoModule"]


def execute_analysis(
    modules: List[AnyModule],
    runs: List["SimulationRun"],
    output: Callable[[str], None] | None = None,
) -> list[str]:
    """
    对已加载的数据运行一组已发现的分析模块。

    Args:
        modules: 已发现的模块实例列表（按调用顺序）。
        runs: 已加载的 SimulationRun 对象列表。
        output: 可选的状态消息回调。若为 None，则静默执行。

    Returns:
        错误消息列表。若为空则全部成功。
    """
    if not modules:
        return ["未选择任何分析模块。"]
    if not runs:
        return ["未提供任何模拟数据。"]

    errors: list[str] = []

    for mod in modules:
        try:
            mod.run(runs)
        except Exception as e:
            import traceback
            msg = f"执行模块 '{mod.name}' 时出错: {e}"
            errors.append(msg)
            if output:
                output(f"[bold red]✗ {msg}[/bold red]")
                output(traceback.format_exc())

    return errors
