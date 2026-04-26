import abc
from typing import List, TypeVar, Generic, Optional, Callable

from rich import box
from rich.console import Console
from rich.prompt import Prompt
from rich.table import Table

console = Console()
T = TypeVar('T')


class BaseSelector(Generic[T], abc.ABC):
    """
    交互式选择器抽象基类 (Template Method Pattern)。

    职责：
    1. 管理待选数据 (items)
    2. 处理输入解析、验证、默认值 (Control Logic)
    3. 强制子类实现展示逻辑 (View Logic)
    """

    def __init__(self, items: List[T], title: str = "请选择"):
        self.items = items
        self.title = title

    @abc.abstractmethod
    def render_menu(self) -> None:
        """
        [抽象方法] 子类必须实现此方法来绘制菜单（列表或表格）。
        不需要处理输入，只需要 print 出带索引的内容即可。
        """
        pass

    def _parse_indices(self, input_str: str) -> List[int]:
        """内部工具：解析索引字符串 (支持 1-5, 7)"""
        input_str = input_str.strip().lower()
        if not input_str:
            return []

        # 1. 处理 'all' 关键字
        if input_str == 'all':
            return list(range(len(self.items)))

        indices = set()
        parts = input_str.replace('，', ',').replace(',', ' ').split()
        max_idx = len(self.items)

        for part in parts:
            if '-' in part:
                try:
                    start_s, end_s = part.split('-', 1)
                    start, end = int(start_s), int(end_s)
                    if start > end: start, end = end, start
                    # 包含 end
                    indices.update(range(start, end + 1))
                except ValueError:
                    raise ValueError(f"无效范围: {part}")
            else:
                try:
                    indices.add(int(part))
                except ValueError:
                    raise ValueError(f"无效数字: {part}")

        result = sorted(list(indices))

        # 越界检查
        if result and (result[0] < 0 or result[-1] >= max_idx):
            raise IndexError(f"索引超出范围 (0-{max_idx - 1})")

        return result

    def select(self,
               default: Optional[str] = "all",
               prompt_text: str = "请输入索引",
               single: bool = False) -> List[T]:
        """
        主流程方法。

        Args:
            default: 默认值
            prompt_text: 提示语
            single (bool): 如果为 True，则强制单选模式，此时default = "all" 自动无效。
        """
        if not self.items:
            console.print(f"[yellow]{self.title}: 列表为空，无法选择。[/yellow]")
            return []

        # 1. 渲染菜单 (由子类定义样子)
        self.render_menu()

        # 2. 交互循环
        if single:
            help_text = "(请输入单个数字)"
        else:
            help_text = "(支持范围 e.g. 1-3)"

        prompt_full = f"[bold]{prompt_text} {help_text}[/bold]"

        while True:
            choice = Prompt.ask(prompt_full, default=default)
            if choice is None: choice = ""  # 防御性
            choice = choice.strip()

            try:
                # 解析输入
                indices = self._parse_indices(choice)

                # 如果输入为空且 default 是 None (或者被 Rich 处理了)
                if not indices:
                    console.print("[yellow]⚠ 未选择任何内容。[/yellow]")
                    continue

                # 如果是单选模式，但解析出了多个索引（比如输入了 1-3 或 all）
                if single and len(indices) > 1:
                    console.print(f"[red]错误: 当前处于单选模式，但你的输入包含了 {len(indices)} 个项目。请只选择一个。[/red]")
                    continue

                selected = [self.items[i] for i in indices]
                console.print(f"[green]✔ 已选择 {len(selected)} 项。[/green]")
                return selected

            except (ValueError, IndexError) as e:
                console.print(f"[red]输入错误: {e}[/red]")


class SimpleTableSelector(BaseSelector[T]):
    """
    通用实现：适用于简单对象，只需传入一个格式化函数即可。
    替代之前的 InteractiveSelector。
    """

    def __init__(self, items: List[T],
                 columns: List[str],
                 row_converter: Callable,
                 title: str = "请选择"):
        super().__init__(items, title)
        self.columns = columns
        self.row_converter = row_converter

    def render_menu(self) -> None:
        console.print(f"\n[bold underline]{self.title}[/bold underline]")

        # 自动构建 Table
        table = Table(box=box.SIMPLE_HEAD)
        table.add_column("ID", justify="right", style="cyan", no_wrap=True)
        for col in self.columns:
            table.add_column(col)

        for i, item in enumerate(self.items):
            # 用户转换函数返回一个列表，对应 columns
            row_data = self.row_converter(item)
            # 确保全是 string
            row_str = [str(x) for x in row_data]
            table.add_row(str(i), *row_str)

        console.print(table)
