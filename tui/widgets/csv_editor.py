"""
CSV 表格编辑器
通过 ModalScreen + Input 弹窗编辑单元格
支持行增删
"""
import csv
from pathlib import Path

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, Horizontal, Container
from textual.message import Message
from textual.screen import ModalScreen
from textual.widgets import DataTable, Static, Button, Input


class CellEditScreen(ModalScreen[str | None]):
    """单元格编辑弹窗"""

    CSS = """
    CellEditScreen {
        align: center middle;
        background: $modal-overlay;  /* 半透明遮罩，背景可见 */
    }
    #cell_edit_dialog {
        width: 50;
        height: auto;
        background: $bg-secondary;
        border: solid $border-primary;
        padding: 1 2;
    }
    #cell_edit_dialog Static {
        color: $text-primary;
    }
    #cell_input {
        background: $bg-input;
        color: $text-accent;
        border: solid $border-primary;
        margin: 1 0;
    }
    """

    def __init__(self, current_value: str, column_name: str, type_hint: str = ""):
        super().__init__()
        self.current_value = current_value
        self.column_name = column_name
        self.type_hint = type_hint

    def compose(self) -> ComposeResult:
        yield Container(
            Static(
                f"编辑: {self.column_name} {self.type_hint}".strip(),
                classes="panel_title",
            ),
            Input(value=self.current_value, id="cell_input"),
            Static("Enter 确认  Esc 取消", classes="panel_title"),
            id="cell_edit_dialog",
        )

    def on_mount(self) -> None:
        inp = self.query_one("#cell_input", Input)
        inp.focus()
        inp.action_select_all()

    @on(Input.Submitted)
    def on_submit(self, event: Input.Submitted) -> None:
        self.dismiss(event.value)

    def key_escape(self) -> None:
        self.dismiss(None)


class CsvEditor(Vertical):
    """可编辑的 CSV 表格"""

    CSS = """
    CsvEditor {
        border: solid #0f3460;
        background: #1a1a2e;
    }
    #csv_table {
        height: 1fr;
    }
    """

    class CellChanged(Message):
        """单元格被编辑后发送"""

        def __init__(self, row: int, col: int, old_value: str, new_value: str):
            super().__init__()
            self.row = row
            self.col = col
            self.old_value = old_value
            self.new_value = new_value

    BINDINGS = [
        Binding("enter", "edit_current_cell", "编辑单元格", priority=True),
        Binding("a", "add_row", "添加行", priority=True),
        Binding("d", "delete_row", "删除行", priority=True),
    ]

    def __init__(self, schema: dict | None = None, **kwargs):
        super().__init__(**kwargs)
        self.schema = schema
        self._data: list[dict] = []
        self._column_order: list[str] = []

    def compose(self) -> ComposeResult:
        yield Static("📊 tasks.csv 编辑器", classes="panel_title")
        yield DataTable(cursor_type="cell", id="csv_table")
        with Horizontal(id="editor_buttons"):
            yield Button("添加行 (a)", id="btn_add_row", variant="primary")
            yield Button("删除行 (d)", id="btn_delete_row", variant="error")

    def on_mount(self) -> None:
        table = self.query_one("#csv_table", DataTable)
        table.focus()

    # ── 数据加载 ──

    def load_csv(self, path: Path) -> None:
        """从 CSV 文件加载数据"""
        table = self.query_one("#csv_table", DataTable)
        table.clear()
        self._data = []
        self._column_order = []

        if not path.exists():
            return

        with open(path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            self._column_order = list(reader.fieldnames or [])
            for row in reader:
                if any(v.strip() for v in row.values()):
                    self._data.append(row)

        if not self._column_order:
            return

        table.add_columns(*self._build_columns())
        for row_idx, row_data in enumerate(self._data):
            values = [row_data.get(col, "") for col in self._column_order]
            table.add_row(*values, key=str(row_idx))

    def _build_columns(self) -> list[str]:
        """构建列标签（含类型提示）"""
        hints = self._get_type_hints()
        columns = []
        for col in self._column_order:
            hint = hints.get(col, "")
            label = f"{col}\n{hint}" if hint else col
            columns.append(label)
        return columns

    def _get_type_hints(self) -> dict:
        """从 schema 获取类型提示"""
        if not self.schema:
            return {}
        hints = {}
        for p in self.schema.get("params", []):
            t = p["type"]
            if t in ("bool", "int", "float"):
                hints[p["name"]] = f"({t})"
        return hints

    def _get_default_for(self, col_name: str):
        """从 schema 获取某列的默认值"""
        if not self.schema:
            return None
        for p in self.schema.get("params", []):
            if p["name"] == col_name:
                return p.get("default")
        return None

    # ── 编辑单元格 ──

    def action_edit_current_cell(self) -> None:
        """打开弹窗编辑当前单元格"""
        table = self.query_one("#csv_table", DataTable)
        if table.cursor_coordinate is None:
            return
        row, col = table.cursor_coordinate.row, table.cursor_coordinate.column
        if row >= len(self._data) or col >= len(self._column_order):
            return

        col_name = self._column_order[col]
        current = self._data[row].get(col_name, "")
        hints = self._get_type_hints()
        hint = hints.get(col_name, "")

        self.app.push_screen(
            CellEditScreen(str(current), col_name, hint),
            callback=lambda new_val: self._on_cell_edited(row, col, new_val),
        )

    def _on_cell_edited(self, row: int, col: int, new_value: str | None) -> None:
        """编辑弹窗回调"""
        if new_value is None:
            return
        col_name = self._column_order[col]
        old = self._data[row].get(col_name, "")
        self._data[row][col_name] = new_value

        table = self.query_one("#csv_table", DataTable)
        # 用坐标获取真正的 RowKey 和 ColumnKey
        from textual.coordinate import Coordinate
        row_key, column_key = table.coordinate_to_cell_key(Coordinate(row, col))
        table.update_cell(row_key, column_key, new_value)

        self.post_message(self.CellChanged(row, col, old, new_value))

    # ── 添加行 ──

    def action_add_row(self) -> None:
        """添加一行"""
        table = self.query_one("#csv_table", DataTable)
        task_col = self._column_order[0] if self._column_order else "task_name"

        new_row = {}
        for col in self._column_order:
            if col == task_col:
                new_row[col] = f"task_{len(self._data) + 1}"
            else:
                default = self._get_default_for(col)
                new_row[col] = str(default) if default is not None else ""

        self._data.append(new_row)
        row_key = str(len(self._data) - 1)
        values = [new_row.get(col, "") for col in self._column_order]
        table.add_row(*values, key=row_key)

    # ── 删除行 ──

    def action_delete_row(self) -> None:
        """删除当前行"""
        table = self.query_one("#csv_table", DataTable)
        if table.cursor_coordinate is None:
            return
        row_key, _ = table.coordinate_to_cell_key(table.cursor_coordinate)
        if row_key is None:
            return

        row_idx = int(row_key.value)
        if row_idx >= len(self._data):
            return

        del self._data[row_idx]
        self._refresh_table()

    def _refresh_table(self) -> None:
        """完全重建表格"""
        table = self.query_one("#csv_table", DataTable)
        table.clear()
        if not self._column_order:
            return
        table.add_columns(*self._build_columns())
        for row_idx, row_data in enumerate(self._data):
            values = [row_data.get(col, "") for col in self._column_order]
            table.add_row(*values, key=str(row_idx))

    # ── 保存 ──

    def save_csv(self, path: Path) -> None:
        """保存到 CSV 文件"""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=self._column_order)
            writer.writeheader()
            writer.writerows(self._data)

    # ── 按钮事件 ──

    @on(Button.Pressed, "#btn_add_row")
    def on_btn_add_row(self) -> None:
        self.action_add_row()

    @on(Button.Pressed, "#btn_delete_row")
    def on_btn_delete_row(self) -> None:
        self.action_delete_row()
