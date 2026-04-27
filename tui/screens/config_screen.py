"""
配置编辑 Screen - 编辑脚本名、额外参数和 tasks.csv
"""

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, Horizontal
from textual.screen import Screen
from textual.widgets import Header, Footer, Static, Input, Button, Label

from tui.controllers.csv_tool import CsvToolRunner
from tui.store.app_store import app_store
from tui.store.config_store import config_store
from tui.store.log_store import logger
from tui.widgets.csv_editor import CsvEditor
from utils.project_config import FILENAME_TASKS_CSV


class ConfigScreen(Screen):
    """项目配置编辑界面"""

    CSS = """
    #config_container {
        padding: 0 1;
        height: 1fr;
    }
    #script_config {
        height: auto;
    }
    .config_row {
        height: 3;
        align: left middle;
    }
    .config_row Label {
        width: 10;
        color: #888888;
    }
    .config_row Input {
        width: 1fr;
        background: #1a1a2e;
        color: #00ff88;
        border: solid #0f3460;
    }
    #csv_editor {
        height: auto;
    }
    """

    BINDINGS = [
        Binding("escape", "escape_handler", "取消聚焦/保存返回", priority=True),
        Binding("s", "save_all", "保存全部", priority=True),
        Binding("c", "save_all", "保存全部", priority=True),
        Binding("r", "regen_template", "重新生成模板", priority=True),
    ]

    def __init__(self):
        super().__init__()
        self.job_dir = app_store.job_dir
        self.config = config_store.load()
        self.csv_tool = CsvToolRunner()
        self._schema = None

    def compose(self) -> ComposeResult:
        yield Header()

        # 脚本配置
        with Vertical(id="config_container"):
            yield Static(f"🛠 项目配置 - {self.job_dir.name}", classes="panel_title")
            with Vertical(id="script_config"):
                with Horizontal(classes="config_row"):
                    yield Label("脚本:")
                    yield Input(value="", id="input_script", placeholder="如 csv_tool.py")
                with Horizontal(classes="config_row"):
                    yield Label("额外参数:")
                    yield Input(value="", id="input_extra_args", placeholder="如 --flag value")

            # 按钮行
            with Horizontal(id="config_buttons"):
                yield Button("重新生成模板", id="btn_regen", variant="warning")
                yield Button("保存全部", id="btn_save", variant="success")
                yield Button("保存并返回", id="btn_save_back", variant="primary")

            # CSV 编辑器
            yield CsvEditor(id="csv_editor")

        yield Footer()

    def on_mount(self) -> None:
        """加载 schema 和 CSV 数据"""
        config = config_store.load()
        self.query_one("#input_script", Input).value = config.script_name
        self.query_one("#input_extra_args", Input).value = config.extra_args

        self._load_schema()
        self._load_csv()

        csv_editor = self.query_one("#csv_editor", CsvEditor)
        csv_editor.styles.align = ("center", "top")
        csv_editor.refresh(layout=True)

    def _load_schema(self) -> None:
        """从 csv_tool 获取参数元信息"""
        self._schema = self.csv_tool.get_schema(self.config.script_name)
        if self._schema:
            editor = self.query_one("#csv_editor", CsvEditor)
            editor.schema = self._schema
        else:
            logger.warn("警告: 无法获取参数 schema，编辑器将无类型提示。")

    def _load_csv(self) -> None:
        """加载 tasks.csv 到编辑器"""
        csv_path = self.job_dir / FILENAME_TASKS_CSV
        editor = self.query_one("#csv_editor", CsvEditor)
        if csv_path.exists():
            editor.load_csv(csv_path)
        else:
            logger.warn(f"提示: {FILENAME_TASKS_CSV} 不存在，请先生成模板。")

    def action_save_all(self) -> None:
        """保存配置和 CSV"""
        # 保存脚本配置
        script = self.query_one("#input_script", Input).value.strip()
        extra = self.query_one("#input_extra_args", Input).value.strip()
        self.config.script_name = script
        self.config.extra_args = extra
        config_store.save(self.config)

        # 保存 CSV
        csv_path = self.job_dir / FILENAME_TASKS_CSV
        editor = self.query_one("#csv_editor", CsvEditor)
        editor.save_csv(csv_path)

        logger.info("✅ 配置和 CSV 已保存。")

    def action_escape_handler(self) -> None:
        """第一次取消聚焦，第二次保存并返回"""
        focused = self.focused
        if focused is not None and not isinstance(focused, Screen):
            self.set_focus(None)
        else:
            self.action_save_and_back()

    def action_save_and_back(self) -> None:
        """保存并返回主界面"""
        self.action_save_all()
        self.app.pop_screen()

    def action_regen_template(self) -> None:
        """调用 csv_tool 重新生成 CSV 模板"""
        if self.csv_tool.generate_template(self.job_dir, self.config.script_name):
            logger.info("✅ 模板已重新生成。")
            self._load_csv()
        else:
            logger.error("❌ 模板生成失败。")

    @on(Button.Pressed, "#btn_regen")
    def on_btn_regen(self) -> None:
        self.action_regen_template()

    @on(Button.Pressed, "#btn_save")
    def on_btn_save(self) -> None:
        self.action_save_all()

    @on(Button.Pressed, "#btn_save_back")
    def on_btn_save_back(self) -> None:
        self.action_save_and_back()
