"""主分析界面 — 组合 DirPicker + ModulePicker + ConsoleWidget + 执行"""
import asyncio
import concurrent.futures

from textual import on
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Header, Footer, Button

from analysis.core.executor import execute_analysis
from analysis_tui.stores.analysis_store import analysis_store
from analysis_tui.widgets.console_widget import ConsoleWidget
from analysis_tui.widgets.dir_picker import DirPicker
from analysis_tui.widgets.module_picker import ModulePicker


class AnalysisScreen(Screen):
    """分析主界面"""

    AUTO_FOCUS = None

    CSS = """
    #left_panel {
        width: 30%;
    }

    #console_panel {
        width: 70%;
    }

    #run_buttons {
        height: auto;
        dock: bottom;
        align: center middle;
        padding: 0 1;
    }

    #run_buttons Button {
        height: 3;
        width: 1fr;
        margin: 0 1;
    }
    """

    BINDINGS = [
        Binding("F5,f5", "run_analysis", "运行分析", priority=True),
        Binding("R,r", "refresh_modules", "刷新模块", priority=True),
    ]

    def compose(self):
        yield Header()

        yield DirPicker(id="dir_picker")

        with Horizontal():
            with Vertical(id="left_panel"):
                yield ModulePicker(id="module_picker")

            with Vertical(id="console_panel"):
                yield ConsoleWidget(id="console_widget")

        with Horizontal(id="run_buttons"):
            yield Button("▶ 运行分析", id="btn_run", variant="success")
            yield Button("🔄 刷新模块", id="btn_refresh", variant="default")

        yield Footer()

    def on_mount(self):
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        analysis_store.subscribe(self._on_store_changed)

    def on_unmount(self):
        analysis_store.unsubscribe(self._on_store_changed)
        self._executor.shutdown(wait=False)

    # ── 按钮事件 ──

    @on(Button.Pressed, "#btn_run")
    def _on_btn_run(self):
        self.action_run_analysis()

    @on(Button.Pressed, "#btn_refresh")
    def _on_btn_refresh(self):
        self.action_refresh_modules()

    # ── 动作 ──

    def action_run_analysis(self):
        if analysis_store.is_running:
            return

        modules = analysis_store.get_selected_modules()
        runs = analysis_store.get_loaded_runs()

        if not modules:
            console = self.query_one("#console_widget", ConsoleWidget)
            console.write("[yellow]请先选择分析模块。[/yellow]")
            return
        if not runs:
            console = self.query_one("#console_widget", ConsoleWidget)
            console.write("[yellow]请先选择工作目录。[/yellow]")
            return

        analysis_store.is_running = True
        self._set_buttons_disabled(True)

        console = self.query_one("#console_widget", ConsoleWidget)
        console.clear()
        console.write("[bold inverse] 开始分析 [/bold inverse]")
        console.write(f"  目录: [green]{len(runs)}[/green] 个")
        console.write(f"  模块: [green]{len(modules)}[/green] 个")
        console.write("─" * 50)

        # 在线程池中运行，stdout 重定向到控制台
        def _run():
            try:
                with console.redirect_stdout():
                    errors = execute_analysis(modules, runs, output=console.write)
                if errors:
                    console.write(f"\n[bold red]✗ {len(errors)} 个模块出错[/bold red]")
                else:
                    console.write("\n[bold green]✓ 全部完成[/bold green]")
            finally:
                self.app.call_from_thread(self._on_analysis_done)

        asyncio.get_event_loop().run_in_executor(self._executor, _run)

    def action_refresh_modules(self):
        analysis_store.individual_modules.clear()
        analysis_store.comparison_modules.clear()
        analysis_store.video_modules.clear()
        analysis_store.selected_module_names.clear()
        analysis_store.discover_modules()
        module_picker = self.query_one("#module_picker", ModulePicker)
        module_picker._refresh_list()

    # ── 内部 ──

    def _on_analysis_done(self):
        """分析执行完成（在 UI 线程中调用）"""
        analysis_store.is_running = False
        self._set_buttons_disabled(False)

    def _on_store_changed(self):
        pass

    def _set_buttons_disabled(self, disabled: bool):
        try:
            self.query_one("#btn_run", Button).disabled = disabled
        except Exception:
            pass
