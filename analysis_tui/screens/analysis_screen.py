"""分析主界面 — ModulePicker + pyte 终端 + Rich 目录选择"""
import asyncio
import concurrent.futures
import queue
import sys

from textual import on
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Header, Footer, Button, Static

from analysis.core.executor import execute_analysis
from analysis.core.utils import console as rich_console, select_directories
from analysis_tui.stores.analysis_store import analysis_store
from analysis_tui.widgets.module_picker import ModulePicker
from analysis_tui.widgets.terminal_widget import TerminalWidget


class _TuiStdin:
    def __init__(self, q: queue.Queue):
        self._q = q

    def readline(self) -> str:
        return self._q.get() + '\n'

    def read(self, n: int = -1) -> str:
        return self.readline()

    def isatty(self) -> bool:
        return False

    def flush(self):
        pass


class AnalysisScreen(Screen):
    """分析主界面"""

    AUTO_FOCUS = None

    CSS = """
    #left_panel {
        width: 26;
    }

    #term_panel {
        width: 1fr;
    }

    #run_row {
        height: auto;
        padding: 0 1;
    }

    #run_row Button {
        width: auto;
        margin-right: 1;
    }

    #dir_status {
        height: auto;
        padding: 0 1;
        color: $text-accent;
    }
    """

    BINDINGS = [
        Binding("F5,f5", "run_analysis", "运行分析", priority=True),
        Binding("D,d", "select_dirs", "选择目录", priority=True),
        Binding("R,r", "refresh_modules", "刷新模块", priority=True),
    ]

    def compose(self):
        yield Header()

        yield Static(id="dir_status")

        with Horizontal():
            with Vertical(id="left_panel"):
                yield ModulePicker(id="module_picker")

            with Vertical(id="term_panel"):
                yield TerminalWidget(id="terminal")

        with Horizontal(id="run_row"):
            yield Button("📂 选择目录", id="btn_dirs", variant="default")
            yield Button("▶ 运行分析", id="btn_run", variant="success")
            yield Button("🔄 刷新模块", id="btn_refresh", variant="default")

        yield Footer()

    def on_mount(self):
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self._stdin_queue: queue.Queue = queue.Queue()
        # 把 stdin 队列注入终端，终端键盘输入直接推入
        term = self.query_one("#terminal", TerminalWidget)
        term.stdin_queue = self._stdin_queue
        term.focus()
        analysis_store.subscribe(self._on_store_changed)
        self._update_dir_status()

    def on_unmount(self):
        analysis_store.unsubscribe(self._on_store_changed)
        self._executor.shutdown(wait=False)

    # ── 按钮 ──

    @on(Button.Pressed, "#btn_dirs")
    def _on_btn_dirs(self):
        self.action_select_dirs()

    @on(Button.Pressed, "#btn_run")
    def _on_btn_run(self):
        self.action_run_analysis()

    @on(Button.Pressed, "#btn_refresh")
    def _on_btn_refresh(self):
        self.action_refresh_modules()

    # ── 动作 ──

    def action_select_dirs(self):
        """在终端中运行旧 Rich select_directories()"""
        term = self.query_one("#terminal", TerminalWidget)
        term.clear_screen()
        tui_stdin = _TuiStdin(self._stdin_queue)

        def _run_select():
            original_stdout = sys.stdout
            original_stdin = sys.stdin
            original_console_file = rich_console.file
            try:
                sys.stdout = term
                sys.stdin = tui_stdin
                rich_console.file = term
                dirs = select_directories()
                if dirs:
                    for d in dirs:
                        analysis_store.add_dir(d)
                        analysis_store.load_run(d)
                self.app.call_from_thread(self._on_dirs_selected)
            except Exception:
                import traceback
                term.feed(f"\r\n{traceback.format_exc()}\r\n")
            finally:
                sys.stdout = original_stdout
                sys.stdin = original_stdin
                rich_console.file = original_console_file

        asyncio.get_event_loop().run_in_executor(self._executor, _run_select)

    def action_run_analysis(self):
        if analysis_store.is_running:
            return

        modules = analysis_store.get_selected_modules()
        runs = analysis_store.get_loaded_runs()

        if not runs:
            self.action_select_dirs()
            return

        if not modules:
            term = self.query_one("#terminal", TerminalWidget)
            term.feed("\r\n[yellow]请先选择分析模块。[/yellow]\r\n")
            return

        analysis_store.is_running = True
        self._set_buttons_disabled(True)

        term = self.query_one("#terminal", TerminalWidget)
        term.clear_screen()
        term.feed(f"目录: {len(runs)} 个 | 模块: {len(modules)} 个\r\n")
        term.feed("─" * 50 + "\r\n")

        tui_stdin = _TuiStdin(self._stdin_queue)

        def _run():
            original_stdout = sys.stdout
            original_stdin = sys.stdin
            original_console_file = rich_console.file
            try:
                sys.stdout = term
                sys.stdin = tui_stdin
                rich_console.file = term
                errors = execute_analysis(modules, runs, output=lambda s: term.feed(s + '\r\n'))
                if errors:
                    term.feed(f"\r\n✗ {len(errors)} 个模块出错\r\n")
                else:
                    term.feed("\r\n✓ 全部完成\r\n")
            except Exception:
                import traceback
                term.feed(f"\r\n{traceback.format_exc()}\r\n")
            finally:
                sys.stdout = original_stdout
                sys.stdin = original_stdin
                rich_console.file = original_console_file
                self.app.call_from_thread(self._on_analysis_done)

        asyncio.get_event_loop().run_in_executor(self._executor, _run)

    def action_refresh_modules(self):
        analysis_store.individual_modules.clear()
        analysis_store.comparison_modules.clear()
        analysis_store.video_modules.clear()
        analysis_store.selected_module_names.clear()
        analysis_store.discover_modules()
        picker = self.query_one("#module_picker", ModulePicker)
        picker._refresh_all()

    # ── 回调 ──

    def _on_dirs_selected(self):
        self._update_dir_status()

    def _on_analysis_done(self):
        analysis_store.is_running = False
        self._set_buttons_disabled(False)

    def _on_store_changed(self):
        pass

    # ── 内部 ──

    def _update_dir_status(self):
        display = self.query_one("#dir_status", Static)
        dirs = analysis_store.selected_dir_paths
        if not dirs:
            display.update("[dim]未选择目录 — 按 D 或点击「选择目录」[/dim]")
            return
        from pathlib import Path
        names = [Path(p).name for p in dirs]
        display.update(f"[bold green]已选 {len(dirs)} 个目录:[/bold green] {', '.join(names)}")

    def _set_buttons_disabled(self, disabled: bool):
        try:
            self.query_one("#btn_run", Button).disabled = disabled
            self.query_one("#btn_dirs", Button).disabled = disabled
        except Exception:
            pass
