"""工作目录多选器 — 使用 Textual 内置 SelectionList"""
from __future__ import annotations

from pathlib import Path

from textual import on
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Input, SelectionList, Static

from analysis.core.utils import get_valid_simulation_runs, natural_sort_key
from analysis_tui.stores.analysis_store import analysis_store
from utils.project_config import PROJECT_ROOT


class DirPicker(Vertical):
    """目录选择面板"""

    CSS = """
    DirPicker {
        border: solid $border-primary;
        background: $bg-primary;
        height: 10;
    }

    #selected_dirs {
        height: auto;
        min-height: 1;
        padding: 0 1;
        color: $text-accent;
    }

    #dir_list {
        height: 1fr;
    }

    #dir_buttons {
        height: auto;
        padding: 0 1;
    }

    #dir_buttons Button {
        width: auto;
        margin-right: 1;
    }
    """

    DIVIDER = "─" * 8

    def compose(self):
        yield Static("— 分析目录", classes="panel_title")
        yield Static(id="selected_dirs")
        yield SelectionList(id="dir_list")
        with Horizontal(id="dir_buttons"):
            yield Button("刷新扫描", id="btn_scan", variant="primary")
            yield Button("手动添加", id="btn_add", variant="default")
            yield Button("清空", id="btn_clear", variant="default")

    def on_mount(self):
        self._refresh_list()
        self._update_selected_display()
        analysis_store.subscribe(self._on_store_changed)

    def on_unmount(self):
        analysis_store.unsubscribe(self._on_store_changed)

    # ── 事件 ──

    @on(SelectionList.SelectedChanged, "#dir_list")
    def _on_selection_changed(self):
        sel = self.query_one("#dir_list", SelectionList)
        current = set(analysis_store.selected_dir_paths)
        new = set(sel.selected)

        # 新增的
        for path in new - current:
            analysis_store.add_dir(path)
            analysis_store.load_run(path)
        # 移除的
        for path in current - new:
            analysis_store.remove_dir(path)
            analysis_store.evict_run(path)

        self._update_selected_display()

    @on(Button.Pressed, "#btn_scan")
    def _on_btn_scan(self):
        self._refresh_list()

    @on(Button.Pressed, "#btn_add")
    def _on_btn_add(self):
        self._show_input_dialog()

    @on(Button.Pressed, "#btn_clear")
    def _on_btn_clear(self):
        analysis_store.clear_dirs()
        self._refresh_list()
        self._update_selected_display()

    # ── 内部 ──

    def _get_available_paths(self) -> list[Path]:
        jobs_root = PROJECT_ROOT / "sim_jobs"
        if not jobs_root.exists():
            return []
        paths = []
        for job_dir in sorted(
            [d for d in jobs_root.iterdir() if d.is_dir()],
            key=lambda x: natural_sort_key(x.name)
        ):
            search_scope = job_dir / "sim_results"
            if not search_scope.exists():
                search_scope = job_dir
            paths.extend(get_valid_simulation_runs(search_scope))
        return paths

    def _refresh_list(self):
        sel = self.query_one("#dir_list", SelectionList)
        sel.clear_options()

        paths = self._get_available_paths()
        selected = set(analysis_store.selected_dir_paths)

        selections = []
        for p in paths:
            path_str = str(p)
            try:
                rel = str(p.relative_to(PROJECT_ROOT))
            except ValueError:
                rel = str(p)
            selections.append((rel, path_str, path_str in selected))

        if selections:
            sel.add_options(selections)

    def _update_selected_display(self):
        display = self.query_one("#selected_dirs", Static)
        dirs = analysis_store.selected_dir_paths
        if not dirs:
            display.update("[dim]未选择目录[/dim]")
            return
        names = []
        for p in dirs:
            path_obj = Path(p)
            parts = path_obj.parts
            if "sim_results" in parts:
                idx = parts.index("sim_results")
                name = "/".join(parts[idx - 1:idx + 2])
            else:
                name = path_obj.name
            names.append(name)
        display.update(" | ".join(f"[bold green]{n}[/bold green]" for n in names))

    def _show_input_dialog(self):
        from textual.screen import Screen

        class PathInputDialog(Screen):
            def compose(self):
                yield Static("输入模拟目录路径（绝对路径或相对于项目根目录）")
                yield Input(id="dlg_input", placeholder="/path/to/sim_dir")
                with Horizontal():
                    yield Button("确定", id="dlg_ok", variant="primary")
                    yield Button("取消", id="dlg_cancel", variant="default")

            @on(Button.Pressed, "#dlg_ok")
            def _ok(self):
                inp = self.query_one("#dlg_input", Input)
                path_str = inp.value.strip()
                if path_str:
                    path = Path(path_str)
                    if not path.is_absolute():
                        path = PROJECT_ROOT / path
                    if path.exists() and path.is_dir():
                        analysis_store.add_dir(str(path.resolve()))
                        analysis_store.load_run(str(path.resolve()))
                self.dismiss()

            @on(Button.Pressed, "#dlg_cancel")
            def _cancel(self):
                self.dismiss()

        self.app.push_screen(PathInputDialog())

    def _on_store_changed(self):
        self.call_after_refresh(self._update_selected_display)
