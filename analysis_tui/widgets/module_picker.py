"""模块多选器 — 使用 Textual 内置 SelectionList，按分类分组"""
from __future__ import annotations

from typing import TYPE_CHECKING

from textual import on
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Button, SelectionList, Static

from analysis_tui.stores.analysis_store import analysis_store

if TYPE_CHECKING:
    pass


CATEGORIES = [
    ("single", "独立分析", "sel_single"),
    ("compare", "对比分析", "sel_compare"),
    ("video", "视频生成", "sel_video"),
]

CAT_KEY_TO_MODULES = {
    "single": lambda: analysis_store.individual_modules,
    "compare": lambda: analysis_store.comparison_modules,
    "video": lambda: analysis_store.video_modules,
}


class ModulePicker(Vertical):
    """模块选择面板"""

    CSS = """
    ModulePicker {
        border: solid $border-primary;
        background: $bg-primary;
    }

    .section_label {
        color: $text-accent;
        text-style: bold;
        padding: 0 1;
        height: 1;
    }

    .section_list {
        height: auto;
        max-height: 12;
    }

    #mod_buttons {
        height: auto;
        padding: 0 1;
    }

    #mod_buttons Button {
        width: auto;
    }
    """

    def compose(self):
        yield Static("— 分析模块", classes="panel_title")
        with VerticalScroll():
            for cat_key, cat_label, sel_id in CATEGORIES:
                yield Static(f"— {cat_label}", classes="section_label")
                yield SelectionList(id=sel_id, classes="section_list")
        with Horizontal(id="mod_buttons"):
            yield Button("全选", id="btn_select_all", variant="default")
            yield Button("取消全选", id="btn_deselect_all", variant="default")

    def on_mount(self):
        analysis_store.discover_modules()
        self._refresh_all()

    # ── 事件 ──

    @on(SelectionList.SelectedChanged, "#sel_single")
    def _on_single_changed(self):
        self._sync_selection("sel_single", analysis_store.individual_modules)

    @on(SelectionList.SelectedChanged, "#sel_compare")
    def _on_compare_changed(self):
        self._sync_selection("sel_compare", analysis_store.comparison_modules)

    @on(SelectionList.SelectedChanged, "#sel_video")
    def _on_video_changed(self):
        self._sync_selection("sel_video", analysis_store.video_modules)

    @on(Button.Pressed, "#btn_select_all")
    def _on_select_all(self):
        for modules in [
            analysis_store.individual_modules,
            analysis_store.comparison_modules,
            analysis_store.video_modules,
        ]:
            analysis_store.select_all_in_category(modules)
        self._refresh_all()

    @on(Button.Pressed, "#btn_deselect_all")
    def _on_deselect_all(self):
        analysis_store.selected_module_names.clear()
        self._refresh_all()

    # ── 内部 ──

    def _sync_selection(self, sel_id: str, modules: dict):
        """将 SelectionList 的选中状态同步到 store"""
        sel = self.query_one(f"#{sel_id}", SelectionList)
        selected = set(sel.selected)
        for name in modules:
            if name in selected:
                analysis_store.selected_module_names.add(name)
            else:
                analysis_store.selected_module_names.discard(name)

    def _refresh_all(self):
        """刷新所有三组 SelectionList"""
        selected = analysis_store.selected_module_names
        for cat_key, cat_label, sel_id in CATEGORIES:
            modules = CAT_KEY_TO_MODULES[cat_key]()
            sel = self.query_one(f"#{sel_id}", SelectionList)
            sel.clear_options()
            if not modules:
                continue
            options = []
            for name, mod in modules.items():
                desc = getattr(mod, "description", "")
                if len(desc) > 50:
                    desc = desc[:47] + "..."
                label = f"[bold]{name}[/bold]\n  [dim]{desc}[/dim]"
                options.append((label, name, name in selected))
            sel.add_options(options)
