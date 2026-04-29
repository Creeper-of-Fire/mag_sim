"""模块多选器 — 按分类展示，多选勾选"""
from __future__ import annotations

from typing import TYPE_CHECKING

from textual import on
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, ListView, ListItem, Label, Static

from analysis_tui.stores.analysis_store import analysis_store

if TYPE_CHECKING:
    from analysis.modules.abstract.base_module import BaseAnalysisModule


class ModulePicker(Vertical):
    """模块选择面板"""

    CSS = """
    ModulePicker {
        border: solid $border-primary;
        background: $bg-primary;
    }

    #mod_header {
        height: auto;
        padding: 0 1;
    }

    #mod_list {
        height: 1fr;
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
        yield ListView(id="mod_list")
        with Horizontal(id="mod_buttons"):
            yield Button("全选", id="btn_select_all", variant="default")
            yield Button("取消全选", id="btn_deselect_all", variant="default")

    def on_mount(self):
        analysis_store.discover_modules()
        self._refresh_list()
        analysis_store.subscribe(self._on_store_changed)

    def on_unmount(self):
        analysis_store.unsubscribe(self._on_store_changed)

    # ── 事件 ──

    @on(ListView.Selected, "#mod_list")
    def _on_item_selected(self, event: ListView.Selected):
        if event.item is None:
            return
        name = event.item.id
        if name is None:
            return
        # 跳过类别标题（以 __category__ 为前缀）
        if name.startswith("__"):
            return

        analysis_store.toggle_module(name)
        self._refresh_list()

    @on(Button.Pressed, "#btn_select_all")
    def _on_select_all(self):
        for modules in [
            analysis_store.individual_modules,
            analysis_store.comparison_modules,
            analysis_store.video_modules,
        ]:
            analysis_store.select_all_in_category(modules)
        self._refresh_list()

    @on(Button.Pressed, "#btn_deselect_all")
    def _on_deselect_all(self):
        analysis_store.selected_module_names.clear()
        analysis_store._notify()
        self._refresh_list()

    # ── 内部 ──

    def _refresh_list(self):
        list_view = self.query_one("#mod_list", ListView)
        list_view.clear()
        selected = analysis_store.selected_module_names

        categories = [
            ("独立分析", analysis_store.individual_modules),
            ("对比分析", analysis_store.comparison_modules),
            ("视频生成", analysis_store.video_modules),
        ]

        for cat_name, modules in categories:
            if not modules:
                continue

            # 类别标题（不可选）
            list_view.append(
                ListItem(
                    Label(f"[bold underline]{cat_name} ({len(modules)})[/bold underline]"),
                    id=f"__category__{cat_name}",
                    disabled=True,
                )
            )

            for name, mod in modules.items():
                prefix = "☑" if name in selected else "☐"
                desc = getattr(mod, "description", "")
                # 截断过长的描述
                if len(desc) > 60:
                    desc = desc[:57] + "..."
                label_text = f"  {prefix} [bold]{name}[/bold]\n     [dim]{desc}[/dim]"
                list_view.append(
                    ListItem(Label(label_text), id=name)
                )

    def _on_store_changed(self):
        self.call_after_refresh(self._refresh_list)
