#!/usr/bin/env python3
"""Plasma Analysis Console — Rich 交互式总控页面

轻量级分析管理器。不依赖 Textual，只用 Rich + questionary 提供:
- 交互式模块选择 (questionary.checkbox: 空格切换, 方向键导航)
- 目录选择 (复用现有两级选择器)
- 分析运行 (复用现有执行器)
- 模块选择持久化缓存
- 增量刷新 (按文件 mtime 判断是否需要重导)
- Ctrl+C 任意位置回到主页面
"""

import importlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich.text import Text

from analysis.core.config import config as analysis_config
from analysis.core.executor import execute_analysis
from analysis.core.utils import console, select_directories
from analysis.plotting.styles import StyleTheme, set_style
from analysis.utils import setup_chinese_font
from analysis_tui.stores.analysis_store import analysis_store
from utils.project_config import PROJECT_ROOT

# ── 缓存路径 ──
CACHE_DIR = PROJECT_ROOT / "data"
CACHE_FILE = CACHE_DIR / "module_selections.json"


# ═══════════════════════════════════════════════════════════════════════════════
# 总控台
# ═══════════════════════════════════════════════════════════════════════════════

class AnalysisConsole:
    """Rich 总控页面 — 循环交互式分析管理器。"""

    def __init__(self):
        self._console = console
        self._store = analysis_store
        self._output_dir = analysis_config.global_output_dir
        self._style = StyleTheme.PRESENTATION
        self._last_run_info: Optional[dict] = None

        # 模块文件 mtime 缓存: {file_path: mtime}
        self._file_mtimes: Dict[str, float] = {}
        # 模块文件 → 模块名称 映射: {file_path: [name, ...]}
        self._file_module_map: Dict[str, List[str]] = {}

    # ── 主循环 ──

    def run(self):
        """启动总控台主循环。"""
        set_style(StyleTheme.PRESENTATION)
        analysis_config.global_output_dir = self._output_dir
        os.makedirs(self._output_dir, exist_ok=True)
        setup_chinese_font()

        with self._console.status("[bold green]正在发现分析模块...[/bold green]"):
            self._discover_all()

        self._load_module_cache()

        self._console.print()

        while True:
            try:
                self._render_full_dashboard()
                cmd = Prompt.ask("  [bold]>[/bold] ", console=self._console).strip().lower()
                if not cmd:
                    continue
                self._dispatch(cmd)
            except KeyboardInterrupt:
                self._console.print("\n[yellow]按 [bold]q[/bold] 退出程序[/yellow]")
            except Exception:
                self._console.print_exception()

    # ── 仪表盘 ──

    def _render_full_dashboard(self):
        dirs = self._store.selected_dir_paths
        mods = self._store.selected_module_names
        all_mods = self._store.all_modules

        info = Table(box=None, padding=(0, 1), show_header=False)
        info.add_column(style="bold cyan")
        info.add_column()
        info.add_row("输出目录:", f"[green]{self._output_dir}[/green]")
        info.add_row("绘图样式:", f"[green]{self._style.name}[/green]")
        info.add_row("数据目录:", f"[green]{len(dirs)} 个已选择[/green]")
        info.add_row(
            "分析模块:",
            f"[green]{len(mods)} 个已选择[/green]  [dim](共 {len(all_mods)} 个可用)[/dim]",
        )
        if self._last_run_info:
            when = self._last_run_info.get("time", "")
            ok = self._last_run_info.get("ok", 0)
            fail = self._last_run_info.get("fail", 0)
            info.add_row("上次运行:", f"[dim]{when}  |  成功 {ok} 个, 失败 {fail} 个[/dim]")
        else:
            info.add_row("上次运行:", "[dim]尚无[/dim]")

        keys_text = Text.from_markup(
            "  [cyan]d[/cyan] 选择目录  [cyan]s[/cyan] 选择模块  [cyan]f[/cyan] 刷新  "
            "[cyan]r[/cyan] 运行  [cyan]t[/cyan] 工具  [cyan]o[/cyan] 输出  [cyan]q[/cyan] 退出")

        panel = Panel(
            info,
            title="[bold inverse] Plasma Analysis Console [/bold inverse]",
            subtitle=keys_text,
            padding=(1, 2),
        )
        self._console.print(panel)

    # ── 命令分发 ──

    def _dispatch(self, cmd: str):
        handlers = {
            'd': self._cmd_select_dirs,
            's': self._cmd_select_modules,
            'f': self._cmd_refresh,
            'r': self._cmd_run,
            't': self._cmd_tools,
            'o': self._cmd_output_dir,
            'q': self._cmd_quit,
        }
        handler = handlers.get(cmd)
        if handler:
            handler()
        else:
            self._console.print(f"[red]未知命令: {cmd}[/red]")

    # ── d: 选择目录 ──

    def _cmd_select_dirs(self):
        try:
            dirs = select_directories()
        except KeyboardInterrupt:
            self._console.print("[yellow]已取消目录选择[/yellow]")
            return

        if not dirs:
            self._console.print("[yellow]未选择任何目录[/yellow]")
            return

        self._store.clear_dirs()
        for d in dirs:
            self._store.add_dir(d)
            self._store.load_run(d)
        self._console.print(f"[green]已选择 {len(dirs)} 个目录[/green]\n")

    # ── s: 选择模块 ──

    def _cmd_select_modules(self):
        from analysis.core.selector import SimpleTableSelector

        all_mods = self._store.all_modules
        if not all_mods:
            self._console.print("[yellow]没有可用的模块[/yellow]")
            return

        # 构建 item 列表: (name, instance, tag)
        items = []
        for name, inst in all_mods.items():
            if name in self._store.individual_modules:
                tag = "[cyan]独立[/cyan]"
            elif name in self._store.comparison_modules:
                tag = "[yellow]对比[/yellow]"
            else:
                tag = "[magenta]视频[/magenta]"
            items.append((name, inst, tag))

        preselected = self._store.selected_module_names

        def fmt(item):
            name, inst, tag = item
            desc = inst.description[:70] if inst.description else ""
            mark = "[green]✓[/green]" if name in preselected else " "
            return [mark, f"[bold]{name}[/bold]", f"{tag}", f"[dim]{desc}[/dim]"]

        selector = SimpleTableSelector(
            items=items,
            columns=["", "模块", "类型", "描述"],
            row_converter=fmt,
            title="请选择分析模块 (all=全选, 0,2,5-8=多选, Ctrl+C 保留旧选择)",
        )
        try:
            selected_items = selector.select(
                default="all" if not preselected else None,
            )
        except KeyboardInterrupt:
            self._console.print("[yellow]已取消模块选择[/yellow]")
            return

        if selected_items:
            self._store.selected_module_names = {item[0] for item in selected_items}
            self._save_module_cache()
        self._console.print(f"[green]已选择 {len(self._store.selected_module_names)} 个模块[/green]\n")

    # ── f: 刷新 ──

    def _cmd_refresh(self):
        with self._console.status("[bold green]正在刷新模块...[/bold green]"):
            self._refresh_modules()

        with self._console.status("[bold green]正在刷新目录...[/bold green]"):
            self._refresh_dirs()

        self._console.print("[green]✓ 刷新完成[/green]\n")

    # ── r: 运行分析 ──

    def _cmd_run(self):
        modules = self._store.get_selected_modules()
        runs = self._store.get_loaded_runs()

        if not runs:
            self._console.print("[yellow]请先选择数据目录 [cyan]d[/cyan][/yellow]")
            return
        if not modules:
            self._console.print("[yellow]请先选择分析模块 [cyan]s[/cyan][/yellow]")
            return

        analysis_config.global_output_dir = self._output_dir

        self._console.print(f"[bold]目录: {len(runs)} 个 | 模块: {len(modules)} 个[/bold]")
        self._console.print("─" * 50)

        ok = 0
        fail = 0
        try:
            errors = execute_analysis(modules, runs, output=self._console.print)
            fail = len(errors) if errors else 0
            ok = len(modules) - fail
            if fail:
                self._console.print(f"\n[red]{fail} 个模块出错[/red]")
            else:
                self._console.print("\n[bold green]✓ 全部完成[/bold green]")
        except KeyboardInterrupt:
            self._console.print("\n[yellow]已中断[/yellow]")
            return
        finally:
            self._last_run_info = {
                "time": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "ok": ok,
                "fail": fail,
            }

        self._console.print("─" * 50)

    # ── t: 工具 ──

    def _cmd_tools(self):
        from analyze import _run_tool_workflow

        dirs = self._store.selected_dir_paths
        if not dirs:
            self._console.print("[yellow]请先选择数据目录 [cyan]d[/cyan][/yellow]")
            return

        try:
            _run_tool_workflow("interactive", dirs)
        except KeyboardInterrupt:
            self._console.print("\n[yellow]已中断工具[/yellow]")

        self._console.print("─" * 50)

    # ── o: 输出目录 ──

    def _cmd_output_dir(self):
        try:
            new_dir = Prompt.ask(
                "  [bold]输出目录[/bold]", console=self._console, default=self._output_dir
            )
        except KeyboardInterrupt:
            return
        if new_dir:
            self._output_dir = new_dir
            analysis_config.global_output_dir = new_dir
            os.makedirs(new_dir, exist_ok=True)
            self._console.print(f"[green]输出目录已设为: {new_dir}[/green]")

    # ── q: 退出 ──

    def _cmd_quit(self):
        try:
            ans = Prompt.ask(
                "  [bold]确认退出?[/bold] (y/n)", console=self._console,
                choices=["y", "n"], default="y",
            )
        except KeyboardInterrupt:
            return
        if ans == 'y':
            self._console.print("[dim]再见。[/dim]")
            sys.exit(0)

    # ═══════════════════════════════════════════════════════════════════════════
    # 模块发现与刷新
    # ═══════════════════════════════════════════════════════════════════════════

    def _discover_all(self):
        """首次全量发现模块，记录 mtime 和文件→模块映射。"""
        from analyze import discover_modules as _discover

        individual, comparison, video = _discover()
        self._store.individual_modules = individual
        self._store.comparison_modules = comparison
        self._store.video_modules = video

        self._rebuild_mtime_cache()

    def _rebuild_mtime_cache(self):
        """重建 mtime 缓存和文件→模块映射。"""
        self._file_mtimes.clear()
        self._file_module_map.clear()

        base_dir = Path(__file__).resolve().parent
        modules_dir = base_dir / "analysis" / "modules"

        for f in modules_dir.rglob("*.py"):
            if f.name.startswith(('_', 'base_')):
                continue
            fpath = str(f)
            self._file_mtimes[fpath] = f.stat().st_mtime
            self._file_module_map[fpath] = []

        all_mods = self._store.all_modules
        for name, inst in all_mods.items():
            cls = type(inst)
            mod = cls.__module__
            parts = mod.split('.')
            rel = Path(*parts).with_suffix('.py')
            file_path = str(base_dir / rel)
            if file_path in self._file_module_map:
                self._file_module_map[file_path].append(name)
            else:
                found = False
                for fpath in self._file_mtimes:
                    if Path(fpath).stem == parts[-1]:
                        self._file_module_map[fpath].append(name)
                        found = True
                        break
                if not found:
                    self._file_module_map[file_path] = [name]
                    self._file_mtimes[file_path] = Path(file_path).stat().st_mtime if Path(file_path).exists() else 0

    def _refresh_modules(self):
        """增量刷新: 按 mtime 判断是否需要重导模块。"""
        base_dir = Path(__file__).resolve().parent
        modules_dir = base_dir / "analysis" / "modules"

        old_selected = set(self._store.selected_module_names)

        current_files = set()
        for f in modules_dir.rglob("*.py"):
            if f.name.startswith(('_', 'base_')):
                continue
            fpath = str(f)
            current_files.add(fpath)
            new_mtime = f.stat().st_mtime

            relative_path = f.relative_to(base_dir)
            module_name = ".".join(relative_path.with_suffix("").parts)

            if fpath in self._file_mtimes and self._file_mtimes[fpath] == new_mtime:
                continue

            try:
                if module_name in sys.modules:
                    importlib.reload(sys.modules[module_name])
                else:
                    importlib.import_module(module_name)
            except Exception:
                continue

        deleted_files = set(self._file_mtimes.keys()) - current_files
        for fpath in deleted_files:
            for name in self._file_module_map.get(fpath, []):
                self._store.individual_modules.pop(name, None)
                self._store.comparison_modules.pop(name, None)
                self._store.video_modules.pop(name, None)

        from analyze import discover_modules as _discover
        individual, comparison, video = _discover()
        self._store.individual_modules = individual
        self._store.comparison_modules = comparison
        self._store.video_modules = video

        self._rebuild_mtime_cache()

        all_names = set(self._store.all_modules.keys())
        self._store.selected_module_names = old_selected & all_names

    def _refresh_dirs(self):
        """刷新目录: 验证已选目录仍然存在，清理无效的。"""
        valid = []
        for p in self._store.selected_dir_paths:
            path = Path(p)
            if path.exists():
                valid.append(p)
            else:
                self._store.evict_run(p)
        self._store.selected_dir_paths = valid

    # ═══════════════════════════════════════════════════════════════════════════
    # 模块选择缓存
    # ═══════════════════════════════════════════════════════════════════════════

    def _load_module_cache(self):
        """从磁盘加载上次的模块选择。"""
        if not CACHE_FILE.exists():
            return

        try:
            data = json.loads(CACHE_FILE.read_text(encoding='utf-8'))
            cached = set(data.get("selected", []))
        except Exception:
            return

        if not cached:
            return

        available = set(self._store.all_modules.keys())
        valid = cached & available
        if valid:
            self._store.selected_module_names = valid
            self._console.print(f"[dim]已恢复 {len(valid)} 个模块选择[/dim]")

    def _save_module_cache(self):
        """将当前模块选择持久化到磁盘。"""
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        data = {"selected": sorted(self._store.selected_module_names)}
        CACHE_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')


# ═══════════════════════════════════════════════════════════════════════════════
# 入口
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    """启动分析总控台。"""
    app = AnalysisConsole()
    app.run()


if __name__ == "__main__":
    main()
