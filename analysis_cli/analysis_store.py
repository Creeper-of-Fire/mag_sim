"""分析状态管理 — 单例 + pub-sub

负责模块发现缓存、数据加载缓存、选择状态管理。
不依赖任何 UI 框架。
"""
from __future__ import annotations

from typing import Callable, List, TYPE_CHECKING

from analysis.core.load_run_data_loader import load_run_data

if TYPE_CHECKING:
    from analysis.core.simulation import SimulationRun
    from analysis.modules.abstract.base_module import (
        BaseAnalysisModule,
        BaseComparisonModule,
        BaseVideoModule,
    )


class AnalysisStore:
    """分析全局状态（单例）"""

    _instance: AnalysisStore | None = None

    def __new__(cls) -> AnalysisStore:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init()
        return cls._instance

    def _init(self):
        self._listeners: list[Callable[[], None]] = []

        # 模块发现缓存
        self.individual_modules: dict[str, "BaseAnalysisModule"] = {}
        self.comparison_modules: dict[str, "BaseComparisonModule"] = {}
        self.video_modules: dict[str, "BaseVideoModule"] = {}

        # 选择状态
        self.selected_module_names: set[str] = set()
        self.selected_dir_paths: list[str] = []

        # 数据缓存 — {dir_path: SimulationRun}
        self._loaded_runs: dict[str, "SimulationRun"] = {}

        # 运行时状态
        self.is_running: bool = False

    # ── 模块发现 ──

    def discover_modules(self) -> bool:
        """扫描并缓存所有分析模块。返回 True 表示有可用模块。"""
        if self.individual_modules or self.comparison_modules:
            return True  # 已缓存

        from analyze import discover_modules as _discover

        try:
            individual, comparison, video = _discover()
            self.individual_modules = individual
            self.comparison_modules = comparison
            self.video_modules = video
            self._notify()
            return bool(individual or comparison or video)
        except Exception:
            return False

    @property
    def all_modules(self) -> dict[str, "BaseAnalysisModule"]:
        """合并所有类型的模块为一个 dict"""
        result = {}
        result.update(self.individual_modules)
        result.update(self.comparison_modules)
        result.update(self.video_modules)
        return result

    # ── 目录 & 数据 ──

    def load_run(self, dir_path: str) -> "SimulationRun | None":
        """懒加载一个 SimulationRun 并缓存。已缓存的直接返回。"""
        if dir_path in self._loaded_runs:
            return self._loaded_runs[dir_path]
        run = load_run_data(dir_path)
        if run is not None:
            self._loaded_runs[dir_path] = run
        return run

    def get_loaded_runs(self) -> list["SimulationRun"]:
        """返回当前选中目录对应的已加载 SimulationRun 引用列表（按选中顺序）。"""
        runs = []
        for p in self.selected_dir_paths:
            run = self._loaded_runs.get(p)
            if run is not None:
                runs.append(run)
        return runs

    def evict_run(self, dir_path: str):
        """从缓存中移除一个 run"""
        self._loaded_runs.pop(dir_path, None)

    # ── 选择操作 ──

    def add_dir(self, dir_path: str):
        if dir_path not in self.selected_dir_paths:
            self.selected_dir_paths.append(dir_path)
            self._notify()

    def remove_dir(self, dir_path: str):
        if dir_path in self.selected_dir_paths:
            self.selected_dir_paths.remove(dir_path)
            self._notify()

    def clear_dirs(self):
        self.selected_dir_paths.clear()
        self._loaded_runs.clear()
        self._notify()

    def toggle_module(self, name: str):
        if name in self.selected_module_names:
            self.selected_module_names.discard(name)
        else:
            self.selected_module_names.add(name)
        self._notify()

    def select_all_in_category(self, modules: dict[str, "BaseAnalysisModule"]):
        for name in modules:
            self.selected_module_names.add(name)
        self._notify()

    def deselect_all_in_category(self, modules: dict[str, "BaseAnalysisModule"]):
        for name in modules:
            self.selected_module_names.discard(name)
        self._notify()

    def get_selected_modules(self) -> list["BaseAnalysisModule"]:
        """返回选中的模块实例引用列表（保持 all_modules 的遍历顺序）。"""
        all_mods = self.all_modules
        return [m for name, m in all_mods.items() if name in self.selected_module_names]

    # ── 订阅 ──

    def subscribe(self, callback: Callable[[], None]):
        self._listeners.append(callback)

    def unsubscribe(self, callback: Callable[[], None]):
        if callback in self._listeners:
            self._listeners.remove(callback)

    def _notify(self):
        for listener in self._listeners:
            try:
                listener()
            except Exception:
                pass


analysis_store = AnalysisStore()
