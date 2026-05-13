"""
模拟进度条 — 订阅 ProgressStore，实时渲染进度信息
"""
from textual.widgets import Static

from tui.store.progress_store import progress_store, ProgressState


def _format_duration(seconds: float) -> str:
    if seconds <= 0:
        return "--"
    if seconds >= 3600:
        return f"{seconds / 3600:.1f} 小时"
    if seconds >= 60:
        return f"{seconds / 60:.1f} 分钟"
    return f"{seconds:.0f} 秒"


class SimulationProgressBar(Static):
    """模拟进度条，dock 在 LogPanel 顶部"""

    DEFAULT_CSS = """
    SimulationProgressBar {
        height: 1;
        background: $bg-tertiary;
        color: $text-accent;
        padding: 0 1;
        text-style: bold;
    }
    """

    def on_mount(self):
        progress_store.subscribe(self._on_progress)
        self._update_display(progress_store.state)

    def on_unmount(self):
        progress_store.unsubscribe(self._on_progress)

    def _on_progress(self, state: ProgressState):
        self._update_display(state)

    def _update_display(self, state: ProgressState):
        if state.total_steps <= 0:
            self.update("")
            return

        step = state.current_step
        total = state.total_steps
        pct = state.percentage

        bar_width = 20
        filled = int(bar_width * pct / 100)
        bar = "|" + "=" * filled + "-" * (bar_width - filled) + "|"

        avg = f"{state.avg_per_step:.3f}s/步" if state.avg_per_step > 0 else "--"
        elapsed = _format_duration(state.elapsed_seconds)
        eta = _format_duration(state.eta_seconds)

        self.update(
            f" 模拟进度 {bar} {step}/{total} ({pct:.1f}%) "
            f"| 均速: {avg} "
            f"| 已用: {elapsed} "
            f"| 剩余: {eta}"
        )
