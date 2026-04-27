"""
Runner 配置组件 - 后端选择 + 额外参数
"""
from textual.containers import Horizontal, Vertical
from textual.widgets import Static, Select, Input


class RunnerConfig(Vertical):
    """Runner 后端与参数配置"""

    DEFAULT_CSS = """
    RunnerConfig {
        height: auto;
    }
    .runner_row {
        height: 3;
        align: left middle;
    }
    .runner_row Static {
        width: 14;
        color: $text-muted;
    }
    .runner_row Select {
        width: 40;
    }
    .runner_row Input {
        width: 1fr;
        background: $bg-input;
        color: $text-accent;
        border: solid $border-primary;
    }
    """

    BACKENDS = [
        ("yingbo (英博云)", "yingbo"),
        ("gongji (共绩云)", "gongji"),
        ("wsl (本地)", "wsl"),
    ]

    def compose(self):
        with Horizontal(classes="runner_row"):
            yield Static("后端:")
            yield Select(
                self.BACKENDS,
                id="select_backend",
                value="yingbo",
            )
        with Horizontal(classes="runner_row"):
            yield Static("额外参数:")
            yield Input(
                id="input_runner_extra",
                placeholder="如 --gpu A800",
            )

    def set_from_args(self, runner_args: str):
        """从 runner_args 字符串恢复 UI 状态"""
        manager = "yingbo"
        extra_parts = []
        tokens = runner_args.strip().split()
        i = 0
        while i < len(tokens):
            if tokens[i] == "--manager" and i + 1 < len(tokens):
                manager = tokens[i + 1]
                i += 2
            else:
                extra_parts.append(tokens[i])
                i += 1

        self.query_one("#select_backend", Select).value = manager
        self.query_one("#input_runner_extra", Input).value = " ".join(extra_parts)

    def get_runner_args(self) -> str:
        """组装完整的 runner_args 字符串"""
        manager = self.query_one("#select_backend", Select).value
        extra = self.query_one("#input_runner_extra", Input).value.strip()
        parts = [f"--manager {manager}"]
        if extra:
            parts.append(extra)
        return " ".join(parts)
