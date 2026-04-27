"""
项目级配置管理 - 读写 {job_dir}/job_config.json
"""
import json
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class JobConfig:
    """单个项目目录下的运行配置"""
    script_name: str = "csv_tool_constant_energy.py"
    extra_args: str = ""


class JobConfigManager:
    """管理项目目录下的 job_config.json"""

    FILENAME = "job_config.json"

    def __init__(self, job_dir: Path):
        self._job_dir = job_dir
        self._config_path = job_dir / self.FILENAME

    def load(self) -> JobConfig:
        """加载配置，文件不存在时返回默认值"""
        if not self._config_path.exists():
            return JobConfig()

        try:
            with open(self._config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return JobConfig(
                script_name=data.get("script_name", "csv_tool_constant_energy.py"),
                extra_args=data.get("extra_args", "")
            )
        except (json.JSONDecodeError, FileNotFoundError):
            return JobConfig()

    def save(self, config: JobConfig) -> None:
        """保存配置到文件"""
        self._config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._config_path, "w", encoding="utf-8") as f:
            json.dump(asdict(config), f, indent=4, ensure_ascii=False)

    @property
    def config_path(self) -> Path:
        return self._config_path