"""
CSV 工具调用器 - 模板生成 + 运行前转换
"""
import subprocess
import sys
from pathlib import Path

from utils.project_config import PROJECT_ROOT, FILENAME_TASKS_CSV


class CsvToolRunner:
    """封装对 batch/csv_tool_*.py 的调用"""

    def __init__(self, on_log=None):
        self.on_log = on_log or (lambda msg: None)

    def generate_template(self, job_dir: Path, script_name: str) -> bool:
        """
        调用 csv_tool 生成 tasks.csv 模板

        Returns:
            是否成功
        """
        script_path = PROJECT_ROOT / "batch" / script_name
        target_csv = job_dir / FILENAME_TASKS_CSV

        cmd = [
            sys.executable, str(script_path),
            "generate-template",
            "-o", str(target_csv)
        ]

        self.on_log(f"[CSV工具] 生成模板: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace"
            )
            if result.stdout:
                self.on_log(result.stdout.strip())
            if result.stderr:
                self.on_log(result.stderr.strip())

            return result.returncode == 0
        except Exception as e:
            self.on_log(f"[CSV工具] 模板生成异常: {e}")
            return False

    def convert_csv(self, job_dir: Path, script_name: str, extra_args: str = "") -> bool:
        """
        在运行前将 tasks.csv 转换为 runner 可用的格式

        Returns:
            是否成功
        """
        script_path = PROJECT_ROOT / "batch" / script_name
        csv_path = job_dir / FILENAME_TASKS_CSV

        cmd = [sys.executable, str(script_path), "convert", str(csv_path)]

        # 追加额外参数
        if extra_args.strip():
            cmd.extend(extra_args.strip().split())

        self.on_log(f"[CSV工具] 转换CSV: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace"
            )
            if result.stdout:
                self.on_log(result.stdout.strip())
            if result.stderr:
                self.on_log(result.stderr.strip())

            return result.returncode == 0
        except Exception as e:
            self.on_log(f"[CSV工具] CSV转换异常: {e}")
            return False