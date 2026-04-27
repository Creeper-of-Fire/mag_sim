"""
CSV 工具调用器 - 模板生成 + 运行前转换
"""
import json
import subprocess
import sys
from pathlib import Path

from tui.store.log_store import logger
from utils.project_config import PROJECT_ROOT, FILENAME_TASKS_CSV


class CsvToolRunner:
    """封装对 batch/csv_tool_*.py 的调用"""

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

        logger.info(f"[CSV工具] 生成模板: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace"
            )
            if result.stdout:
                logger.write(result.stdout.strip())
            if result.stderr:
                logger.error(result.stderr.strip())

            return result.returncode == 0
        except Exception as e:
            logger.error(f"[CSV工具] 模板生成异常: {e}")
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

        logger.info(f"[CSV工具] 转换CSV: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace"
            )
            if result.stdout:
                logger.write(result.stdout.strip())
            if result.stderr:
                logger.error(result.stderr.strip())

            return result.returncode == 0
        except Exception as e:
            logger.error(f"[CSV工具] CSV转换异常: {e}")
            return False

    def get_schema(self, script_name: str) -> dict | None:
        """
        调用 csv_tool dump-schema 获取参数元信息。

        Returns:
            schema dict，包含 column_task_name 和 params 列表。
            失败返回 None。
        """
        script_path = PROJECT_ROOT / "batch" / script_name

        cmd = [
            sys.executable, str(script_path),
            "dump-schema"
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=10
            )
            if result.returncode == 0 and result.stdout.strip():
                return json.loads(result.stdout.strip())
            else:
                logger.error(f"[CSV工具] dump-schema 失败: {result.stderr.strip() if result.stderr else '无输出'}")
                return None
        except subprocess.TimeoutExpired:
            logger.error("[CSV工具] dump-schema 超时")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"[CSV工具] dump-schema JSON 解析失败: {e}")
            return None
        except Exception as e:
            logger.error(f"[CSV工具] dump-schema 异常: {e}")
            return None