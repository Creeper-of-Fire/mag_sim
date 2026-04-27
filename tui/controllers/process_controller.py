# process_controller.py
"""
进程遥控器 - 管理子进程的生命周期
"""
import asyncio
import os
import re
import sys
from pathlib import Path

from tui.store.log_store import logger
from tui.store.runtime_store import runtime_store


class BatchProcessController:
    """管理批处理子进程"""

    def __init__(self):
        self._process = None

    async def start(self, job_dir: Path, runner_path: Path):
        """启动批处理进程"""
        if runtime_store.is_running:
            logger.warn("[Controller] 已有进程在运行中")
            return

        # 构建命令
        cmd = [sys.executable, str(runner_path), str(job_dir)]
        logger.info(f"[Controller] 启动进程: {' '.join(cmd)}")

        try:
            # 创建子进程
            self._process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                env={**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
            )

            # 读取输出
            await self._read_output()

            # 等待进程结束
            returncode = await self._process.wait()

            logger.info(f"[Controller] 进程结束，返回码: {returncode}")

        except Exception as e:
            logger.error(f"[Controller] 进程异常: {e}")
        finally:
            self._process = None
            runtime_store.set_running(False)

    async def _read_output(self):
        """持续读取子进程输出"""
        if not self._process or not self._process.stdout:
            return

        buffer = ""
        while True:
            try:
                line = await self._process.stdout.readline()
                if not line:
                    break

                text = line.decode('utf-8', errors='replace').rstrip()

                # 捕获 PID
                if "PID:" in text:
                    match = re.search(r'PID:(\d+)', text)
                    if match:
                        logger.info(f"[Controller] 捕获进程 PID: {match.group(1)}")

                if text:
                    logger.write(text)

            except Exception as e:
                logger.error(f"[Controller] 读取输出异常: {e}")
                break

    async def stop(self):
        """停止进程"""
        if not self._process:
            logger.info("[Controller] 没有正在运行的进程")
            return

        logger.info("[Controller] 正在终止进程...")

        try:
            self._process.terminate()
            try:
                await asyncio.wait_for(self._process.wait(), timeout=3.0)
            except asyncio.TimeoutError:
                logger.info("[Controller] 进程未响应，强制结束")
                self._process.kill()
                await self._process.wait()
        except Exception as e:
            logger.error(f"[Controller] 停止进程异常: {e}")
