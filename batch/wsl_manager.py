# batch/wsl_manager.py
import queue
import subprocess
import os
import json
import threading

from .manager_api import BaseComputeManager, JobStatus
from utils.project_config import get_spack_activation_command, MAIN_SCRIPT_PATH, PROJECT_ROOT_WSL


class WSLComputeManager(BaseComputeManager):
    def __init__(self):
        self.process = None
        self._status = JobStatus.PENDING
        self.log_queue = queue.Queue() # 存放日志的队列
        # 远端代理脚本在 WSL 里的路径
        self.spack_cmd = get_spack_activation_command()
        self.agent_path = f"{PROJECT_ROOT_WSL}/agent/node_executor.py"

    def _read_stream(self):
        """后台线程：将标准输出实时灌入队列"""
        if self.process and self.process.stdout:
            # 迭代读取每一行，直到管道关闭
            for line in iter(self.process.stdout.readline, ''):
                if line:
                    self.log_queue.put(line)
            self.process.stdout.close()

    def submit(self, task_hash: str, params: dict, output_dir_name: str):
        config_json = json.dumps(params)

        # 构建 WSL 内部执行命令
        # 激活环境 -> 运行代理脚本
        # 显式使用 bash -l 确保环境加载，并强制 Python 以 UTF-8 输出
        cmd = (
            f"export PYTHONIOENCODING=utf-8 && "
            f"export LANG=C.UTF-8 && "  # 添加这行
            f"export LC_ALL=C.UTF-8 && "  # 添加这行
            f"{self.spack_cmd} && "
            f"python {self.agent_path} "
            f"--hash {task_hash} --out_name {output_dir_name} --config '{config_json}'"
        )

        self.process = subprocess.Popen(
            ["wsl", "bash", "-l", "-c", cmd],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',
            errors='replace',
            bufsize=1  # 行缓冲
        )

        # 启动后台读取线程
        self.reader_thread = threading.Thread(target=self._read_stream, daemon=True)
        self.reader_thread.start()

        self._status = JobStatus.RUNNING

    def get_status(self) -> JobStatus:
        if self._status not in [JobStatus.RUNNING, JobStatus.PENDING]:
            return self._status

        ret_code = self.process.poll()
        if ret_code is None:
            return JobStatus.RUNNING

        self._status = JobStatus.SUCCESS if ret_code == 0 else JobStatus.FAILED
        return self._status

    def get_logs(self) -> list[str]:
        """非阻塞地从队列中提取所有当前积压的日志"""
        lines = []
        try:
            while not self.log_queue.empty():
                # 使用 nowait 确保绝对不阻塞
                lines.append(self.log_queue.get_nowait())
        except queue.Empty:
            pass
        return lines

    def interrupt(self):
        if self.process:
            # 中断 WSL 内部的所有相关进程
            subprocess.run(["wsl", "killall", "python"], capture_output=True)
            self.process.terminate()
            self._status = JobStatus.CANCELLED