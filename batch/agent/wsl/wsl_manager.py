# batch/wsl_manager.py
import queue
import subprocess
import threading

import node_executor_wsl as wsl_agent
from batch.manager_api import BaseComputeManager, JobStatus
from project_config_wsl import PROJECT_ROOT_WSL, get_spack_activation_command


class WSLComputeManager(BaseComputeManager):
    def __init__(self):
        self.process = None
        self._status = JobStatus.PENDING
        self.log_queue = queue.Queue()  # 存放日志的队列

    def _read_stream(self):
        """后台线程：将标准输出实时灌入队列"""
        if self.process and self.process.stdout:
            # 迭代读取每一行，直到管道关闭
            for line in iter(self.process.stdout.readline, ''):
                if line:
                    self.log_queue.put(line)
            self.process.stdout.close()

    def submit(self, task_hash: str, params: dict, output_dir_name: str, rel_job_path: str):
        spack_cmd = get_spack_activation_command()

        agent_cmd = self.build_node_command(
            executor_module=wsl_agent,
            remote_root=PROJECT_ROOT_WSL,  # 直接使用 .env 配置的根目录
            task_hash=task_hash,
            output_dir_name=output_dir_name,
            rel_job_path=rel_job_path,
            params=params,
            python_exe="python"
        )

        # 构建 WSL 内部执行命令
        # 激活环境 -> 运行代理脚本
        # 显式使用 bash -l 确保环境加载，并强制 Python 以 UTF-8 输出
        cmd = (
            f"export PYTHONIOENCODING=utf-8 && "
            f"export LANG=C.UTF-8 && "  # 添加这行
            f"export LC_ALL=C.UTF-8 && "  # 添加这行
            f"{spack_cmd} && "
            f"{agent_cmd}"
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
