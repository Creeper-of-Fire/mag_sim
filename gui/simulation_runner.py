# gui/simulation_runner.py
import re
import subprocess
# !/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from PySide6.QtCore import QObject, Signal, QProcess

from gui.utils.notifications import run_notification_script

# --- 任务状态常量 ---
STATUS_PENDING = "待运行"
STATUS_RUNNING = "正在运行"
STATUS_COMPLETED = "已完成"
STATUS_FAILED = "失败"


# --- 用于跨线程通信的信号类 ---
class WorkerSignals(QObject):
    """定义了从工作线程发出的可用信号。"""
    log_message = Signal(str)      # 发送日志消息
    task_update = Signal()         # 请求更新任务列表UI
    queue_finished = Signal()      # 通知所有任务已完成


class SimulationRunner(QObject):
    """
    这个类将在一个单独的线程中运行，负责执行所有模拟任务。
    它通过信号与主GUI线程通信，避免冻结界面。
    """

    def __init__(self, queue):
        super().__init__()
        self.simulation_queue = queue
        self.signals = WorkerSignals()
        self.is_running = True
        self.current_process = None
        self.linux_pid = None         # 用于存储从WSL获取的Linux进程ID
        self.pid_capture_buffer = ""  # 用于处理PID数据流的缓冲区

    def handle_process_output(self):
        """处理来自子进程的输出，并从中捕获Linux PID。"""
        if not self.current_process:
            return

        data = self.current_process.readAllStandardOutput().data().decode('utf-8', errors='ignore')

        # --- 捕获 PID 的逻辑 ---
        if self.linux_pid is None:
            self.pid_capture_buffer += data
            # PID 应该是输出的第一行，以换行符结尾
            if '\n' in self.pid_capture_buffer:
                first_line, rest_of_data = self.pid_capture_buffer.split('\n', 1)
                match = re.match(r'^PID:(\d+)', first_line.strip())
                if match:
                    self.linux_pid = int(match.group(1))
                    self.signals.log_message.emit(f"日志: 已捕获到 WSL 中的主进程 PID: {self.linux_pid}\n")
                    # 将剩余的数据作为正常日志发出
                    if rest_of_data:
                        self.signals.log_message.emit(rest_of_data)
                else:
                    # 如果第一行不是预期的PID格式，则全部作为日志输出
                    self.signals.log_message.emit(self.pid_capture_buffer)
                self.pid_capture_buffer = ""  # 清空缓冲区
            # 如果还没收到换行符，则继续等待数据
        else:
            # PID 已经捕获，所有数据都是正常日志
            self.signals.log_message.emit(data)

    def run(self):
        """
        依次执行队列中的所有任务。
        """
        # 项目在 WSL 中的绝对路径
        # sys.argv[0] 是 main_gui.py 在 WSL 中的路径 (\\wsl.localhost\Ubuntu\...)
        # 我们需要把它转换成 Linux 风格的路径
        wsl_project_root = '/home/cof/Plasma_Simulation' # 根据您的路径硬编码，或者更智能地转换
        main_script_path_wsl = f"{wsl_project_root}/main.py"

        # Conda 环境名称
        conda_env_name = "warpx_env"

        for task in self.simulation_queue:
            if not self.is_running:
                self.signals.log_message.emit("日志: 任务队列被用户中止。\n")
                break

            # 更新任务状态为“正在运行”并通知UI
            task['status'] = STATUS_RUNNING
            self.signals.task_update.emit()
            self.signals.log_message.emit(f"\n{'=' * 20} 开始执行: {task['name']} {'=' * 20}\n")

            try:
                # 将参数字典序列化为 JSON 字符串
                config_json = json.dumps(task['params'])

                # --- 模拟完整的 Conda Shell 初始化 ---
                command_in_wsl = (
                    # 1. 强制设置哑终端，抑制所有ANSI颜色代码，解决乱码问题。
                    #    用 && 连接，确保后续命令在它成功后执行。
                    "export TERM=dumb && "

                    # 2. 手动加载 Conda 的核心初始化脚本。
                    #    这会定义 'conda' shell 函数并设置必要的路径，解决 'command not found' 问题。
                    #    这是比使用绝对路径更推荐的官方方式。
                    f"source /home/cof/miniconda3/etc/profile.d/conda.sh && "

                    # 3. 激活目标环境。
                    f"conda activate {conda_env_name} && "
                    "echo \"PID:$$\"; " # 打印自己的PID

                    # 4. 最后，在激活的环境中执行 Python 脚本。
                    # 在最后一个命令前加上 exec
                    # 这会用 python (mpirun) 进程替换掉 bash shell 进程
                    f"exec python {main_script_path_wsl} -c '{config_json}'"
                )

                # 使用 wsl.exe -e bash -c "..." 的方式来执行
                # -e: 执行指定的命令而不使用默认的交互式shell
                # -c: 后面跟着要执行的命令字符串
                command = ["wsl.exe", "-e", "bash", "-c", command_in_wsl]

                # QProcess 是 PySide/PyQt 的 subprocess 替代品，能更好地集成事件循环
                process = QProcess()
                # 合并标准输出和标准错误通道，方便一同读取
                process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
                self.current_process = process  # <--- 跟踪当前进程

                # 连接信号，实时获取子进程的输出
                self.current_process.readyReadStandardOutput.connect(self.handle_process_output)

                # 启动子进程
                process.start(command[0], command[1:])
                process.waitForFinished(-1)  # -1 表示无限期阻塞等待，直到进程结束

                self.current_process = None

                # 检查退出码，如果is_running是False，说明是用户主动停止的
                if not self.is_running:
                    task['status'] = STATUS_FAILED # 标记为失败，因为未完成
                    self.signals.log_message.emit(f"\n--- 任务 '{task['name']}' 被用户中止。 ---\n")
                elif process.exitCode() == 0:
                    task['status'] = STATUS_COMPLETED
                    self.signals.log_message.emit(f"\n--- 任务 '{task['name']}' 成功完成。 ---\n")
                else:
                    task['status'] = STATUS_FAILED
                    self.signals.log_message.emit(f"\n--- 错误: 任务 '{task['name']}' 失败，返回代码 {process.exitCode()}。 ---\n")
            except Exception as e:
                task['status'] = STATUS_FAILED
                self.signals.log_message.emit(f"\n--- 严重错误: 启动任务 '{task['name']}' 时发生异常: {e} ---\n")
            finally:
                # 无论成功失败，都更新一次UI
                self.signals.task_update.emit()

        self.signals.log_message.emit("\n\n****** 所有队列任务已执行完毕或被中止。 ******\n")
        # 在所有任务结束后，执行通知脚本
        notification_log = run_notification_script()
        self.signals.log_message.emit(notification_log)
        # 发送队列完成信号
        self.signals.queue_finished.emit()

    def stop(self):
        """停止任务循环，并使用wsl kill命令终止WSL中的进程组。"""
        self.is_running = False
        if self.linux_pid:
            self.signals.log_message.emit(f"日志: 正在通过 WSL kill 命令终止进程组 {self.linux_pid}...\n")
            # --- 精确打击 ---
            # `kill -9 -<PID>`: 发送 SIGKILL 信号给整个进程组。
            # 这里的负号至关重要，它表示目标是进程组，而不是单个进程。
            kill_command = ["wsl.exe", "kill", "-9", f"-{self.linux_pid}"]
            try:
                # 使用 subprocess.run 执行一次性的命令更简单
                result = subprocess.run(kill_command, capture_output=True, encoding='utf-8', errors='replace')
                if result.returncode == 0:
                    self.signals.log_message.emit("日志: 进程组终止信号已成功发送。\n")
                else:
                    self.signals.log_message.emit(f"日志: 发送终止信号失败: {result.stderr}\n")
            except Exception as e:
                self.signals.log_message.emit(f"日志: 执行 kill 命令时发生错误: {e}\n")

        # 无论是否成功发送kill，都尝试关闭 QProcess 句柄
        if self.current_process:
            self.current_process.terminate()