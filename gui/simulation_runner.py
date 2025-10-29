# gui/simulation_runner.py

# !/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import json
from PySide6.QtCore import QObject, Signal, QProcess

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

    def run(self):
        """
        依次执行队列中的所有任务。
        """
        # 定位主模拟脚本 main.py 的路径
        # sys.argv[0] 是 main_gui.py，所以它的目录就是项目根目录
        main_script_path = os.path.join(os.path.dirname(os.path.realpath(sys.argv[0])), 'main.py')

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
                command = [sys.executable, main_script_path, "-c", config_json]

                # QProcess 是 PySide/PyQt 的 subprocess 替代品，能更好地集成事件循环
                process = QProcess()
                # 合并标准输出和标准错误通道，方便一同读取
                process.setProcessChannelMode(QProcess.MergedChannels)

                # 连接信号，实时获取子进程的输出
                process.readyReadStandardOutput.connect(
                    lambda: self.signals.log_message.emit(process.readAllStandardOutput().data().decode('utf-8', errors='ignore')))

                # 启动子进程
                process.start(command[0], command[1:])
                process.waitForFinished(-1)  # -1 表示无限期阻塞等待，直到进程结束

                # 检查子进程的退出码
                if process.exitCode() == 0:
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

        self.signals.log_message.emit("\n\n****** 所有队列任务已执行完毕。 ******\n")
        # 发送队列完成信号
        self.signals.queue_finished.emit()

    def stop(self):
        """停止任务循环。"""
        self.is_running = False