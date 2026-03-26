# Plasma_Simulation/batch/batch_runner.py

import argparse
import datetime
import hashlib
import json
import os
import sys
from pathlib import Path
from time import sleep

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from batch.manager_api import JobStatus
from batch.wsl_manager import WSLComputeManager

from utils.project_config import FILENAME_QUEUE, FILENAME_HISTORY


# --- 系统日志格式化 ---
def log_system_message(log_file_handle, message):
    """
    记录并打印带有时间戳和特殊标记的系统级日志。
    实现了 Tee 策略：同时输出到标准输出和日志文件。
    """
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    formatted_message = f"[{timestamp}][BATCH_RUNNER] {message}"

    # 打印到控制台 (stdout)
    print(formatted_message, flush=True)

    # 写入日志文件
    if log_file_handle:
        log_file_handle.write(formatted_message + '\n')
        log_file_handle.flush()


def load_history_hashes(history_file_path: str) -> set:
    """
    从 history.jsonl 文件中加载所有已完成任务的哈希值。
    返回一个集合(set)以便快速查找。
    """
    hashes = set()
    if os.path.exists(history_file_path):
        with open(history_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    if 'hash' in entry:
                        hashes.add(entry['hash'])
                except json.JSONDecodeError:
                    # 忽略格式错误的行
                    continue
    return hashes


def run_batch(work_dir_win: str):
    queue_file = os.path.join(work_dir_win, FILENAME_QUEUE)
    history_file = os.path.join(work_dir_win, FILENAME_HISTORY)

    if not os.path.exists(queue_file):
        print(f"错误: 队列文件 {FILENAME_QUEUE} 不存在。")
        return

    # 加载历史记录
    history_hashes = load_history_hashes(history_file)
    print(f"[Batch] 已加载 {len(history_hashes)} 条历史记录")

    with open(queue_file, 'r', encoding='utf-8') as f:
        tasks = [json.loads(line) for line in f]

    for task in tasks:
        # 1. 实例化 Manager (这里是唯一可以手动指定实现类的地方)
        manager = WSLComputeManager()

        task_hash = task['hash']
        task_params = task['params']
        task_name = task['task_name']

        # 检查是否已执行过
        if task_hash in history_hashes:
            print(f"[Batch] >>> 跳过任务: {task_name} (Hash: {task_hash[:8]}...) 已在历史记录中")
            continue

        print(f"\n[Batch] >>> 准备启动任务: {task_name}")

        # 2. 提交
        manager.submit(task_hash, task_params, task_name)

        # 3. 监控日志与状态
        while True:
            # 实时获取增量日志
            new_logs = manager.get_logs()
            for line in new_logs:
                print(line, end='', flush=True)

            status = manager.get_status()
            if status != JobStatus.RUNNING:
                break

            sleep(0.5)

        # 4. 记录历史记录
        print(f"\n[Batch] >>> 任务结束。最终状态: {status.name}")

        history_entry = {
            "hash": task_hash,
            "task_name": task_name,
            "status": "success" if status == JobStatus.SUCCESS else "failed",
            "completed_at": datetime.datetime.now().isoformat()
        }
        with open(history_file, 'a', encoding='utf-8') as hf:
            hf.write(json.dumps(history_entry) + '\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plasma Simulation Batch Runner - 一个简单的文件驱动执行器。")
    parser.add_argument("work_dir", type=str, help="工作目录的路径，必须包含 queue.jsonl 文件。")

    args = parser.parse_args()

    run_batch(args.work_dir)
