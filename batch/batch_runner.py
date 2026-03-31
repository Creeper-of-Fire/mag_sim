# Plasma_Simulation/batch/batch_runner.py

import argparse
import datetime
import json
import os
import sys
from pathlib import Path
from time import sleep

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from batch.agent.wsl.wsl_manager import WSLComputeManager
from batch.agent.gongji.gongji_manager import GongjiComputeManager
from batch.agent.yingbo.yingbo_manager import YingboComputeManager

from batch.manager_api import JobStatus, BaseComputeManager

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


def load_history_hashes(history_file_path: str) -> dict:
    """
    从 history.jsonl 文件中加载所有任务哈希值 + 对应状态。
    返回字典：{hash: "success"/"failed", ...}
    """
    hash_status_map = {}
    if os.path.exists(history_file_path):
        with open(history_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    if 'hash' in entry and 'status' in entry:
                        hash_status_map[entry['hash']] = entry['status']
                except json.JSONDecodeError:
                    continue
    return hash_status_map


def run_batch(job_dir_win: str):
    job_path = Path(job_dir_win).resolve()
    job_name = job_path.name  # 获取 {job_name}

    queue_file = job_path / FILENAME_QUEUE
    history_file = job_path / FILENAME_HISTORY

    results_base_dir = job_path / "sim_results"
    results_base_dir.mkdir(exist_ok=True)

    # 计算 job_dir 相对于项目根目录的相对路径，用于传给云端/WSL
    # 假设项目根目录是 job_dir 的上两级 (sim_jobs/..)
    # 或者通过 PROJECT_ROOT 自动计算
    try:
        rel_job_dir = job_path.relative_to(root_dir)
    except ValueError:
        # 如果 job_dir 不在项目路径下，可能需要处理，这里暂设为绝对路径
        rel_job_dir = job_path

    if not os.path.exists(queue_file):
        print(f"错误: 队列文件 {FILENAME_QUEUE} 不存在。")
        return

    # 加载历史记录
    history_hash_status = load_history_hashes(history_file)
    print(f"[Batch] 已加载 {len(history_hash_status)} 条历史记录")

    with open(queue_file, 'r', encoding='utf-8') as f:
        tasks = [json.loads(line) for line in f]

    for task in tasks:
        task_hash = task['hash']
        task_params = task['params']
        task_name = task['task_name']

        # 只有历史状态是 success 才跳过
        if task_hash in history_hash_status and history_hash_status[task_hash] == "success":
            print(f"[Batch] >>> 跳过任务: {task_name} (Hash: {task_hash[:8]}...) 已成功执行")
            continue

        # 如果是失败 / 无记录 → 执行
        if task_hash in history_hash_status:
            print(f"[Batch] >>> 任务曾失败，将重新执行: {task_name} (Hash: {task_hash[:8]}...)")
        else:
            print(f"\n[Batch] >>> 准备启动新任务: {task_name}")

        # 记录开始时间
        start_time = datetime.datetime.now()


        # 实例化 Manager (这里是唯一可以手动指定实现类的地方)
        manager: BaseComputeManager = WSLComputeManager()

        # 提交
        manager.submit(
            task_hash=task_hash,
            params=task_params,
            output_dir_name=task_name,
            rel_job_path=str(rel_job_dir).replace("\\", "/") # 统一用 Linux 斜杠
        )

        print(f"[Batch] 正在等待云端初始化...", flush=True)

        # 3. 监控日志与状态
        while True:
            status = manager.get_status()

            # 统一获取“信息流”（包含系统事件和业务日志）
            new_lines = manager.get_logs()
            for line in new_lines:
                print(line, end='', flush=True)

            # 只要没结束，就一直等
            if status not in [JobStatus.RUNNING, JobStatus.PENDING]:
                break

            sleep(2)

        # 任务结束，记录结束时间和时长
        end_time = datetime.datetime.now()
        duration = (end_time - start_time).total_seconds()

        # 4. 记录历史记录
        print(f"\n[Batch] >>> 任务结束。最终状态: {status.name}")

        history_entry = {
            "hash": task_hash,
            "task_name": task_name,
            "status": "success" if status == JobStatus.SUCCESS else "failed",
            "start_at": start_time.strftime('%Y-%m-%d %H:%M:%S'),
            "completed_at": end_time.strftime('%Y-%m-%d %H:%M:%S'),
            "duration_sec": round(duration, 2),
            "rel_job_path": str(rel_job_dir),
            "params": task_params  # 记录当时运行的参数快照
        }
        with open(history_file, 'a', encoding='utf-8') as hf:
            hf.write(json.dumps(history_entry) + '\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plasma Simulation Batch Runner - 一个简单的文件驱动执行器。")
    parser.add_argument("work_dir", type=str, help="工作目录的路径，必须包含 queue.jsonl 文件。")

    args = parser.parse_args()

    run_batch(args.work_dir)
