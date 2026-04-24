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

from batch.logger_manager import create_standard_log_manager
from batch.manager_api import JobStatus, BaseComputeManager

from utils.project_config import FILENAME_QUEUE, FILENAME_HISTORY


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


def run_batch(job_dir_win: str, manager_type: str = "yingbo"):
    """
    批量执行任务

    Args:
        job_dir_win: Windows 格式的工作目录路径
        manager_type: 计算管理器类型 ("wsl", "gongji", "yingbo")
    """
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

    total_tasks = len(tasks)
    completed_count = 0
    skipped_count = 0

    # 创建总的日志目录
    logs_base_dir = job_path / "logs"
    logs_base_dir.mkdir(exist_ok=True)

    # 批处理日志文件
    batch_timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    batch_log_path = logs_base_dir / f"batch_{batch_timestamp}.log"
    # 创建日志管理器（自动包含控制台 + batch 文件）
    log_manager = create_standard_log_manager(batch_log_path, tag="BATCH_RUNNER")

    log_manager.log_system(f"批处理开始，共 {total_tasks} 个任务")
    log_manager.log_system(f"使用管理器: {manager_type}")

    for idx, task in enumerate(tasks, 1):
        task_hash = task['hash']
        task_params = task['params']
        task_name = task['task_name']

        # 只有历史状态是 success 才跳过
        if task_hash in history_hash_status and history_hash_status[task_hash] == "success":
            log_manager.log_system(f"[Batch] >>> 跳过任务: {task_name} (Hash: {task_hash[:8]}...) 已成功执行")
            skipped_count += 1
            continue

        # 如果是失败 / 无记录 → 执行
        if task_hash in history_hash_status:
            log_manager.log_system(f"[Batch] >>> 任务曾失败，将重新执行: {task_name} (Hash: {task_hash[:8]}...)")
        else:
            log_manager.log_system(f"\n[Batch] >>> 准备启动新任务: {task_name}")

        # 记录开始时间
        start_time = datetime.datetime.now()

        # 使用 with 申请任务日志
        with log_manager.create_task_context(task_name, logs_base_dir):
            log_manager.log_task_start(task_name, task_hash)

            # 实例化 Manager
            manager: BaseComputeManager
            if manager_type == "wsl":
                manager = WSLComputeManager()
            elif manager_type == "gongji":
                manager = GongjiComputeManager()
            elif manager_type == "yingbo":
                manager = YingboComputeManager()
            else:
                raise ValueError(f"不支持的管理器类型: {manager_type}")

            # 提交任务
            try:
                manager.submit(
                    task_hash=task_hash,
                    params=task_params,
                    output_dir_name=task_name,
                    rel_job_path=str(rel_job_dir).replace("\\", "/")
                )
                log_manager.log_system("任务已提交，等待执行...")
            except Exception as e:
                log_manager.log_system(f"任务提交失败: {e}")
                # 记录失败历史
                end_time = datetime.datetime.now()
                duration = (end_time - start_time).total_seconds()
                _write_history(history_file, task_hash, task_name, "failed",
                               start_time, end_time, duration, rel_job_dir, task_params)
                continue

            # 监控日志与状态
            last_status = JobStatus.PENDING
            update_interval = 2  # 秒

            while True:
                try:
                    status = manager.get_status()

                    # 获取新的日志行
                    new_lines = manager.get_logs()
                    for line in new_lines:
                        log_manager.log_raw(line)

                    # 状态变化时记录
                    if status != last_status:
                        log_manager.log_system(f"状态变化: {last_status.name} -> {status.name}")
                        last_status = status

                    # 检查是否结束
                    if status not in [JobStatus.RUNNING, JobStatus.PENDING]:
                        break

                    sleep(update_interval)

                except KeyboardInterrupt:
                    log_manager.log_system("收到中断信号，正在停止任务...")
                    manager.interrupt()
                    # 等待一下让中断生效
                    sleep(3)
                    break
                except Exception as e:
                    log_manager.log_system(f"监控异常: {e}")
                    sleep(update_interval)

            # 任务结束
            end_time = datetime.datetime.now()
            duration = (end_time - start_time).total_seconds()

            final_status = "success" if last_status == JobStatus.SUCCESS else "failed"
            log_manager.log_task_end(task_name, final_status, duration)

            # 记录历史
            _write_history(history_file, task_hash, task_name, final_status,
                           start_time, end_time, duration, rel_job_dir, task_params)

            completed_count += 1

            # 在批处理日志中也记录
            log_manager.log_system(f"任务完成 [{idx}/{total_tasks}]: {task_name} -> {final_status} (耗时: {duration:.2f}s)")

    # 批处理结束
    log_manager.log_system(f"{'=' * 60}")
    log_manager.log_system(f"批处理完成！")
    log_manager.log_system(f"总计: {total_tasks} 个任务")
    log_manager.log_system(f"执行: {completed_count} 个")
    log_manager.log_system(f"跳过: {skipped_count} 个")
    log_manager.log_system(f"日志保存在: {logs_base_dir}")
    log_manager.log_system(f"{'=' * 60}")

    # 关闭所有日志器（运行结束，实际上它们也会被GC）
    log_manager.close_all()


def _write_history(history_file: Path, task_hash: str, task_name: str,
                   status: str, start_time: datetime.datetime,
                   end_time: datetime.datetime, duration: float,
                   rel_job_dir, task_params: dict):
    """写入历史记录"""
    history_entry = {
        "hash": task_hash,
        "task_name": task_name,
        "status": status,
        "start_at": start_time.strftime('%Y-%m-%d %H:%M:%S'),
        "completed_at": end_time.strftime('%Y-%m-%d %H:%M:%S'),
        "duration_sec": round(duration, 2),
        "rel_job_path": str(rel_job_dir),
        "params": task_params
    }
    with open(history_file, 'a', encoding='utf-8') as hf:
        hf.write(json.dumps(history_entry) + '\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plasma Simulation Batch Runner - 一个简单的文件驱动执行器。")
    parser.add_argument("work_dir", type=str, help="工作目录的路径，必须包含 queue.jsonl 文件。")
    parser.add_argument("--manager", type=str, default="yingbo",
                        choices=["wsl", "gongji", "yingbo"],
                        help="选择计算管理器类型 (默认: yingbo)")

    args = parser.parse_args()

    run_batch(args.work_dir, args.manager)
