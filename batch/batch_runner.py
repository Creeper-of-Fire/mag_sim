# Plasma_Simulation/batch/batch_runner.py

import argparse
import datetime
import hashlib
import json
import os
import subprocess
import sys

from dotenv import load_dotenv

# --- 从 .env 文件加载配置 ---
# 通过从此脚本位置向上两级目录来找到项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_FILE_PATH = os.path.join(PROJECT_ROOT, '.env.warpx')

if not load_dotenv(ENV_FILE_PATH):
    print(f"错误: 未找到 .env.warpx 文件于 '{ENV_FILE_PATH}'。程序退出。", file=sys.stderr)
    sys.exit(1)

# --- 从环境变量获取常量 ---
PROJECT_ROOT_WSL = os.getenv('PROJECT_ROOT_WSL')
MAIN_SCRIPT_PATH = os.path.join(PROJECT_ROOT_WSL, 'main.py')
CONDA_INIT_PATH = os.getenv('CONDA_INIT_PATH')
CONDA_ENV_NAME = os.getenv('CONDA_ENV_NAME')

# --- 验证关键路径是否已加载 ---
if not all([PROJECT_ROOT_WSL, CONDA_INIT_PATH, CONDA_ENV_NAME]):
    print("错误: .env.warpx 文件中缺少一个或多个必要的变量 (PROJECT_ROOT_WSL, CONDA_INIT_PATH, CONDA_ENV_NAME)。", file=sys.stderr)
    sys.exit(1)


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


def create_task_hash(params: dict) -> str:
    """
    为参数字典创建一个稳定、唯一的SHA256哈希值。
    通过对键进行排序来确保哈希的一致性。
    """
    # 将字典转换为规范的JSON字符串（排序键，无空格）
    param_string = json.dumps(params, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(param_string.encode('utf-8')).hexdigest()


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


def run_batch(work_dir: str):
    """
    批处理引擎的主函数。
    """
    # --- 1. 设置路径和文件 ---
    work_dir = os.path.abspath(work_dir)
    queue_file = os.path.join(work_dir, 'queue.jsonl')
    history_file = os.path.join(work_dir, 'history.jsonl')
    logs_dir = os.path.join(work_dir, 'logs')

    os.makedirs(logs_dir, exist_ok=True)

    run_timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file_path = os.path.join(logs_dir, f'batch_run_{run_timestamp}.log')

    log_file_handle = None
    try:
        log_file_handle = open(log_file_path, 'w', encoding='utf-8')

        log_system_message(log_file_handle, f"工作目录: {work_dir}")
        log_system_message(log_file_handle, f"任务队列文件: {queue_file}")
        log_system_message(log_file_handle, f"历史记录文件: {history_file}")
        log_system_message(log_file_handle, f"日志输出到: {log_file_path}")

        if not os.path.exists(queue_file):
            log_system_message(log_file_handle, "错误: 队列文件 'queue.jsonl' 不存在。正在退出。")
            return

        # --- 2. 加载历史记录 ---
        history_hashes = load_history_hashes(history_file)
        log_system_message(log_file_handle, f"已加载 {len(history_hashes)} 条历史记录。")

        # --- 3. 循环处理队列 ---
        with open(queue_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                line_num = i + 1
                try:
                    task_definition = json.loads(line)
                    # 验证任务定义格式
                    if 'params' not in task_definition or 'output_dir' not in task_definition:
                        log_system_message(log_file_handle, f"错误: 第 {line_num} 行任务定义缺少 'params' 或 'output_dir' 键，已跳过。")
                        continue

                    task_params = task_definition['params']
                    output_dir = task_definition['output_dir']

                except json.JSONDecodeError:
                    log_system_message(log_file_handle, f"错误: 第 {line_num} 行JSON格式错误，已跳过。")
                    continue

                task_hash = create_task_hash(task_params)

                # --- 4. 检查是否需要跳过 ---
                if task_hash in history_hashes:
                    log_system_message(log_file_handle, f"任务 {line_num} (Hash: {task_hash[:8]}...) 已在历史记录中，跳过。")
                    continue

                # --- 5. 执行任务 ---
                log_system_message(log_file_handle, f"开始执行任务 {line_num} (Hash: {task_hash[:8]}...) -> 输出到 '{output_dir}'")

                # 记录开始时间
                started_at = datetime.datetime.now().isoformat()

                config_json_str = json.dumps(task_params)

                # 构建在 WSL 内部执行的完整命令
                command_in_wsl = (
                    f"source {CONDA_INIT_PATH} && "
                    f"conda activate {CONDA_ENV_NAME} && "
                    f"exec python {MAIN_SCRIPT_PATH} -o '{output_dir}' -c '{config_json_str}'"
                )

                # 使用 Popen 启动子进程以实时获取输出
                process = subprocess.Popen(
                    ["bash", "-c", command_in_wsl],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding='utf-8',
                    errors='replace'
                )

                # 实时流式传输子进程的输出 (Tee策略)
                if process.stdout:
                    for output_line in iter(process.stdout.readline, ''):
                        # 直接打印和写入，不加任何前缀
                        print(output_line, end='', flush=True)
                        log_file_handle.write(output_line)
                        log_file_handle.flush()

                # 等待进程结束并获取返回码
                return_code = process.wait()
                status = "success" if return_code == 0 else "failed"

                log_system_message(log_file_handle, f"任务 {line_num} 执行完毕。状态: {status} (返回码: {return_code})")

                # --- 6. 记录到历史文件 (Append-only) ---
                history_entry = {
                    "hash": task_hash,
                    "params": task_params,
                    "output_dir": output_dir, # 记录本次运行的输出目录
                    "status": status,
                    "return_code": return_code,
                    "started_at": started_at,
                    "completed_at": datetime.datetime.now().isoformat()
                }
                with open(history_file, 'a', encoding='utf-8') as hf:
                    hf.write(json.dumps(history_entry) + '\n')

                # 更新内存中的哈希集合
                history_hashes.add(task_hash)

        log_system_message(log_file_handle, "所有队列任务已处理完毕。")

    finally:
        if log_file_handle:
            log_file_handle.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plasma Simulation Batch Runner - 一个简单的文件驱动执行器。")
    parser.add_argument("work_dir", type=str, help="工作目录的路径，必须包含 queue.jsonl 文件。")

    args = parser.parse_args()

    run_batch(args.work_dir)
