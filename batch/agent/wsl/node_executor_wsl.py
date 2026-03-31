# agent/node_executor.py
import argparse
import subprocess
import sys
import shutil
from pathlib import Path


def run_node():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hash", required=True)
    parser.add_argument("--out_name", required=True)
    parser.add_argument("--work_dir", required=True)
    parser.add_argument("--main_py", required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    print(f"\n[Agent] Worker process started for task: {args.out_name}", flush=True)

    # 1. 准备工作空间
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Agent] 启动节点任务: {args.hash}")
    print(f"[Agent] 本地工作目录: {work_dir}")

    # 2. 调用核心模拟脚本

    # 将输出重定向到模拟器的 stdout，这样本地 Manager 能接到
    process = subprocess.Popen(
        [sys.executable, args.main_py, "-o", str(work_dir), "-c", args.config],
        stdout=sys.stdout,
        stderr=sys.stderr
    )

    ret_code = process.wait()

    # if ret_code == 0:
    #     print(f"[Agent] 模拟成功。正在打包数据...", flush=True)
    #
    #     storage_path = work_dir / "storage"
    #     storage_path.mkdir(parents=True, exist_ok=True)
    #
    #     # 打包到指定的存储目录
    #     archive_base = storage_path / args.out_name
    #     shutil.make_archive(str(archive_base), 'zip', work_dir)
    #
    #     print(f"[Agent] 归档已保存至: {archive_base}.zip", flush=True)
    #
    #     # 清理临时工作目录
    #     print(f"[Agent] 清理临时目录: {work_dir}", flush=True)
    #     shutil.rmtree(work_dir)
    #
    #     sys.exit(0)
    # else:
    #     print(f"[Agent] 核心程序返回非零退出码: {ret_code}", flush=True)
    #     sys.exit(ret_code)


if __name__ == "__main__":
    run_node()
