# agent/node_executor.py
import argparse
import json
import subprocess
import sys
import os
import shutil
from pathlib import Path

# 这里假设 agent 目录和 simulation 目录在同级，可以引用 main.py
AGENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = AGENT_DIR.parent


def run_node():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hash", required=True)
    parser.add_argument("--out_name", required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    print(f"\n[Agent] Worker process started for task: {args.out_name}", flush=True)

    # 1. 准备本地临时工作空间 (服务器硬盘上的空间)
    # 在真实云端，这可能是 /tmp/job
    work_dir = REPO_ROOT / "remote_workdir" / args.out_name
    work_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Agent] 启动节点任务: {args.hash}")
    print(f"[Agent] 本地工作目录: {work_dir}")

    # 2. 调用核心模拟脚本
    # 注意：我们这里引用主项目的 main.py
    main_py = REPO_ROOT / "main.py"

    # 将输出重定向到模拟器的 stdout，这样本地 Manager 能接到
    process = subprocess.Popen(
        [sys.executable, str(main_py), "-o", str(work_dir), "-c", args.config],
        stdout=sys.stdout,
        stderr=sys.stderr
    )

    ret_code = process.wait()

    if ret_code == 0:
        print(f"[Agent] 模拟成功完成。开始后期处理...")

        # 3. 模拟打包与上传 (核心解耦点)
        # 在这里执行 tar, zip 或调用对象存储 API
        # 目前我们 mock 它：移动到项目的总结果目录
        results_repo = REPO_ROOT / "sim_results_storage"
        results_repo.mkdir(exist_ok=True)

        target_path = results_repo / f"{args.out_name}.zip"
        print(f"[Agent] 正在打包数据并‘上传’至: {target_path}")

        # 简单模拟打包 (实际上直接挪过去)
        shutil.make_archive(str(results_repo / args.out_name), 'zip', work_dir)

        print(f"[Agent] 上传完成。")
        # 清理工作目录 (Ephemeral 特性)
        shutil.rmtree(work_dir)
    else:
        print(f"[Agent] 模拟失败，退出码: {ret_code}")
        sys.exit(ret_code)


if __name__ == "__main__":
    run_node()