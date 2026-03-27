# batch/agent/yingbo/node_executor_yingbo.py
import argparse
import subprocess
import sys
import time
from pathlib import Path


def run_node():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hash", required=True)
    parser.add_argument("--out_name", required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    # 路径处理
    repo_root = Path(__file__).resolve().parent.parent.parent.parent
    work_dir = repo_root / "results" / args.out_name
    work_dir.mkdir(parents=True, exist_ok=True)

    main_py = repo_root / "main.py"

    print(f"[Agent] 启动任务: {args.out_name}", flush=True)

    # 直接执行，无需 self-destruct
    try:
        process = subprocess.Popen(
            [sys.executable, str(main_py), "-o", str(work_dir), "-c", args.config],
            stdout=sys.stdout,
            stderr=sys.stderr,
            universal_newlines=True
        )
        ret_code = process.wait()
        print(f"[Agent] 核心程序执行完毕，退出码: {ret_code}", flush=True)
        sys.exit(ret_code)
    except Exception as e:
        print(f"[Agent] 运行异常: {e}", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    run_node()