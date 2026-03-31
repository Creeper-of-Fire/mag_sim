# batch/agent/yingbo/node_executor_yingbo.py
import argparse
import subprocess
import sys
from pathlib import Path


def run_node():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hash", required=True)
    parser.add_argument("--out_name", required=True)
    parser.add_argument("--work_dir", required=True)
    parser.add_argument("--main_py", required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Agent] 启动任务: {args.out_name}", flush=True)
    print(f"[Agent] 工作目录: {work_dir}", flush=True)

    try:
        process = subprocess.Popen(
            [sys.executable, args.main_py, "-o", str(work_dir), "-c", args.config],
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
