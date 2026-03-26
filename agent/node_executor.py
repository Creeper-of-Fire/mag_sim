# agent/node_executor.py
import argparse
import json
import subprocess
import sys
import os
import time
import requests
from pathlib import Path

AGENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = AGENT_DIR.parent


def self_destruct(task_name, token, base_url):
    """
    通过 task_name 找到自己的 task_id 并销毁任务。
    """
    print(f"\n[Agent] 任务完成，正在尝试自我销毁 (Task Name: {task_name})...", flush=True)
    headers = {
        "token": token,
        "timestamp": str(int(time.time() * 1000)),
        "version": "1.0.0"
    }

    try:
        # 1. 在列表中根据名字搜寻 ID
        list_url = f"{base_url}/api/deployment/task/list"
        resp = requests.get(list_url, headers=headers, params={"task_name": task_name})
        data = resp.json()

        if data.get("code") == "0000":
            tasks = data.get("data", {}).get("results", [])
            my_task = next((t for t in tasks if t['task_name'] == task_name), None)

            if my_task:
                task_id = my_task['task_id']
                print(f"[Agent] 确认 Task ID: {task_id}. 发送删除指令...", flush=True)

                # 2. 发送删除请求
                del_url = f"{base_url}/api/deployment/task/delete"
                del_resp = requests.post(del_url, headers=headers, json={"task_id": task_id})
                print(f"[Agent] 删除请求结果: {del_resp.text}", flush=True)
            else:
                print("[Agent] 未在列表中找到匹配的任务名称，可能已被手动删除。")
        else:
            print(f"[Agent] 获取任务列表失败: {data.get('message')}")
    except Exception as e:
        print(f"[Agent] 自我销毁流程异常: {e}")


def run_node():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hash", required=True)
    parser.add_argument("--out_name", required=True)  # 对应 Manager 侧的 task_name
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    # 这里的 work_dir 设置在共享存储内，方便持久化查看
    work_dir = REPO_ROOT / "results" / args.out_name
    work_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Agent] 核心任务启动: {args.hash}", flush=True)

    # 调用核心模拟脚本 (main.py)
    main_py = REPO_ROOT / "main.py"
    process = subprocess.Popen(
        [sys.executable, str(main_py), "-o", str(work_dir), "-c", args.config],
        stdout=sys.stdout,
        stderr=sys.stderr
    )

    ret_code = process.wait()

    # 无论成功失败，尝试自杀以节省算力费
    token = os.getenv("GONGJI_TOKEN")
    base_url = os.getenv("GONGJI_BASE_URL")

    if token and base_url:
        # 给日志缓冲区留一点时间上传
        time.sleep(5)
        self_destruct(args.out_name, token, base_url)
    else:
        print("[Agent] 未检测到 GONGJI_TOKEN，跳过自我销毁。")

    sys.exit(ret_code)


if __name__ == "__main__":
    run_node()