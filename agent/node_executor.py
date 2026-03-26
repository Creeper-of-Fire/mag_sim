# agent/node_executor.py
import argparse
import json
import subprocess
import sys
import os
import time
import urllib.request
import urllib.parse
import urllib.error
from pathlib import Path

AGENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = AGENT_DIR.parent


def self_destruct(task_name, token, base_url):
    """
    使用 Python 原生 urllib 实现自我销毁，无需安装 requests。
    """
    print(f"\n[Agent] 任务完成，正在尝试自我销毁 (Task Name: {task_name})...", flush=True)

    headers = {
        "token": token,
        "timestamp": str(int(time.time() * 1000)),
        "version": "1.0.0",
        "Content-Type": "application/json"
    }

    try:
        # 1. 在列表中根据名字搜寻 ID (GET 请求)
        list_url = f"{base_url}/api/deployment/task/list?task_name={urllib.parse.quote(task_name)}"
        req = urllib.request.Request(list_url, headers=headers)

        with urllib.request.urlopen(req, timeout=10) as response:
            res_data = json.loads(response.read().decode())

            if res_data.get("code") == "0000":
                tasks = res_data.get("data", {}).get("results", [])
                my_task = next((t for t in tasks if t['task_name'] == task_name), None)

                if my_task:
                    task_id = my_task['task_id']
                    print(f"[Agent] 确认 Task ID: {task_id}. 发送删除指令...", flush=True)

                    # 2. 发送删除请求 (POST 请求)
                    del_url = f"{base_url}/api/deployment/task/delete"
                    del_payload = json.dumps({"task_id": task_id}).encode('utf-8')
                    del_req = urllib.request.Request(del_url, data=del_payload, headers=headers, method='POST')

                    with urllib.request.urlopen(del_req, timeout=10) as del_response:
                        print(f"[Agent] 删除请求响应: {del_response.read().decode()}", flush=True)
                else:
                    print("[Agent] 未在列表中找到匹配的任务名称。")
            else:
                print(f"[Agent] 获取任务列表失败: {res_data.get('message')}")

    except urllib.error.HTTPError as e:
        print(f"[Agent] HTTP 错误: {e.code} - {e.reason}", flush=True)
        error_body = e.read().decode()
        print(f"[Agent] 错误详情: {error_body}", flush=True)
    except Exception as e:
        print(f"[Agent] 自我销毁异常: {e}", flush=True)


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
    print(f"[Agent] main.py 退出码: {ret_code}", flush=True)

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