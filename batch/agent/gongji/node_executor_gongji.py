# # batch/agent/gongji/node_executor_gongji.py
import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


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
        # 1. 在列表中根据名字搜寻 ID
        # 构造查询参数
        params = {
            "type": "Deployment",
            "status": "Running,Pending,Paused",  # 搜索所有非删除状态
            "search_value": task_name,
            "page": 1,
            "page_size": 100
        }
        query_string = urllib.parse.urlencode(params)
        list_url = f"{base_url}/api/deployment/task/search?{query_string}"

        print(f"[Agent] 请求 URL: {list_url}", flush=True)
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
                    del_url = f"{base_url}/api/deployment/task/pause"
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
    parser.add_argument("--out_name", required=True)
    parser.add_argument("--work_dir", required=True)
    parser.add_argument("--main_py", required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    # 创建日志目录
    log_dir = work_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # 生成带时间戳的日志文件名
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"{args.out_name}_{timestamp}.log"

    print(f"[Agent] 核心任务启动: {args.hash}", flush=True)
    print(f"[Agent] 日志将保存至: {log_file}", flush=True)

    ret_code = 0

    try:
        # 同时输出到控制台和日志文件
        with open(log_file, 'w', buffering=1) as f:
            f.write(f"=== 任务启动时间: {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
            f.write(f"任务名称: {args.out_name}\n")
            f.write(f"任务哈希: {args.hash}\n")
            f.write(f"配置: {args.config}\n")
            f.write(f"工作目录: {work_dir}\n")
            f.write("=" * 60 + "\n\n")

            process = subprocess.Popen(
                [sys.executable, args.main_py, "-o", str(work_dir), "-c", args.config],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # 合并 stderr 到 stdout
                bufsize=0,
                universal_newlines=True
            )

            # 实时输出并写入日志
            for line in process.stdout:
                print(line, end='', flush=True)
                f.write(line)

            ret_code = process.wait()

            f.write(f"\n{'=' * 60}\n")
            f.write(f"任务结束时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"退出码: {ret_code}\n")
            f.write(f"{'=' * 60}\n")

        print(f"[Agent] main.py 执行完成，退出码: {ret_code}", flush=True)
        print(f"[Agent] 完整日志已保存至: {log_file}", flush=True)

    except Exception as e:
        print(f"[Agent] main.py 执行异常: {e}", flush=True)
        # 异常情况也写入日志
        with open(log_file, 'a', buffering=1) as f:
            f.write(f"\n{'=' * 60}\n")
            f.write(f"异常时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"异常信息: {e}\n")
            f.write(f"{'=' * 60}\n")
        ret_code = 1

    # 无论模拟成功失败，都尝试自我销毁
    token = os.getenv("GONGJI_TOKEN")
    base_url = os.getenv("GONGJI_BASE_URL")

    if token and base_url:
        print("[Agent] 等待 5 秒后尝试自我销毁...", flush=True)
        time.sleep(5)
        try:
            self_destruct(args.out_name, token, base_url)
        except Exception as e:
            print(f"[Agent] 自我销毁过程出错（不影响退出）: {e}", flush=True)
    else:
        print("[Agent] 未检测到 GONGJI_TOKEN，跳过自我销毁。", flush=True)

    # 强制以 0 退出
    print("[Agent] 容器即将退出（退出码 0）", flush=True)
    sys.exit(0)


if __name__ == "__main__":
    run_node()
