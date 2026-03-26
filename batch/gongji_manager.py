# batch/gongji_manager.py
import json
import os
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

from .manager_api import BaseComputeManager, JobStatus

current_dir = Path(__file__).resolve().parent
env_path = current_dir / ".service.env"
load_dotenv(env_path)


class GongjiComputeManager(BaseComputeManager):
    def __init__(self):
        self.token = os.getenv("GONGJI_TOKEN")
        self.base_url = os.getenv("GONGJI_BASE_URL")
        self.task_id = None
        self.task_name = None
        self.service_id = None
        self.point_id = None

        self.last_log_line_count = 0
        self._last_event_count = 0  # 用于事件增量打印
        self._status = JobStatus.PENDING

    def _get_headers(self):
        return {
            "token": self.token,
            "timestamp": str(int(time.time() * 1000)),
            "version": "1.0.0",
            "Content-Type": "application/json"
        }

    def _find_existing_task(self, name):
        """在线上搜索同名任务，实现持久化找回"""
        try:
            resp = requests.get(f"{self.base_url}/api/deployment/task/list",
                                headers=self._get_headers(), params={"task_name": name})
            tasks = resp.json().get("data", {}).get("results", [])
            for t in tasks:
                if t['task_name'] == name:
                    return t['task_id']
        except:
            pass
        return None

    def _update_point_id(self):
        try:
            resp = requests.get(f"{self.base_url}/api/deployment/task/point_list",
                                headers=self._get_headers(), params={"task_id": self.task_id})
            points = resp.json().get("data", {}).get("results", [])
            if points: self.point_id = points[0]["point_id"]
        except:
            pass

    def _fetch_incremental_events(self) -> list[str]:
        """私有方法：抓取增量系统事件并格式化为日志行"""
        try:
            resp = requests.get(f"{self.base_url}/api/deployment/task/pod_event",
                                headers=self._get_headers(), params={"point_id": self.point_id}, timeout=5)
            events = resp.json().get("data", [])
            if not isinstance(events, list): return []

            new_events = events[self._last_event_count:]
            self._last_event_count = len(events)

            results = []
            for e in new_events:
                reason = e.get('reason', 'Event')
                msg = e.get('message', '')
                # 将事件包装成看起来像日志的样子
                results.append(f">>> [CLOUD_SYSTEM] {reason}: {msg}\n")
            return results
        except:
            return []

    def submit(self, task_hash: str, params: dict, output_dir_name: str):
        """
        提交任务到共绩云平台
        """
        self.task_name = output_dir_name

        # --- 持久化找回逻辑 ---
        existing_id = self._find_existing_task(self.task_name)
        if existing_id:
            print(f"[Gongji] 检测到已存在同名任务 (ID: {existing_id})，正在直接挂载监控...")
            self.task_id = existing_id
            self._status = JobStatus.PENDING  # 设为准备中，让循环跑起来
            return

        # 1. 构造云端执行命令 (在容器内执行)
        # 激活环境 -> 设置 Token -> 运行 Agent
        config_json = json.dumps(params).replace("'", "'\\''")

        # spack 的 setup 脚本路径
        spack_setup = "/root/spack/share/spack/setup-env.sh"

        cmd = (
            f"source {spack_setup} && "
            f"spack env activate warpx-4090 && "
            f"export GONGJI_TOKEN={self.token} && "
            f"export GONGJI_BASE_URL={self.base_url} && "
            f"python3 /mnt/warpx/mag_sim/agent/node_executor.py "
            f"--hash {task_hash} --out_name {self.task_name} --config '{config_json}'"
        )

        # 2. 构造 API Payload (严格对应你提供的文档结构)
        payload = {
            "task_name": self.task_name,
            "task_type": "Deployment",
            "points": 1,
            "resources": [{
                "mark": os.getenv("GONGJI_RESOURCE_MARK"),
                "resource": {
                    "device_name": "4090",
                    "region": "chengde-p1",
                    "gpu_name": "4090",
                    "gpu_count": 1,
                    "gpu_memory": 24560,
                    "memory": 64512,
                    "cpu_cores": 16
                }
            }],
            "services": [{
                "service_name": f"sim-{task_hash[:8]}",
                "service_image": os.getenv("GONGJI_IMAGE"),
                "env": None,
                "share_storage_config": [{
                    "storage_id": int(os.getenv("GONGJI_STORAGE_ID", 2031)),
                    "target_dir": "/mnt/warpx"
                }],
                "start_script_v2": {
                    "command": "bash",
                    "args": ["-c", cmd]
                },
                "remote_ports": [{"service_port": 22}]  # 占位端口
            }]
        }

        # 3. 提交
        resp = requests.post(f"{self.base_url}/api/deployment/task/create",
                             headers=self._get_headers(), json=payload)

        if resp.status_code != 200:
            print(f"[DEBUG] 服务器返回状态码: {resp.status_code}")
            print(f"[DEBUG] 服务器返回内容: {resp.text}")

        if not resp.text.strip():
            raise Exception("服务器返回了空响应（Empty Body）")

        try:
            res = resp.json()
        except Exception as e:
            print(f"解析 JSON 失败。原始响应内容：\n{resp.text}")
            raise e

        if res.get("code") == "0000":
            self.task_id = res["data"]["task_id"]
            self._status = JobStatus.RUNNING
            print(f"[Gongji] 任务已提交到云端，ID: {self.task_id}")
        else:
            self._status = JobStatus.FAILED
            raise Exception(f"云端提交失败: {res.get('message')}")

    def get_status(self) -> JobStatus:
        if not self.task_id: return self._status

        try:
            resp = requests.get(f"{self.base_url}/api/deployment/task/details",
                                headers=self._get_headers(), params={"task_id": self.task_id})
            res_json = resp.json()
            data = res_json.get("data")

            # 调试打印，看看云端到底返回了什么
            # print(f"[Debug] Cloud Status: {data.get('status') if data else 'None'}")

            if data is None:
                # 如果刚提交不到 5 秒，即使找不到任务也不要判定为成功，可能是同步延迟
                return self._status

            cloud_status = data.get("status")

            # 更新 ID 链路
            if data.get("services"):
                self.service_id = data["services"][0]["service_id"]

            status_map = {
                "Pending": JobStatus.PENDING,
                "Running": JobStatus.RUNNING,
                "Paused": JobStatus.CANCELLED,
                "End": JobStatus.SUCCESS,
                "Other": JobStatus.FAILED
            }

            new_status = status_map.get(cloud_status, JobStatus.UNKNOWN)

            # 打印状态切换，方便观察
            if new_status != self._status:
                print(f"[Gongji] 任务 {self.task_id} 状态切换: {self._status} -> {new_status}")

            self._status = new_status
            return self._status

        except Exception as e:
            print(f"[Gongji] 轮询状态异常: {e}")
            return self._status

    def get_logs(self) -> list[str]:
        if not self.task_id:
            return []

        combined_output = []

        if not self.point_id:
            self._update_point_id()

        # --- 1. 获取系统事件 (如调度、拉镜像) ---
        # 只要不是终端状态就持续获取
        if self._status not in [JobStatus.SUCCESS, JobStatus.FAILED, JobStatus.CANCELLED]:
            events = self._fetch_incremental_events()
            combined_output.extend(events)

        # --- 2. 获取业务日志 ---
        # 必须有 service_id 和 point_id 才能拿日志
        if self.service_id and self.point_id:
            try:
                log_resp = requests.get(
                    f"{self.base_url}/api/deployment/task/point_log",
                    headers=self._get_headers(),
                    params={
                        "task_id": self.task_id,
                        "point_id": self.point_id,
                        "service_id": self.service_id
                    },
                    timeout=5
                )
                all_text = log_resp.json().get("data", "")
                if all_text:
                    lines = all_text.splitlines()
                    new_business_logs = lines[self.last_log_line_count:]
                    self.last_log_line_count = len(lines)
                    # 业务日志保持原样
                    combined_output.extend([l + "\n" for l in new_business_logs])
            except:
                pass  # 网络波动暂时忽略

        return combined_output

    def interrupt(self):
        if self.task_id:
            print(f"[Gongji] 正在强制停止云端任务 {self.task_id}...")
            requests.post(f"{self.base_url}/api/deployment/task/delete",
                          headers=self._get_headers(), json={"task_id": self.task_id})
            self._status = JobStatus.CANCELLED
