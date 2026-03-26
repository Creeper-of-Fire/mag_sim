# batch/gongji_manager.py
import requests
import time
import json
import os
from .manager_api import BaseComputeManager, JobStatus


class GongjiComputeManager(BaseComputeManager):
    def __init__(self, token: str, base_url: str):
        self.token = token
        self.base_url = base_url
        self.task_id = None
        self.task_name = None
        self.service_id = None
        self.point_id = None

        self.last_log_line_count = 0
        self._status = JobStatus.PENDING

    def _get_headers(self):
        return {
            "token": self.token,
            "timestamp": str(int(time.time() * 1000)),
            "version": "1.0.0",
            "Content-Type": "application/json"
        }

    def submit(self, task_hash: str, params: dict, output_dir_name: str):
        """
        提交任务到共绩云平台
        """
        self.task_name = output_dir_name

        # 1. 构造云端执行命令 (在容器内执行)
        # 激活环境 -> 设置 Token -> 运行 Agent
        config_json = json.dumps(params).replace("'", "'\\''")
        cmd = (
            f"source /etc/profile && "
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
                    "gpu_name": "4090",
                    "gpu_count": 1
                }
            }],
            "services": [{
                "service_name": f"sim-{task_hash[:6]}",
                "service_image": os.getenv("GONGJI_IMAGE"),
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
        res = resp.json()

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
            data = resp.json()

            # 如果接口返回任务不存在（因为 Agent 自杀了）
            if data.get("code") != "0000" or data.get("data") is None:
                if self._status == JobStatus.RUNNING:
                    # 之前在跑现在没了，视为成功结束
                    return JobStatus.SUCCESS
                return JobStatus.UNKNOWN

            cloud_status = data["data"].get("status")

            # 自动维护 ID 链路
            if data["data"].get("services"):
                self.service_id = data["data"]["services"][0]["service_id"]

            # 映射
            status_map = {
                "Pending": JobStatus.PENDING,
                "Running": JobStatus.RUNNING,
                "Paused": JobStatus.CANCELLED,
                "End": JobStatus.SUCCESS
            }
            self._status = status_map.get(cloud_status, JobStatus.UNKNOWN)
            return self._status

        except Exception:
            return self._status

    def get_logs(self) -> list[str]:
        if not self.task_id or self._status == JobStatus.SUCCESS:
            return []

        # 1. 确保有 point_id (通过 point_list 接口)
        if not self.point_id:
            try:
                resp = requests.get(f"{self.base_url}/api/deployment/task/point_list",
                                    headers=self._get_headers(), params={"task_id": self.task_id})
                points = resp.json().get("data", {}).get("results", [])
                if points:
                    self.point_id = points[0]["point_id"]
                else:
                    return []
            except:
                return []

        # 2. 拉取全量日志并切片
        try:
            log_resp = requests.get(f"{self.base_url}/api/deployment/task/point_log",
                                    headers=self._get_headers(),
                                    params={
                                        "task_id": self.task_id,
                                        "point_id": self.point_id,
                                        "service_id": self.service_id
                                    })
            all_text = log_resp.json().get("data", "")
            if not all_text: return []

            lines = all_text.splitlines()
            new_content = lines[self.last_log_line_count:]
            self.last_log_line_count = len(lines)

            return [l + "\n" for l in new_content]
        except:
            return []

    def interrupt(self):
        if self.task_id:
            print(f"[Gongji] 正在强制停止云端任务 {self.task_id}...")
            requests.post(f"{self.base_url}/api/deployment/task/delete",
                          headers=self._get_headers(), json={"task_id": self.task_id})
            self._status = JobStatus.CANCELLED