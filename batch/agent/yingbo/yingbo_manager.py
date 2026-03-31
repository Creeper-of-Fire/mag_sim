import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from kubernetes import client, config, watch

import node_executor_yingbo as yingbo_agent
from batch.manager_api import BaseComputeManager, JobStatus

current_dir = Path(__file__).resolve().parent

ENV_FILE_PATH = current_dir / '.yingbo.service.env'
if not load_dotenv(ENV_FILE_PATH):
    print(f"[Config] 警告: 未找到配置文件: {ENV_FILE_PATH}", file=sys.stderr)


class YingboComputeManager(BaseComputeManager):
    def __init__(self):
        # 加载 kubeconfig
        kubeconfig_path = os.getenv("KUBECONFIG_PATH")
        print(f"[DEBUG] 正在尝试加载 Kubeconfig: {kubeconfig_path}")  # 添加这一行
        config.load_kube_config(config_file=kubeconfig_path)

        self.batch_v1 = client.BatchV1Api()
        self.core_v1 = client.CoreV1Api()
        self.namespace = "default"

        self.job_name = None
        self._status = JobStatus.PENDING
        self.last_log_line_count = 0

    def submit(self, task_hash: str, params: dict, output_dir_name: str, rel_job_path: str):
        # 刚才测试成功的 A800 规格
        EB_A800_SPEC = {
            "label": "A800_NVLINK_80GB",
            "cpu": "10",
            "mem": "100Gi",
            "flavor": "bob-eci.a800.5large"
        }

        self.job_name = f"sim-{task_hash[:8]}-{int(time.time())}"

        # 1. 构造容器启动命令
        remote_root = "/mnt/warpx/mag_sim"  # 挂载点约定
        agent_cmd = self.build_node_command(
            executor_module=yingbo_agent,
            remote_root=remote_root,
            task_hash=task_hash,
            output_dir_name=output_dir_name,
            rel_job_path=rel_job_path,
            params=params,
            python_exe="python3"
        )

        spack_setup = "/root/spack/share/spack/setup-env.sh"
        cmd = (
            f"source {spack_setup} && "
            f"spack env activate warpx-a800 && "
            f"{agent_cmd}"
        )

        env_vars = [
            # 让 Python 日志不进入缓冲区，直接实时输出到 K8S 日志流
            client.V1EnvVar(name="PYTHONUNBUFFERED", value="1"),
        ]

        # 2. 定义 Job 对象
        container = client.V1Container(
            name="warpx-worker",
            image=os.getenv("DOCKER_IMAGE"),
            command=["bash", "-c", cmd],
            env=env_vars,
            resources=client.V1ResourceRequirements(
                limits={
                    "nvidia.com/gpu": "1",
                    "cpu": EB_A800_SPEC["cpu"],
                    "memory": EB_A800_SPEC["mem"]
                }
            ),
            volume_mounts=[
                client.V1VolumeMount(name="warpx-vol", mount_path="/mnt/warpx")
            ]
        )

        job = client.V1Job(
            metadata=client.V1ObjectMeta(name=self.job_name),
            spec=client.V1JobSpec(
                backoff_limit=0,  # 崩了别重试，省钱
                ttl_seconds_after_finished=3600,  # 1小时后自动毁尸灭迹
                template=client.V1PodTemplateSpec(
                    spec=client.V1PodSpec(
                        image_pull_secrets=[client.V1LocalObjectReference(name="eb-registry-secret")],
                        node_selector={"cloud.ebtech.com/gpu": EB_A800_SPEC["label"]},
                        containers=[container],
                        volumes=[
                            client.V1Volume(
                                name="warpx-vol",
                                persistent_volume_claim=client.V1PersistentVolumeClaimVolumeSource(claim_name="warpx")
                            )
                        ],
                        restart_policy="Never"
                    )
                )
            )
        )

        # 3. 提交到集群
        try:
            self.batch_v1.create_namespaced_job(namespace=self.namespace, body=job)
            print(f"[Yingbo] Job {self.job_name} 已提交")
            self._status = JobStatus.RUNNING
        except Exception as e:
            self._status = JobStatus.FAILED
            raise Exception(f"K8S 提交失败: {e}")

    def get_status(self) -> JobStatus:
        try:
            job = self.batch_v1.read_namespaced_job_status(name=self.job_name, namespace=self.namespace)
            status = job.status
            if status.active:
                return JobStatus.RUNNING
            if status.succeeded:
                return JobStatus.SUCCESS
            if status.failed:
                return JobStatus.FAILED
            return JobStatus.PENDING
        except:
            return JobStatus.UNKNOWN

    def get_logs(self) -> list[str]:
        # K8S 需要先根据 Job 找到 Pod
        pods = self.core_v1.list_namespaced_pod(
            namespace=self.namespace,
            label_selector=f"job-name={self.job_name}"
        )
        if not pods.items:
            return []

        pod_name = pods.items[0].metadata.name
        try:
            # 获取全部日志并切片（K8S API 限制，通常直接拿全量进行本地 offset 过滤）
            all_logs = self.core_v1.read_namespaced_pod_log(name=pod_name, namespace=self.namespace)
            lines = all_logs.splitlines()
            new_lines = lines[self.last_log_line_count:]
            self.last_log_line_count = len(lines)
            return [l + "\n" for l in new_lines]
        except:
            return []

    def interrupt(self):
        print(f"[Yingbo] 正在删除 Job {self.job_name}...")
        self.batch_v1.delete_namespaced_job(
            name=self.job_name,
            namespace=self.namespace,
            propagation_policy='Background'
        )
