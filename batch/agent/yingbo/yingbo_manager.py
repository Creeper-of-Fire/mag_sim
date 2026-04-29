import argparse
import os
import sys
import time
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, List, Union

from dotenv import load_dotenv
from kubernetes import client, config, watch
from pydantic import BaseModel, Field, field_validator

import batch.agent.yingbo.node_executor_yingbo as yingbo_agent
from batch.manager_api import BaseComputeManager, JobStatus

current_dir = Path(__file__).resolve().parent

ENV_FILE_PATH = current_dir / '.yingbo.service.env'
if not load_dotenv(ENV_FILE_PATH):
    print(f"[Config] 警告: 未找到配置文件: {ENV_FILE_PATH}", file=sys.stderr)


class EnvironmentType(str, Enum):
    """环境类型枚举"""
    SPACK = "spack"
    CONDA = "conda"
    NONE = "none"


class EnvironmentConfig(BaseModel):
    """环境配置"""
    type: EnvironmentType = Field(..., description="环境类型")
    env_name: Optional[str] = Field(None, description="环境名称")
    image_tag: str = Field(..., description="Docker 镜像标签")

    def get_command_builder(self) -> Optional[Union["SpackCommandBuilder", "CondaCommandBuilder"]]:
        """根据环境类型获取对应的命令构建器"""
        if self.type == EnvironmentType.SPACK and self.env_name:
            return SpackCommandBuilder(spack_env=self.env_name)
        elif self.type == EnvironmentType.CONDA and self.env_name:
            return CondaCommandBuilder(conda_env=self.env_name)
        return None

    def get_image(self, prefix: str) -> str:
        """获取完整镜像地址"""
        return prefix + self.image_tag

class CPUType(str, Enum):
    """CPU 类型枚举"""
    C1M2 = "C1M2"      # 1核 2GB
    C2M4 = "C2M4"      # 2核 4GB
    C4M8 = "C4M8"      # 4核 8GB
    C8M16 = "C8M16"    # 8核 16GB
    C16M32 = "C16M32"  # 16核 32GB
    C32M64 = "C32M64"  # 32核 64GB
    C64M128 = "C64M128"  # 64核 128GB
    C1M4 = "C1M4"      # 1核 4GB
    C2M8 = "C2M8"      # 2核 8GB
    C4M16 = "C4M16"    # 4核 16GB
    C8M32 = "C8M32"    # 8核 32GB
    C16M64 = "C16M64"  # 16核 64GB
    C32M128 = "C32M128"  # 32核 128GB
    C64M256 = "C64M256"  # 64核 256GB
    C1M8 = "C1M8"      # 1核 8GB
    C2M16 = "C2M16"    # 2核 16GB
    C4M32 = "C4M32"    # 4核 32GB
    C8M64 = "C8M64"    # 8核 64GB
    C16M128 = "C16M128"  # 16核 128GB
    C32M256 = "C32M256"  # 32核 256GB

class GPUType(str, Enum):
    """GPU 类型枚举"""
    A800 = "A800"
    RTX_4090 = "RTX_4090"
    RTX_4090D = "RTX_4090D"
    A40 = "A40"
    A16 = "A16"
    CPU = "CPU"


class YingboGPUSpec(BaseModel):
    """英博云 GPU 规格定义"""
    label: str = Field(..., description="节点选择器标签")
    cpu: Optional[str] = Field(None, description="CPU 核心数限制")
    mem: Optional[str] = Field(None, description="内存限制")
    flavor: Optional[str] = Field(None, description="实例规格")
    gpu_driver: Optional[List[str]] = Field(None, description="支持的 GPU 驱动版本列表")
    environment: Optional[EnvironmentConfig] = Field(None, description="环境配置")

    @property
    def is_gpu(self) -> bool:
        """判断是否为 GPU 规格"""
        return self.gpu_driver is not None

    @field_validator('cpu', mode='before')
    @classmethod
    def validate_cpu(cls, v):
        """确保 CPU 值为字符串格式"""
        if v is not None:
            return str(v)
        return v

    def get_limits(self) -> Dict[str, str]:
        """
        获取 K8s 资源限制字典

        Returns:
            K8s resources.limits 字典
        """
        limits = {
            "cpu": self.cpu,
            "memory": self.mem
        }
        if self.is_gpu:
            limits["nvidia.com/gpu"] = "1"
        return limits

    def get_environment_variables(self) -> List[client.V1EnvVar]:
        """
        获取该规格所需的额外环境变量

        GPU 规格默认不需要额外环境变量，CPU 规格会覆写此方法。

        Returns:
            V1EnvVar 列表
        """
        return []


class CPUSpec(YingboGPUSpec):
    """英博云 CPU 规格定义（继承 GPU 规格，复用字段和逻辑）"""

    @classmethod
    def create(cls, cpu: str, mem: str, flavor: str) -> "CPUSpec":
        """创建 CPU 规格实例"""
        return cls(
            label="amd-epyc-milan",  # 固定标签
            cpu=cpu,
            mem=mem,
            flavor=flavor,
            gpu_driver=None,  # CPU 规格无 GPU 驱动
            environment=EnvironmentConfig(
                type=EnvironmentType.CONDA,
                env_name="warpx-cpu",
                image_tag="warpx/warpxcpu:warpxcpu"
            ), # 固定环境
        )

    @property
    def cpu_cores(self) -> int:
        """
        解析 CPU 核心数（向下取整）

        例如: "1.75" -> 1, "10" -> 10, "3" -> 3
        """
        try:
            return int(float(self.cpu))
        except (ValueError, TypeError):
            return 1  # 默认至少 1 核

    def get_environment_variables(self) -> List[client.V1EnvVar]:
        """
        CPU 规格需要设置 OpenMP 线程数，避免 oversubscription

        Returns:
            包含 OMP_NUM_THREADS 等环境变量的列表
        """
        cores = self.cpu_cores

        return [
            client.V1EnvVar(
                name="OMP_NUM_THREADS",
                value=str(cores)
            ),
            client.V1EnvVar(
                name="OMP_PLACES",
                value="cores"
            ),
            client.V1EnvVar(
                name="OMP_PROC_BIND",
                value="close"
            ),
            # 如果有 MPI，也建议设置
            client.V1EnvVar(
                name="OMPI_MCA_rmaps_base_oversubscribe",
                value="0"  # 禁止 MPI 超订
            ),
        ]


class CPUConfig(BaseModel):
    """英博云 CPU 配置管理"""
    specs: Dict[CPUType, CPUSpec] = Field(default_factory=dict)

    @classmethod
    def create_default(cls) -> "CPUConfig":
        """创建默认 CPU 配置（基于你提供的表格数据）"""
        return cls(specs={
            CPUType.C1M2: CPUSpec.create(cpu="1", mem="2Gi", flavor="bob-eci.cpu.c1m2.0.5large"),
            CPUType.C2M4: CPUSpec.create(cpu="2", mem="4Gi", flavor="bob-eci.cpu.c2m4.large"),
            CPUType.C4M8: CPUSpec.create(cpu="4", mem="8Gi", flavor="bob-eci.cpu.c4m8.2large"),
            CPUType.C8M16: CPUSpec.create(cpu="8", mem="16Gi", flavor="bob-eci.cpu.c8m16.4large"),
            CPUType.C16M32: CPUSpec.create(cpu="16", mem="32Gi", flavor="bob-eci.cpu.c16m32.8large"),
            CPUType.C32M64: CPUSpec.create(cpu="32", mem="64Gi", flavor="bob-eci.cpu.c32m64.16large"),
            CPUType.C64M128: CPUSpec.create(cpu="64", mem="128Gi", flavor="bob-eci.cpu.c64m128.32large"),
            CPUType.C1M4: CPUSpec.create(cpu="1", mem="4Gi", flavor="bob-eci.cpu.c1m4.0.5large"),
            CPUType.C2M8: CPUSpec.create(cpu="2", mem="8Gi", flavor="bob-eci.cpu.c2m8.large"),
            CPUType.C4M16: CPUSpec.create(cpu="4", mem="16Gi", flavor="bob-eci.cpu.c4m16.2large"),
            CPUType.C8M32: CPUSpec.create(cpu="8", mem="32Gi", flavor="bob-eci.cpu.c8m32.4large"),
            CPUType.C16M64: CPUSpec.create(cpu="16", mem="64Gi", flavor="bob-eci.cpu.c16m64.8large"),
            CPUType.C32M128: CPUSpec.create(cpu="32", mem="128Gi", flavor="bob-eci.cpu.c32m128.16large"),
            CPUType.C64M256: CPUSpec.create(cpu="64", mem="256Gi", flavor="bob-eci.cpu.c64m256.32large"),
            CPUType.C1M8: CPUSpec.create(cpu="1", mem="8Gi", flavor="bob-eci.cpu.c1m8.0.5large"),
            CPUType.C2M16: CPUSpec.create(cpu="2", mem="16Gi", flavor="bob-eci.cpu.c2m16.large"),
            CPUType.C4M32: CPUSpec.create(cpu="4", mem="32Gi", flavor="bob-eci.cpu.c4m32.2large"),
            CPUType.C8M64: CPUSpec.create(cpu="8", mem="64Gi", flavor="bob-eci.cpu.c8m64.4large"),
            CPUType.C16M128: CPUSpec.create(cpu="16", mem="128Gi", flavor="bob-eci.cpu.c16m128.8large"),
            CPUType.C32M256: CPUSpec.create(cpu="32", mem="256Gi", flavor="bob-eci.cpu.c32m256.16large"),
        })

    def get_spec(self, cpu_type: CPUType) -> CPUSpec:
        """获取指定类型的 CPU 规格"""
        if cpu_type not in self.specs:
            raise ValueError(f"未知的 CPU 类型: {cpu_type}")
        return self.specs[cpu_type]

class YingboGPUConfig(BaseModel):
    """英博云 GPU 配置管理"""
    specs: Dict[GPUType, YingboGPUSpec] = Field(default_factory=dict)

    @classmethod
    def create_default(cls) -> "YingboGPUConfig":
        """创建默认配置"""
        return cls(specs={
            GPUType.A800: YingboGPUSpec(
                label="A800_NVLINK_80GB",
                cpu="10",
                mem="100Gi",
                flavor="bob-eci.a800.5large",
                gpu_driver=["580.65.06", "535.161.08"],
                environment=EnvironmentConfig(
                    type=EnvironmentType.SPACK,
                    env_name="warpx-a800",
                    image_tag="warpx-a800:warpx-a800-pure"
                ),
            ),
            GPUType.RTX_4090: YingboGPUSpec(
                label="RTX_4090",
                cpu="10",
                mem="100Gi",
                flavor="bob-eci.4090.5large",
                gpu_driver=["580.76.05", "535.161.08"],
                environment=None,
            ),
            GPUType.RTX_4090D: YingboGPUSpec(
                label="RTX_4090D",
                cpu="10",
                mem="100Gi",
                flavor="bob-eci.4090d.5large",
                gpu_driver=["575.64.05"],
                environment=None,
            ),
            GPUType.A40: YingboGPUSpec(
                label="A40",
                cpu="3",
                mem="50Gi",
                flavor="bob-eci.a40.c3m50",
                gpu_driver=["535.161.08"],
                environment=EnvironmentConfig(
                    type=EnvironmentType.SPACK,
                    env_name="warpx-a40",
                    image_tag="warpx-a40:warpx-a40"
                ),
            ),
            GPUType.A16: YingboGPUSpec(
                label="A16",
                cpu="1.75",
                mem="7Gi",
                flavor="bob-eci.a16.c1.75m7",
                gpu_driver=["535.161.08"],
                environment=EnvironmentConfig(
                    type=EnvironmentType.SPACK,
                    env_name="warpx-a16",
                    image_tag="warpx-a16:warpx-a16-pure"
                ),
            )
        })

    def get_spec(self, gpu_type: GPUType) -> YingboGPUSpec:
        """获取指定类型的 GPU 规格"""
        if gpu_type not in self.specs:
            raise ValueError(f"未知的 GPU 类型: {gpu_type}")
        return self.specs[gpu_type]


class CondaCommandBuilder:
    """Conda 命令构建器"""

    CONDA_SETUP_PATH = "/root/miniconda3/etc/profile.d/conda.sh"

    def __init__(self, conda_env: str):
        """
        初始化 Conda 命令构建器

        Args:
            conda_env: Conda 环境名称
        """
        self.conda_env = conda_env

    def build_command(self, agent_cmd: str) -> str:
        """
        构建包含 Conda 环境的完整命令

        Args:
            agent_cmd: Agent 执行命令

        Returns:
            完整的 bash 命令字符串
        """
        if not self.conda_env:
            return agent_cmd

        return (
            f"source {self.CONDA_SETUP_PATH} && "
            f"conda activate {self.conda_env} && "
            f"{agent_cmd}"
        )

    @classmethod
    def from_spec(cls, spec: YingboGPUSpec) -> Optional["CondaCommandBuilder"]:
        """从 GPU 规格创建 Conda 命令构建器"""
        if spec.conda_env:
            return cls(conda_env=spec.conda_env)
        return None


class SpackCommandBuilder:
    """Spack 命令构建器"""

    SPACK_SETUP_PATH = "/root/spack/share/spack/setup-env.sh"

    def __init__(self, spack_env: str):
        """
        初始化 Spack 命令构建器

        Args:
            spack_env: Spack 环境名称
        """
        self.spack_env = spack_env

    def build_command(self, agent_cmd: str) -> str:
        """
        构建包含 Spack 环境的完整命令

        Args:
            agent_cmd: Agent 执行命令

        Returns:
            完整的 bash 命令字符串
        """
        if not self.spack_env:
            return agent_cmd

        return (
            f"source {self.SPACK_SETUP_PATH} && "
            f"spack env activate {self.spack_env} && "
            f"{agent_cmd}"
        )

    @classmethod
    def from_spec(cls, spec: YingboGPUSpec) -> Optional["SpackCommandBuilder"]:
        """从 GPU 规格创建 Spack 命令构建器"""
        if spec.spack_env:
            return cls(spack_env=spec.spack_env)
        return None


class YingboComputeManager(BaseComputeManager):
    def __init__(self,  gpu_type: GPUType | None = None, cpu_type: CPUType | None = None):
        # 加载 kubeconfig
        kubeconfig_path = os.getenv("KUBECONFIG_PATH")
        print(f"[DEBUG] 正在尝试加载 Kubeconfig: {kubeconfig_path}")  # 添加这一行
        config.load_kube_config(config_file=kubeconfig_path)

        self.batch_v1 = client.BatchV1Api()
        self.core_v1 = client.CoreV1Api()
        self.namespace = "default"

        # 区分 GPU 和 CPU 模式
        self.is_gpu_mode = gpu_type is not None

        if self.is_gpu_mode:
            self.gpu_config = YingboGPUConfig.create_default()
            self.current_spec = self.gpu_config.get_spec(gpu_type)
        else:
            if cpu_type is None:
                raise ValueError("必须指定 gpu_type 或 cpu_type")
            self.cpu_config = CPUConfig.create_default()
            self.current_spec = self.cpu_config.get_spec(cpu_type)

        self.job_name = None
        self._status = JobStatus.PENDING
        self.last_log_line_count = 0

    @classmethod
    def from_args(cls, args: list[str]) -> "YingboComputeManager":
        """
        从命令行参数列表构造 Manager。
        支持的参数:
          --gpu <GPUType>   如 --gpu A800 / --gpu RTX_4090
          --cpu <CPUType>   如 --cpu C8M16
        """
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--gpu", type=str, default=None)
        parser.add_argument("--cpu", type=str, default=None)
        parsed, _ = parser.parse_known_args(args)

        if parsed.cpu:
            cpu_type = CPUType(parsed.cpu)
            return cls(gpu_type=None, cpu_type=cpu_type)

        if parsed.gpu:
            gpu_type = GPUType(parsed.gpu)
            return cls(gpu_type=gpu_type, cpu_type=None)

        return cls(gpu_type=GPUType.A800, cpu_type=None)

    def submit(self, task_hash: str, params: dict, output_dir_name: str, rel_job_path: str):
        # 获取镜像地址
        using_image = self.current_spec.environment.get_image(os.getenv("DOCKER_IMAGE_PREFIX"))

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

        # ========== 性能采样包裹 ==========
        # 定义一个绝对路径日志目录
        perf_log_dir = f"{remote_root}/sim_jobs/{rel_job_path}/logs/perf"
        # 确保目录存在（在 Pod 启动时）
        mkdir_cmd = f"mkdir -p {perf_log_dir}"

        # 方案 1：nvidia-smi 前后采样（轻量级）
        nvidia_cmd = (
            f"nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv "
            f">> {perf_log_dir}/gpu_metrics_{task_hash[:8]}.log && "
            f"{agent_cmd} && "
            f"nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv "
            f">> {perf_log_dir}/gpu_metrics_{task_hash[:8]}.log"
        )

        # 方案 2：nsys 深度分析（可选，开销大）
        # 取消注释下面这段来使用 nsys
        # nsys_cmd = (
        #     f"nsys profile --stats=true "
        #     f"--output={perf_log_dir}/nsys_profile_{task_hash[:8]} "
        #     f"--force-overwrite true "
        #     f"{agent_cmd}"
        # )

        # 选择使用哪个包裹命令（默认用 nvidia-smi）
        wrapped_cmd = f"{mkdir_cmd} && {nvidia_cmd}"

        # 把原来的 agent_cmd 替换成 wrapped_cmd
        agent_cmd = wrapped_cmd
        # ========== 性能采样包裹结束 ==========

        # 根据环境类型构建命令
        cmd_builder = self.current_spec.environment.get_command_builder()
        if cmd_builder:
            cmd = cmd_builder.build_command(agent_cmd)
        else:
            cmd = agent_cmd

        env_vars = [
            # 让 Python 日志不进入缓冲区，直接实时输出到 K8S 日志流
            client.V1EnvVar(name="PYTHONUNBUFFERED", value="1"),
        ]

        # 添加规格特定的环境变量
        env_vars.extend(self.current_spec.get_environment_variables())

        # 2. 定义 Job 对象
        container = client.V1Container(
            name="warpx-worker",
            image=using_image,
            command=["bash", "-c", cmd],
            env=env_vars,
            resources=client.V1ResourceRequirements(
                limits=self.current_spec.get_limits(),
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
                        node_selector={"cloud.ebtech.com/gpu": self.current_spec.label},
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
