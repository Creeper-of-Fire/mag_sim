# batch/manager_api.py
import inspect
import json
from abc import ABC, abstractmethod
from enum import Enum, auto
from pathlib import Path
from types import ModuleType


class JobStatus(Enum):
    PENDING = auto()  # 提交中/准备中
    RUNNING = auto()  # 正在运行
    SUCCESS = auto()  # 成功完成
    FAILED = auto()  # 执行失败（模拟器崩溃或环境错误）
    CANCELLED = auto()  # 被手动中断
    UNKNOWN = auto()  # 失去连接或状态异常


# 动态获取本地项目根目录 (假设此文件在 PROJECT_ROOT/batch/ 下)
LOCAL_PROJECT_ROOT = Path(__file__).resolve().parent.parent


class BaseComputeManager(ABC):
    """
    计算任务管理器基类。
    一个 Manager 实例负责 一次 任务的提交、监控、日志获取。
    """

    @abstractmethod
    def submit(self, task_hash: str, params: dict, output_dir_name: str, rel_job_path: str):
        """异步提交任务到计算环境。"""
        pass

    @abstractmethod
    def get_status(self) -> JobStatus:
        """获取当前任务的抽象状态。"""
        pass

    @abstractmethod
    def get_logs(self) -> list[str]:
        """获取自上次调用以来的增量日志。"""
        pass

    @abstractmethod
    def interrupt(self):
        """强制中断当前任务。"""
        pass

    @staticmethod
    def build_node_command(
            executor_module: ModuleType,
            remote_root: str,
            task_hash: str,
            output_dir_name: str,
            rel_job_path: str,
            params: dict,
            python_exe: str = "python3"
    ) -> str:
        """
        基于模块反射，动态生成发往远端的标准执行命令。

        :param executor_module: import 进来的目标 agent 模块 (IDE 可监察)
        :param remote_root: 远端机器上的项目根目录绝对路径 (如 /mnt/warpx/mag_sim)
        :param task_hash: 任务哈希
        :param output_dir_name: 输出目录名
        :param rel_job_path: 任务在项目中的相对路径
        :param params: JSON 配置字典
        :param python_exe: 远端使用的 python 解释器命令
        :return: 拼装好的 bash 命令片段
        """
        # 1. 显式反射：获取 executor 模块在本地的绝对路径
        local_file = Path(inspect.getfile(executor_module)).resolve()

        # 2. 计算相对路径 (如 batch/agent/yingbo/node_executor_yingbo.py)
        try:
            rel_path = local_file.relative_to(LOCAL_PROJECT_ROOT)
        except ValueError:
            raise ValueError(f"模块 {executor_module.__name__} 不在项目根目录 {LOCAL_PROJECT_ROOT} 下！")

        # 3. 映射到远端绝对路径 (注意：强制使用 .as_posix() 保证生成 Linux 的正斜杠)
        remote_root = remote_root.rstrip('/').replace('\\', '/')
        abs_agent_py = f"{remote_root}/{rel_path.as_posix()}"

        # main.py 和 work_dir 永远遵循固定的项目结构约定
        abs_main_py = f"{remote_root}/main.py"
        abs_work_dir = f"{remote_root}/{rel_job_path}/sim_results/{output_dir_name}"

        # 4. JSON 参数转义 (避免 Bash 引号嵌套炸裂)
        config_json = json.dumps(params).replace("'", "'\\''")

        # 5. 组装标准参数部分
        args_str = (
            f"--hash {task_hash} "
            f"--out_name {output_dir_name} "
            f"--work_dir '{abs_work_dir}' "
            f"--main_py '{abs_main_py}' "
            f"--config '{config_json}'"
        )

        # 6. 返回最终命令
        return f"{python_exe} {abs_agent_py} {args_str}"
