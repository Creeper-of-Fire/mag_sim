import shutil
from pathlib import Path

import dill
from mpi4py import MPI

from simulation.utils import master_only, enable_mpi_print

comm = MPI.COMM_WORLD

enable_mpi_print()

class IOManager:
    """
    负责模拟的输入输出管理、目录维护和日志记录。
    隐藏了 MPI rank 0 的检查逻辑。
    """

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.diags_dir = self.output_dir / "diags"

    @master_only
    def prepare_directories(self, overwrite: bool = True):
        """
        准备输出目录。
        """
        print(f"正在准备输出目录: {self.output_dir}")

        if self.output_dir.is_dir():
            if overwrite:
                print(f"警告: 输出目录 {self.output_dir} 已存在，正在删除...")
                shutil.rmtree(self.output_dir)
            else:
                print(f"警告: 输出目录 {self.output_dir} 已存在，将追加或覆盖文件。")

        # parents=True 允许创建多级目录
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.diags_dir.mkdir(parents=True, exist_ok=True)

    @master_only
    def save_simulation_parameters(self, params_dict: dict, filename: str = "sim_parameters.dpkl"):
        """
        保存模拟参数到磁盘。
        """
        file_path = self.output_dir / filename
        with open(file_path, "wb") as f:
            dill.dump(params_dict, f)
        print(f"模拟参数已保存至: {file_path}")

    @master_only
    def clean_diagnostics(self):
        """清理诊断目录（如果在初始化期间需要重置）"""
        if self.diags_dir.exists():
            shutil.rmtree(self.diags_dir)
        self.diags_dir.mkdir(parents=True, exist_ok=True)