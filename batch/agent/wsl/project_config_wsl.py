import os
import sys
from pathlib import Path

from dotenv import load_dotenv

current_dir = Path(__file__).resolve().parent

# --- 环境配置加载 ---
ENV_FILE_PATH = current_dir / '.wsl.service.env'
if not load_dotenv(ENV_FILE_PATH):
    print(f"[Config] 警告: 未找到配置文件: {ENV_FILE_PATH}", file=sys.stderr)

PROJECT_ROOT_WSL = os.getenv('PROJECT_ROOT_WSL')
CONDA_INIT_PATH = os.getenv('CONDA_INIT_PATH')
CONDA_ENV_NAME = os.getenv('CONDA_ENV_NAME')
SPACK_ROOT = os.getenv('SPACK_ROOT')
SPACK_ENV_NAME = os.getenv('SPACK_ENV_NAME')
MAIN_SCRIPT_PATH = os.path.join(PROJECT_ROOT_WSL, 'main.py')


def get_spack_activation_command(env_path: str = None) -> str:
    """获取激活 Spack 环境的 Bash 命令前缀"""
    if env_path is None:
        env_path = SPACK_ENV_NAME

    spack_setup = f"{SPACK_ROOT}/share/spack/setup-env.sh"

    return (
        f"source {spack_setup} && "
        f"spack env activate {env_path}"
    )


def get_wsl_path(win_path: str | Path) -> str:
    """
    将 Windows 路径转换为 WSL 内部路径。
    支持 C:/... 格式以及 //wsl.localhost/... 网络路径格式。
    """
    p = str(Path(win_path).resolve()).replace('\\', '/')

    # 处理通过网络邻居访问的 WSL 路径 (//wsl.localhost/Ubuntu/home/user/...)
    if p.startswith('//wsl.localhost/') or p.startswith('//wsl$/'):
        parts = p.split('/')
        # parts[0,1]是空, parts[2]是host, parts[3]是distro
        # 我们需要保留从 parts[4] 开始的部分，并加上根斜杠
        if len(parts) > 4:
            return '/' + '/'.join(parts[4:])

    # 处理 Windows 物理驱动器路径 (C:/Users/...) -> /mnt/c/Users/...
    if ':' in p:
        drive, path = p.split(':', 1)
        return f"/mnt/{drive.lower()}{path}"

    return p


def get_conda_activation_command() -> str:
    """获取激活 Conda 环境的 Bash 命令前缀"""
    if not all([CONDA_INIT_PATH, CONDA_ENV_NAME]):
        return "echo 'Error: Conda config missing' && exit 1"

    return (
        f"source {CONDA_INIT_PATH} && "
        f"conda activate {CONDA_ENV_NAME}"
    )
