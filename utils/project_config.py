# project_config.py

import os
import sys
import json
from pathlib import Path
from dotenv import load_dotenv

# --- 1. 基础路径定义 ---
# 自动定位项目根目录
UTILS_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --- 2. 环境配置加载 ---
ENV_FILE_PATH = UTILS_ROOT / '.env.warpx'
if not load_dotenv(ENV_FILE_PATH):
    print(f"[Config] 警告: 未找到配置文件: {ENV_FILE_PATH}", file=sys.stderr)

# --- 3. 跨平台/环境常量 ---
# 环境配置
PROJECT_ROOT_WSL = os.getenv('PROJECT_ROOT_WSL')
CONDA_INIT_PATH = os.getenv('CONDA_INIT_PATH')
CONDA_ENV_NAME = os.getenv('CONDA_ENV_NAME')

SPACK_ROOT = os.getenv('SPACK_ROOT')
SPACK_ENV_NAME = os.getenv('SPACK_ENV_NAME')

MAIN_SCRIPT_PATH = os.path.join(PROJECT_ROOT_WSL, 'main.py')

# 文件名常量 (统一管理，防止拼写错误)
FILENAME_QUEUE = 'queue.jsonl'
FILENAME_HISTORY = 'history.jsonl'
FILENAME_TASKS_CSV = 'tasks.csv'
FILENAME_DEFAULT_PARAMS = 'default_params.json'
DIRNAME_LOGS = 'logs'

# GUI 显示用的列名
COLUMN_TASK_NAME = '任务名'

# 任务状态常量
STATUS_COMPLETED = "已完成"
STATUS_FAILED = "失败"
STATUS_PENDING = "待运行"
STATUS_RUNNING = "运行中"  # 预留


# --- 4. 路径工具函数 ---
def get_spack_activation_command(env_path: str = None) -> str:
    """获取激活 Spack 环境的 Bash 命令前缀"""
    if env_path is None:
        env_path = SPACK_ENV_NAME

    # 检查 spack 环境文件是否存在
    spack_setup = os.path.join(SPACK_ROOT, 'share', 'spack', 'setup-env.sh')

    if not os.path.exists(spack_setup):
        return "echo 'Error: Spack not found at {SPACK_ROOT}' && exit 1"

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


def validate_env():
    """验证必要的环境变量是否已加载"""
    required = ['PROJECT_ROOT_WSL', 'CONDA_INIT_PATH', 'CONDA_ENV_NAME']
    missing = [key for key in required if not os.getenv(key)]
    if missing:
        raise EnvironmentError(f"缺少必要的环境变量: {', '.join(missing)}。请检查 .env.warpx 文件。")
