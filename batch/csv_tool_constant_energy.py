# Plasma_Simulation/batch/csv_tool.py

import math
import sys
from pathlib import Path

# 引入物理常数用于计算
from scipy.constants import m_e, c, e, mu_0

# 获取当前脚本所在目录的父目录
root_dir = Path(__file__).resolve().parent.parent
# 强制添加到系统路径中
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from batch import csv_tool
from utils.project_config import PROJECT_ROOT

try:
    from simulation.config import SimulationParameters
except ImportError as e:
    print(f"错误: 无法导入 'SimulationParameters'。请确保 '{PROJECT_ROOT / 'simulation' / 'config.py'}' 文件存在且无误。", file=sys.stderr)
    print(f"详细错误: {e}", file=sys.stderr)
    sys.exit(1)


# --- 核心物理计算逻辑 ---

def calculate_partitioned_energy(n_plasma, T_ref_eV, target_sigma):
    """
    计算能量均分模式下的实际温度和磁场。
    保持总能量密度 = 2 * n * (mc^2 + 3*T_ref) 不变。
    """
    # 物理常数
    mc2_J = m_e * c ** 2
    kb_T_ref_J = T_ref_eV * e

    # 1. 计算 T_new (Joules)
    # 公式: 3 * T_new = [ (mc^2 + 3*T_ref) / (1 + sigma) ] - mc^2
    term1 = (mc2_J + 3.0 * kb_T_ref_J) / (1.0 + target_sigma)
    kb_T_new_J = (term1 - mc2_J) / 3.0

    # 检查物理可行性
    if kb_T_new_J < 0:
        max_sigma = (3.0 * kb_T_ref_J) / mc2_J
        raise ValueError(
            f"无法实现能量均分: 目标 Sigma ({target_sigma}) 过高。\n"
            f"      对于 T_ref={T_ref_eV} eV，最大可达到的 Sigma 约为 {max_sigma:.4f}。\n"
            f"      (由于不能消耗静止质量产生磁场，热能已被抽干)"
        )

    # 转换回 eV
    T_new_eV = kb_T_new_J / e

    # 2. 计算 B_new (Tesla)
    # 粒子能量密度 (新)
    U_p_new = 2 * n_plasma * (mc2_J + 3.0 * kb_T_new_J)
    # 磁能量密度
    U_B_new = target_sigma * U_p_new
    # B = sqrt(2 * mu_0 * U_B)
    B_new = math.sqrt(2 * mu_0 * U_B_new)

    return T_new_eV, B_new


# --- 钩子函数 ---

def constant_energy_processor(task_params: dict, line_num: int):
    """
    这是我们将注入到 csv_tool 中的钩子。
    它接收原始转换后的参数，修改它们，然后返回。
    """

    # 检查必要参数
    req_keys = ['n_plasma', 'T_plasma_eV', 'target_sigma']
    for k in req_keys:
        if k not in task_params:
            raise ValueError(f"Constant Energy 模式缺少必要参数: '{k}'")

    n_p = task_params['n_plasma']
    T_ref = task_params['T_plasma_eV']
    sigma = task_params['target_sigma']

    # 计算
    T_real, B_real = calculate_partitioned_energy(n_p, T_ref, sigma)

    print(f"   [行 {line_num}] 能量均分 (Sigma={sigma}): T_ref={T_ref:.1f}eV -> T_real={T_real:.1f}eV, B={B_real:.4f}T")

    # 修改参数 (直接覆盖)
    task_params['T_plasma_eV'] = T_real
    task_params['B0'] = B_real

    # 标记模式 (可选，用于调试)
    task_params['_generated_mode'] = "constant_energy_partition"

    return task_params


# --- 主程序 ---

def handle_convert_override(args):
    """
    包装原始的 handle_convert，传入我们的处理器
    """
    # 调用 csv_tool 中的核心逻辑，但带上我们的 processor
    csv_tool.handle_convert(args, param_processor=constant_energy_processor)


if __name__ == "__main__":
    # 1. 获取基础工具的 Parser (复用所有参数定义)
    parser = csv_tool.setup_parser()

    # 2. 修改描述
    parser.description = "Plasma Simulation CSV Tool (Constant Energy Mode)"
    parser.epilog = "注意：此模式默认开启 'Constant Energy Partition'。"

    # 3. 解析参数
    args = parser.parse_args()

    # 4. 关键点：拦截 convert 命令
    if args.command == 'convert':
        # 替换执行函数为我们的 override 版本
        args.func = handle_convert_override

    # 执行 (generate-template 等其他命令保持原样)
    args.func(args)