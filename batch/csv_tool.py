# Plasma_Simulation/batch/csv_tool.py

import argparse
import csv
import hashlib
import io
import json
import sys
from pathlib import Path

from utils.project_config import COLUMN_TASK_NAME, PROJECT_ROOT

try:
    # 尝试从 simulation 模块导入参数类
    from simulation.config import SimulationParameters
except ImportError as e:
    print(f"错误: 无法导入 'SimulationParameters'。请确保 '{PROJECT_ROOT / 'simulation' / 'config.py'}' 文件存在且无误。", file=sys.stderr)
    print(f"详细错误: {e}", file=sys.stderr)
    sys.exit(1)

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# --- 辅助函数 ---

def generate_param_hash(params: dict):
    """根据参数内容生成唯一指纹"""
    # 确保 key 排序一致，序列化为字符串
    param_str = json.dumps(params, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(param_str.encode('utf-8')).hexdigest()

def get_simulation_params_info():
    """
    通过内省 SimulationParameters 类获取参数名称、类型和默认值。
    """
    default_instance = SimulationParameters()
    params_info = {}
    # 获取在类中定义的属性顺序
    attributes = [attr for attr in dir(default_instance) if not attr.startswith('__') and not callable(getattr(default_instance, attr))]

    for attr_name in attributes:
        default_value = getattr(default_instance, attr_name)
        param_type = type(default_value)
        params_info[attr_name] = {
            "type": param_type,
            "default": default_value
        }
    return params_info, attributes


def load_defaults_from_json(params_info, json_path_str):
    """
    如果提供了JSON文件路径，则尝试加载默认值并覆盖类中的默认值。
    """
    # 如果用户没有提供 --defaults-json 参数，则直接返回
    if not json_path_str:
        print("信息: 未提供自定义默认值文件，将仅使用类内置的默认值。")
        return params_info

    json_path = Path(json_path_str)
    if not json_path.exists():
        print(f"警告: 提供的默认值文件不存在: '{json_path}'。将使用类内置默认值。", file=sys.stderr)
        return params_info

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            json_defaults = json.load(f)

        for name, value in json_defaults.items():
            if name in params_info:
                # 更新 params_info 中的默认值
                params_info[name]['default'] = value
        print(f"信息: 已成功从 '{json_path}' 加载并应用自定义默认值。")
    except Exception as e:
        print(f"警告: 加载 '{json_path}' 失败，将使用类内置默认值。错误: {e}", file=sys.stderr)

    return params_info


def smart_type_convert(value_str: str, target_type: type):
    """
    将 CSV 中的字符串智能转换为目标 Python 类型。
    """
    # 处理空字符串，视为空值
    if not value_str.strip():
        return None

    if target_type is bool:
        # 对布尔值进行宽松的判断
        return value_str.lower() in ['true', '1', 't', 'y', 'yes']
    try:
        # 对其他类型直接进行转换，float()可以正确处理科学计数法
        return target_type(value_str)
    except (ValueError, TypeError):
        # 如果转换失败，返回原始字符串，让后续的 Pydantic 等模型去验证
        print(f"警告: 值 '{value_str}' 无法转换为 {target_type.__name__} 类型，将保持为字符串。")
        return value_str


def parse_csv_row(row, params_info):
    """提取并转换单行参数"""
    task_params = {}
    for param_name, value_str in row.items():
        if param_name not in params_info: continue
        target_type = params_info[param_name]['type']
        converted_value = smart_type_convert(value_str, target_type)
        if converted_value is not None:
            task_params[param_name] = converted_value
    return task_params


def perform_conversion_logic(input_csv_path, output_jsonl_path, params_info, param_processor=None):
    """
    通用转换逻辑
    :param param_processor: 一个函数 func(params) -> params，用于在Hash前修改参数
    """
    tasks_to_write = []

    try:
        with open(input_csv_path, 'r', newline='', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                if not any(str(v).strip() for v in row.values()): continue

                raw_name = row.pop(COLUMN_TASK_NAME, '').strip() or "sim"
                task_params = parse_csv_row(row, params_info)

                if param_processor:
                    task_params = param_processor(task_params, line_num=i + 2)

                # 生成 Hash
                p_hash = generate_param_hash(task_params)
                final_task_name = f"{raw_name}_{p_hash}"

                # 我们不再生成 WSL 路径，只记录任务标识和参数
                tasks_to_write.append({
                    "hash": p_hash,
                    "task_name": final_task_name,
                    "params": task_params
                })

        output_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_jsonl_path, 'w', encoding='utf-8') as f:
            for task in tasks_to_write:
                f.write(json.dumps(task) + '\n')
        print(f"[SUCCESS] 转换完成: {len(tasks_to_write)} 个任务已加入队列。")
    except Exception as e:
        print(f"[ERROR] 错误: {e}")
        sys.exit(1)


# --- 命令处理函数 ---

def handle_generate_template(args):
    """
    处理 'generate-template' 子命令的逻辑。
    """
    params_info, param_order = get_simulation_params_info()

    # 从命令行参数获取可选的json文件路径并加载
    params_info = load_defaults_from_json(params_info, args.defaults_json)

    output_path = args.output

    try:
        with open(output_path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            headers = [COLUMN_TASK_NAME] + param_order
            writer.writerow(headers)
            default_row = ['example_run_1'] + [params_info[name]['default'] for name in param_order]
            writer.writerow(default_row)

        print(f"[SUCCESS] 成功生成 CSV 模板文件: '{output_path}'")
        print("   现在你可以用电子表格软件打开它，并根据需要添加更多的任务行。")
    except Exception as e:
        print(f"[ERROR] 错误: 生成模板文件失败。错误: {e}", file=sys.stderr)
        sys.exit(1)


def handle_convert(args, param_processor=None):
    """
    处理 'convert' 命令
    """
    input_csv_path = Path(args.input_csv)
    if not input_csv_path.exists():
        print(f"[ERROR] 错误: 文件不存在: '{input_csv_path}'", file=sys.stderr)
        sys.exit(1)

    # 确定输出路径并获取其绝对父目录（默认输出文件在输入文件旁边）
    output_jsonl_path = Path(args.output or input_csv_path.parent / 'queue.jsonl').resolve()

    params_info, _ = get_simulation_params_info()

    perform_conversion_logic(input_csv_path, output_jsonl_path, params_info, param_processor)


# --- 主程序入口 ---

# --- Argument Parser 封装 ---

def setup_parser():
    parser = argparse.ArgumentParser(
        description="Plasma Simulation CSV Tool - 用于生成模板和转换任务队列。",
        formatter_class=argparse.RawTextHelpFormatter
    )
    subparsers = parser.add_subparsers(dest='command', required=True, help='可用的子命令')

    # --- 'generate-template' 子命令 ---
    parser_gen = subparsers.add_parser(
        'generate-template',
        help='生成一个包含所有参数和默认值的CSV模板文件。',
        description='此命令会读取 simulation.config 来创建一个CSV模板。你还可以选择性地提供一个JSON文件来覆盖默认值。'
    )
    parser_gen.add_argument(
        '-o', '--output',
        default='template.csv',
        help='输出模板文件的路径 (默认: template.csv)'
    )
    # --- 新增的可选参数 ---
    parser_gen.add_argument(
        '-d', '--defaults-json',
        metavar='PATH_TO_JSON',
        help='可选的JSON文件路径，用于覆盖从代码中读取的默认参数。'
    )
    parser_gen.set_defaults(func=handle_generate_template)

    # --- 'convert' 子命令 ---
    parser_conv = subparsers.add_parser(
        'convert',
        help='将一个CSV文件转换为 batch_runner.py 使用的 queue.jsonl 文件。',
        description='此命令会读取指定的CSV文件，将其每一行转换为一个JSON对象，并写入到.jsonl文件中。它会自动处理数据类型。'
    )
    parser_conv.add_argument(
        'input_csv',
        help='输入的CSV文件的路径。'
    )
    parser_conv.add_argument(
        '-o', '--output',
        help='输出的 queue.jsonl 文件的路径 (默认: 与输入文件同目录的 queue.jsonl)'
    )
    parser_conv.set_defaults(func=handle_convert)

    return parser


if __name__ == "__main__":
    parser = setup_parser()
    args = parser.parse_args()
    args.func(args)
