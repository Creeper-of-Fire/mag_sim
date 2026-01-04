# main.py
import argparse
import json
import os
import sys

from simulation.config import SimulationParameters
from simulation.simulation import PlasmaReconnection

def main():
    """
    主执行函数
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-v",
        "--verbose",
        help="详细输出",
        action="store_true",
    )
    parser.add_argument(
        "-c",
        "--config-json",
        help="JSON格式的模拟参数",
        type=str,
        default=None
    )
    parser.add_argument(
        "-o",
        "--output",
        help="[必须] 输出目录路径（建议使用绝对路径）",
        type=str,
        required=True  # 提供一个合理的默认值
    )
    args, left = parser.parse_known_args()
    sys.argv = sys.argv[:1] + left

    # 将输出路径转换为绝对路径，这是一种很好的实践
    output_dir = os.path.abspath(os.path.expanduser(args.output))
    print(f"--- 设置输出目录为: {output_dir} ---")

    # 1. 从 config.py 加载参数
    params = SimulationParameters()

    # 如果通过命令行传递了JSON配置，就用它来覆盖默认值
    if args.config_json:
        print("--- 从JSON配置加载参数 ---")
        try:
            custom_params = json.loads(args.config_json)
            # 遍历传入的参数，更新 params 实例
            for key, value in custom_params.items():
                if hasattr(params, key):
                    # 获取原始属性的类型，并进行类型转换
                    original_type = type(getattr(params, key))
                    setattr(params, key, original_type(value))
                    print(f"  - 覆盖 {key}: {value}")
                else:
                    print(f"  - 警告: JSON配置中存在未知参数 '{key}'。")
        except json.JSONDecodeError as e:
            print(f"错误: JSON解码失败: {e}")
            sys.exit(1)

    # 2. 将参数传递给模拟引擎来设置并运行模拟
    #    注意：PlasmaReconnection的__init__方法会调用setup_run和所有初始化
    run = PlasmaReconnection(params=params, output_dir=output_dir, verbose=args.verbose)

    run.run_simulation()

    print("模拟结束。")


if __name__ == "__main__":
    main()
