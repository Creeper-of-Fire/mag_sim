#!/usr/bin/env python3
"""BBN 混合方法模拟入口（占位）。"""
import argparse
import json
import os
import sys

from simulation.bbn.config import SimulationParameters
from simulation.bbn.engine import PlasmaSimulation  # 占位，后续实现


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-c", "--config-json", type=str, default=None)
    parser.add_argument("-o", "--output", type=str, required=True)
    args, left = parser.parse_known_args()
    sys.argv = sys.argv[:1] + left

    output_dir = os.path.abspath(os.path.expanduser(args.output))
    print(f"--- 设置输出目录为: {output_dir} ---")

    params = SimulationParameters()
    if args.config_json:
        print("--- 从JSON配置加载参数 ---")
        try:
            custom_params = json.loads(args.config_json)
            for key, value in custom_params.items():
                if hasattr(params, key):
                    original_type = type(getattr(params, key))
                    setattr(params, key, original_type(value))
                    print(f"  - 覆盖 {key}: {value}")
                else:
                    print(f"  - 警告: JSON配置中存在未知参数 '{key}'。")
        except json.JSONDecodeError as e:
            print(f"错误: JSON解码失败: {e}")
            sys.exit(1)

    run = PlasmaSimulation(params=params, output_dir=output_dir, verbose=args.verbose)
    run.run_simulation()
    print("模拟结束。")


if __name__ == "__main__":
    main()
