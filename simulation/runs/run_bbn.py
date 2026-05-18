#!/usr/bin/env python3
"""BBN 混合 PIC 模拟入口。"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from simulation.bbn.config import SimulationParameters
from simulation.bbn.engine import BBNHybridSimulation


def main():
    parser = argparse.ArgumentParser(description="BBN hybrid PIC simulation")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-c", "--config-json", type=str, default=None)
    parser.add_argument("-o", "--output", type=str, required=True)
    args, left = parser.parse_known_args()
    sys.argv = sys.argv[:1] + left

    output_dir = os.path.abspath(os.path.expanduser(args.output))
    print(f"--- 输出目录: {output_dir} ---")

    params = SimulationParameters()
    if args.config_json:
        print("--- 从 JSON 加载参数 ---")
        custom = json.loads(args.config_json)
        for key, value in custom.items():
            if hasattr(params, key):
                setattr(params, key, type(getattr(params, key))(value))
                print(f"  覆盖 {key}: {value}")

    run = BBNHybridSimulation(params=params, output_dir=output_dir, verbose=args.verbose)
    run.run_simulation()


if __name__ == "__main__":
    main()
