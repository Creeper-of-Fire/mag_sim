# main.py
import argparse
import sys
from config import SimulationParameters
from simulation import PlasmaReconnection


def main():
    """
    主执行函数
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-v",
        "--verbose",
        help="Verbose output",
        action="store_true",
    )
    args, left = parser.parse_known_args()
    sys.argv = sys.argv[:1] + left

    # 1. 从 config.py 加载参数
    params = SimulationParameters()

    # 2. 将参数传递给模拟引擎来设置并运行模拟
    #    注意：PlasmaReconnection的__init__方法会调用setup_run和所有初始化
    run = PlasmaReconnection(params=params, verbose=args.verbose)

    run.run_simulation()

    print("Simulation finished.")


if __name__ == "__main__":
    main()