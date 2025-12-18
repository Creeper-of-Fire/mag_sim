# main.py
import argparse
import atexit
import json
import os
import subprocess
import sys

from simulation.config import SimulationParameters
from simulation.simulation import PlasmaReconnection


def run_notification_script():
    """
    在脚本退出时执行通知脚本。
    这是一个健壮的版本，能处理脚本不存在或执行出错的情况。
    """
    try:
        # 假设 notify_me.sh 和 main.py 在同一个目录下
        script_dir = os.getcwd()
        notification_script_path = os.path.join(script_dir, 'notify_me.sh')

        print(f"计算的通知脚本路径：{notification_script_path}")
        if os.path.exists(notification_script_path):
            print("\n--- 模拟结束，正在发送桌面通知和播放音乐... ---")
            # 使用 Popen 可以在后台“即发即忘”地运行脚本，
            # 不会阻塞 Python 主程序的退出过程。
            result = subprocess.run(
                [notification_script_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            # 打印脚本输出（方便调试）
            if result.stdout:
                print(f"通知脚本输出: {result.stdout}")
            if result.stderr:
                print(f"通知脚本错误: {result.stderr}")
        else:
            print(f"\n--- 警告：未找到通知脚本 {notification_script_path} ---")
    except Exception as e:
        # 捕获任何可能的错误，避免通知失败影响到程序本身
        print(f"--- 错误：执行通知脚本失败: {e} ---")


def main():
    """
    主执行函数
    """
    # --- 在程序一开始就注册退出时要执行的函数 ---
    # 无论程序是正常结束还是中途崩溃，这个函数都会被调用
    atexit.register(run_notification_script)
    # -------------------------------------------
    exec ("from simulation.config import SimulationParameters")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-v",
        "--verbose",
        help="Verbose output",
        action="store_true",
    )
    parser.add_argument(
        "-c",
        "--config-json",
        help="Simulation parameters in JSON format",
        type=str,
        default=None
    )
    args, left = parser.parse_known_args()
    sys.argv = sys.argv[:1] + left

    # 1. 从 config.py 加载参数
    params = SimulationParameters()

    # 如果通过命令行传递了JSON配置，就用它来覆盖默认值
    if args.config_json:
        print("--- Loading parameters from JSON config ---")
        try:
            custom_params = json.loads(args.config_json)
            # 遍历传入的参数，更新 params 实例
            for key, value in custom_params.items():
                if hasattr(params, key):
                    # 获取原始属性的类型，并进行类型转换
                    original_type = type(getattr(params, key))
                    setattr(params, key, original_type(value))
                    print(f"  - Overriding {key}: {value}")
                else:
                    print(f"  - Warning: Unknown parameter '{key}' in JSON config.")
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            sys.exit(1)

    # 2. 将参数传递给模拟引擎来设置并运行模拟
    #    注意：PlasmaReconnection的__init__方法会调用setup_run和所有初始化
    run = PlasmaReconnection(params=params, verbose=args.verbose)

    run.run_simulation()

    print("Simulation finished.")


if __name__ == "__main__":
    main()
