# gui/utils/notifications.py

import os
import subprocess

# 获取当前脚本的绝对路径
CURRENT_FILE_PATH = os.path.abspath(__file__)
# 获取当前脚本所在的目录
SCRIPT_DIRECTORY = os.path.dirname(CURRENT_FILE_PATH)


def run_notification_script():
    """
    在脚本退出时执行通知脚本。
    这是一个健壮的版本，能处理脚本不存在或执行出错的情况。

    返回:
        str: 执行结果的日志信息。
    """
    # 用于收集所有日志信息
    log_messages = []
    try:
        notification_script_path = os.path.join(SCRIPT_DIRECTORY, 'notify_me.sh')

        log_messages.append(f"计算的通知脚本路径：{notification_script_path}")
        if os.path.exists(notification_script_path):
            log_messages.append("\n--- 模拟结束，正在发送桌面通知和播放音乐... ---")
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
                log_messages.append(f"通知脚本输出: {result.stdout}")
            if result.stderr:
                log_messages.append(f"通知脚本错误: {result.stderr}")
        else:
            log_messages.append(f"\n--- 警告：未找到通知脚本 {notification_script_path} ---")
    except Exception as e:
        # 捕获任何可能的错误，避免通知失败影响到程序本身
        log_messages.append(f"--- 错误：执行通知脚本失败: {e} ---")

    # 将所有日志信息合并成一个字符串并返回
    return "\n".join(log_messages)
