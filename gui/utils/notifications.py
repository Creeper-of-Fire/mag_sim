# gui/utils/notifications.py

import os
import random
import subprocess

def run_notification_script():
    """
    在脚本退出时执行通知脚本。
    这是一个健壮的版本，能处理脚本不存在或执行出错的情况。

    返回:
        str: 执行结果的日志信息。
    """
    # 用于收集所有日志信息
    log_messages = []

    # --- 配置 ---
    TITLE = "模拟完成"
    MESSAGE = "您的等离子体模拟已经顺利结束！"
    # 注意：这里是Windows风格的路径
    NOTIFY_DIR = r'D:\User\Desktop\Project\notify'

    log_messages.append("\n--- 模拟结束，正在发送桌面通知并挑选随机音乐... ---")

    try:
        # 1. 调用 PowerShell 发送通知 (BurntToast)
        # 我们使用 subprocess.run 直接调用 powershell.exe
        # 为了不弹出蓝色的命令行窗口，我们添加了 creationflags
        ps_command = f"New-BurntToastNotification -Text '{TITLE}', '{MESSAGE}'"

        try:
            result = subprocess.run(
                ["powershell.exe", "-Command", ps_command],
                capture_output=True,
                text=True,
                encoding='utf-8',
                # 这行代码在 Windows 上运行非常重要：它可以防止闪出一个蓝色的控制台窗口
                creationflags=subprocess.CREATE_NO_WINDOW
            )

            if result.returncode == 0:
                log_messages.append("日志: 已成功通过 PowerShell 发送 BurntToast 通知。")
            else:
                log_messages.append(f"错误日志: PowerShell 返回码 {result.returncode}, 错误信息: {result.stderr}")
        except Exception as e:
            log_messages.append(f"错误日志: 调用 PowerShell 失败: {e}")

        # 2. 随机播放音乐
        if os.path.exists(NOTIFY_DIR) and os.path.isdir(NOTIFY_DIR):
            # 获取文件夹下所有的文件
            all_files = os.listdir(NOTIFY_DIR)
            # 筛选出以 .mp4 结尾的文件 (不区分大小写)
            mp4_files = [f for f in all_files if f.lower().endswith('.mp4')]

            if mp4_files:
                # 随机选择一个文件
                chosen_file = random.choice(mp4_files)
                full_path = os.path.join(NOTIFY_DIR, chosen_file)

                try:
                    # 使用 Windows 默认播放器打开
                    os.startfile(full_path)
                    log_messages.append(f"日志: 随机抽选了音乐: {chosen_file}")
                except Exception as e:
                    log_messages.append(f"错误日志: 播放文件 '{chosen_file}' 失败: {e}")
            else:
                log_messages.append(f"警告: 在目录 '{NOTIFY_DIR}' 中未找到任何 .mp4 文件。")
        else:
            log_messages.append(f"错误日志: 通知目录不存在或不是有效的文件夹: {NOTIFY_DIR}")

    except Exception as e:
        log_messages.append(f"--- 严重错误日志：执行通知功能时发生意外错误: {e} ---")

        # 将所有日志信息合并成一个字符串并返回
    return "\n".join(log_messages)

if __name__ == "__main__":
    run_notification_script()