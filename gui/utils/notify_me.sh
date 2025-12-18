#!/bin/bash

# 定义通知的标题和内容
TITLE="模拟完成"
MESSAGE="您的 WarpX 模拟已经顺利结束！"

# --- 调用 Windows PowerShell 发送一个美观的 Toast 通知 ---
# 这个命令直接让 Windows 创建一个通知，绕过了 WSL 的 D-Bus
powershell.exe -Command "New-BurntToastNotification -Text '$TITLE', '$MESSAGE'"

SONG_PATH='D:\User\Desktop\Project\notify\blue_latus_hajimi.mp4'

# --- 调用 PowerShell 来启动默认播放器播放文件 ---
# "start-process" 命令就相当于你在 Windows 里双击了这个文件
# & at the end of the shell command runs it in the background
powershell.exe -Command "Start-Process '$SONG_PATH'" &