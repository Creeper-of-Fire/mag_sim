# file_dialog.py - 文件对话框工具
"""
跨平台文件/目录选择对话框
自动选择最佳可用的实现方式
"""
import os
import sys
import subprocess
from pathlib import Path


def _open_directory_tkinter(title: str, initial_dir: str) -> str | None:
    """通过 tkinter 打开目录选择对话框"""
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)

        selected = filedialog.askdirectory(
            title=title,
            initialdir=initial_dir
        )
        root.destroy()
        return selected if selected else None
    except Exception as e:
        print(f"tkinter 对话框失败: {e}")
        return None


def _open_directory_zenity(title: str, initial_dir: str) -> str | None:
    """通过 zenity 打开目录选择对话框（Linux）"""
    try:
        result = subprocess.run(
            ['zenity', '--file-selection', '--directory',
             '--title', title,
             '--filename', initial_dir],
            capture_output=True,
            text=True,
            timeout=60
        )
        if result.returncode == 0:
            path = result.stdout.strip()
            if path and Path(path).exists():
                return path
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def _open_directory_kdialog(title: str, initial_dir: str) -> str | None:
    """通过 kdialog 打开目录选择对话框（Linux KDE）"""
    try:
        result = subprocess.run(
            ['kdialog', '--getexistingdirectory', initial_dir,
             '--title', title],
            capture_output=True,
            text=True,
            timeout=60
        )
        if result.returncode == 0:
            path = result.stdout.strip()
            if path and Path(path).exists():
                return path
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def open_directory_dialog(title: str = "选择目录",
                          initial_dir: str = None) -> str | None:
    """
    打开目录选择对话框，自动选择最佳可用方式

    Args:
        title: 对话框标题
        initial_dir: 初始目录，默认为用户主目录

    Returns:
        选中的目录路径，取消返回 None
    """
    if initial_dir is None:
        initial_dir = str(Path.home())

    # 确保初始目录存在
    if not Path(initial_dir).exists():
        initial_dir = str(Path.home())

    # Linux: 优先使用原生工具
    if sys.platform == 'linux':
        # 尝试各种 Linux 原生工具
        for method in [_open_directory_zenity, _open_directory_kdialog]:
            result = method(title, initial_dir)
            if result:
                return result

    # 通用回退：使用 tkinter
    return _open_directory_tkinter(title, initial_dir)