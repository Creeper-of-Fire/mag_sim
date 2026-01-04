# main_gui.py

# !/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
应用程序的主入口点。
负责初始化 QApplication 和主窗口，并启动事件循环。
"""
import os
import sys
from PySide6.QtWidgets import QApplication
from gui.main_window import SimulationControllerGUI

os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
if __name__ == "__main__":
    # 创建 Qt 应用程序实例
    app = QApplication(sys.argv)

    # 创建并显示主窗口
    window = SimulationControllerGUI()
    window.show()

    # 启动应用程序的事件循环，并等待退出
    sys.exit(app.exec())