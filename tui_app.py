"""
等离子体模拟任务管理器 - TUI 版本
运行: python tui_app.py
"""
import sys
from pathlib import Path

from tui.app import SimulationTUI

def main():
    app = SimulationTUI()
    app.run()


if __name__ == "__main__":
    main()
