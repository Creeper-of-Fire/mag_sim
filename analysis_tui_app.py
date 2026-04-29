#!/usr/bin/env python3
"""分析 TUI 入口"""
from analysis_tui.app import AnalysisTUI


def main():
    app = AnalysisTUI()
    app.run()


if __name__ == "__main__":
    main()
