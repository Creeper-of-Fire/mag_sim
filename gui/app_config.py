# gui/app_config.py

import os
from pathlib import Path

# --- 路径配置 ---

# 获取项目的根目录。
# __file__ 是当前文件 (app_config.py) 的路径。
# os.path.dirname(__file__) 是 config 目录。
# os.path.dirname(os.path.dirname(__file__)) 是项目的根目录。
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 定义所有数据文件（如队列、默认参数）的存储目录。
# 相对于项目的根目录。
DATA_DIR = os.path.join(PROJECT_ROOT, Path("gui") / "data")
