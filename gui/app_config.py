# gui/app_config.py

import os
from pathlib import Path

from utils.project_config import PROJECT_ROOT

# 定义所有数据文件（如队列、默认参数）的存储目录。
# 相对于项目的根目录。
DATA_DIR = os.path.join(PROJECT_ROOT, Path("gui") / "data")
