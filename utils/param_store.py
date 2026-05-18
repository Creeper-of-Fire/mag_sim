"""模拟参数存储接口。

所有模拟参数的读写都通过这个模块，格式为 JSON。
文件名常量、读写函数集中在此，外部无需知道具体格式。
"""

import json
import os
from pathlib import Path
from types import SimpleNamespace

PARAM_FILENAME = "sim_parameters.json"


def save(output_dir: str | Path, params: dict) -> Path:
    """保存参数到目录。返回写入的文件路径。"""
    output_dir = Path(output_dir)
    path = output_dir / PARAM_FILENAME
    with open(path, "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2, ensure_ascii=False)
    return path


def load(output_dir: str | Path) -> dict:
    """从目录加载参数为 dict。"""
    path = Path(output_dir) / PARAM_FILENAME
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_as_namespace(output_dir: str | Path) -> SimpleNamespace:
    """从目录加载参数为 SimpleNamespace（属性访问风格）。"""
    return SimpleNamespace(**load(output_dir))


def exists(output_dir: str | Path) -> bool:
    """检查目录中是否存在参数文件。"""
    return (Path(output_dir) / PARAM_FILENAME).is_file()


def find_all(root: Path) -> list[Path]:
    """递归查找 root 下所有包含参数文件的目录。"""
    return sorted(
        {p.parent for p in root.rglob(PARAM_FILENAME)},
        key=lambda x: x.name,
    )
