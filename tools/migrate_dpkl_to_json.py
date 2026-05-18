#!/usr/bin/env python3
"""将现有 sim_parameters.dpkl 文件转换为 sim_parameters.json。

扫描 sim_jobs/ 下所有 sim_parameters.dpkl，用 dill 读取后以 JSON 写出。
转换后保留原 dpkl 文件（不删除），需要手动确认后清理。

用法:
    python tools/migrate_dpkl_to_json.py           # 扫描并报告
    python tools/migrate_dpkl_to_json.py --run      # 执行转换
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def find_dpkl_files(root: Path) -> list[Path]:
    """递归查找所有 sim_parameters.dpkl 文件。"""
    return sorted(root.rglob("sim_parameters.dpkl"))


def convert_one(dpkl_path: Path, dry_run: bool = True) -> bool:
    """转换单个 dpkl 文件为 json。返回是否成功。"""
    json_path = dpkl_path.with_suffix(".json")

    if json_path.exists():
        print(f"  跳过 (json 已存在): {dpkl_path.relative_to(PROJECT_ROOT)}")
        return False

    try:
        import dill
        with open(dpkl_path, "rb") as f:
            data = dill.load(f)
    except ImportError:
        print("  错误: 需要安装 dill 才能读取 dpkl 文件 (pip install dill)")
        return False
    except Exception as e:
        print(f"  错误: 读取失败 {dpkl_path}: {e}")
        return False

    if not isinstance(data, dict):
        # 尝试从对象提取属性
        data = {k: v for k, v in data.__dict__.items()
                if isinstance(v, (int, float, str, bool, list, tuple))}
        data["_source_type"] = type(data).__name__

    if dry_run:
        print(f"  将转换: {dpkl_path.relative_to(PROJECT_ROOT)} ({len(data)} 个参数)")
        return True

    try:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"  ✔ {dpkl_path.relative_to(PROJECT_ROOT)} -> {json_path.name}")
        return True
    except (TypeError, ValueError) as e:
        print(f"  错误: JSON 序列化失败 {dpkl_path}: {e}")
        print(f"       包含不可序列化的值，请检查参数类型")
        return False


def main():
    parser = argparse.ArgumentParser(description="将 sim_parameters.dpkl 转换为 sim_parameters.json")
    parser.add_argument("--run", action="store_true", help="执行转换（默认只扫描报告）")
    args = parser.parse_args()

    sim_jobs = PROJECT_ROOT / "sim_jobs"
    if not sim_jobs.exists():
        print(f"错误: {sim_jobs} 不存在")
        sys.exit(1)

    dpkl_files = find_dpkl_files(sim_jobs)
    if not dpkl_files:
        print("未找到任何 sim_parameters.dpkl 文件。")
        return

    print(f"找到 {len(dpkl_files)} 个 dpkl 文件")
    mode = "执行模式" if args.run else "扫描模式 (加 --run 执行转换)"
    print(f"模式: {mode}\n")

    success = 0
    for dpkl_path in dpkl_files:
        if convert_one(dpkl_path, dry_run=not args.run):
            success += 1

    print(f"\n完成: {success} 个文件{'将被' if not args.run else '已'}转换")


if __name__ == "__main__":
    main()
