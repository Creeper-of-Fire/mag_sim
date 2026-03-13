#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import os
import re
import shutil
from pathlib import Path
from typing import List

from rich.prompt import Prompt, Confirm
from rich.table import Table

from analysis.core.utils import console


def _extract_step(filename: str) -> int:
    """从文件名中提取步数，例如 diag1_00500.h5 -> 500"""
    match = re.search(r'_(\d+)\.(h5|plt|txt)', filename)
    if match:
        return int(match.group(1))
    return -1


def _get_available_steps(run_dir: Path) -> List[int]:
    """扫描 diags 目录获取所有可用的时间步"""
    diags_dir = run_dir / "diags"
    if not diags_dir.exists():
        return []

    steps = set()
    for diag_sub in diags_dir.iterdir():
        if diag_sub.is_dir():
            for f in diag_sub.iterdir():
                step = _extract_step(f.name)
                if step != -1:
                    steps.add(step)
    return sorted(list(steps))


def create_virtual_slices(run_dir: Path):
    """创建虚拟切片目录"""
    console.print(f"\n[bold cyan]--- 正在处理模拟: {run_dir.name} ---[/bold cyan]")
    available_steps = _get_available_steps(run_dir)

    if not available_steps:
        console.print("[red]未在 diags/ 目录中找到任何时间步文件。[/red]")
        return

    console.print(f"找到 [green]{len(available_steps)}[/green] 个可用时间步:")
    # 格式化打印可用步数 (每行显示10个)
    step_strs = [str(s) for s in available_steps]
    for i in range(0, len(step_strs), 10):
        console.print("  " + ", ".join(step_strs[i:i + 10]))

    target_input = Prompt.ask(
        "\n请输入要切片的时间步 (用逗号分隔，如 '100,200,500'，输入 'all' 切片所有步)",
        default="all"
    )

    if target_input.strip().lower() == 'all':
        target_steps = available_steps
    else:
        try:
            target_steps = [int(s.strip()) for s in target_input.split(',')]
            # 过滤掉不存在的步数
            target_steps = [s for s in target_steps if s in available_steps]
        except ValueError:
            console.print("[red]输入格式错误，操作取消。[/red]")
            return

    if not target_steps:
        console.print("[yellow]未选中任何有效的时间步。[/yellow]")
        return

    # 开始创建虚拟目录
    created_count = 0
    with console.status("[bold green]正在建立软链接和虚拟参数...[/bold green]"):
        for step in target_steps:
            virtual_name = f"{run_dir.name}_slice_{step:06d}"
            virtual_dir = run_dir.parent / virtual_name

            # 1. 创建目录
            virtual_dir.mkdir(exist_ok=True)

            # 2. 软链接参数文件 dpkl
            dpkl = run_dir / "sim_parameters.dpkl"
            virt_dpkl = virtual_dir / dpkl.name
            if dpkl.exists() and not virt_dpkl.exists():
                os.symlink(dpkl.resolve(), virt_dpkl)

            # 3. 软链接 diags 目录 (只链接 <= step 的文件)
            orig_diags = run_dir / "diags"
            virt_diags = virtual_dir / "diags"
            virt_diags.mkdir(exist_ok=True)

            for diag_sub in orig_diags.iterdir():
                if not diag_sub.is_dir(): continue
                virt_diag_sub = virt_diags / diag_sub.name
                virt_diag_sub.mkdir(exist_ok=True)

                for f in diag_sub.iterdir():
                    f_step = _extract_step(f.name)
                    # 核心逻辑：只链接时间步小于等于目标时间步的文件
                    if f_step != -1 and f_step <= step:
                        virt_f = virt_diag_sub / f.name
                        if not virt_f.exists():
                            # 使用绝对路径创建软链接，确保不会断链
                            os.symlink(f.resolve(), virt_f)

            # 4. 注入 custom_params.json 欺骗框架
            custom_params = {
                "_is_virtual_slice": True,
                "slice_step": step,  # 你的分析模块将看到这个参数
                "virtual_time_step": step
            }
            with open(virtual_dir / "custom_params.json", 'w', encoding='utf-8') as f:
                json.dump(custom_params, f, indent=4)

            created_count += 1

    console.print(f"[green]✔ 成功创建 {created_count} 个虚拟切片！[/green]")
    console.print(f"[dim]它们将伪装成独立的模拟，且变量 'slice_step' 会发生变化。[/dim]")


def remove_virtual_slices(run_dir: Path):
    """清理属于指定 run_dir 的虚拟切片"""
    parent_dir = run_dir.parent
    base_name = run_dir.name

    slices_to_remove = []

    # 扫描父目录
    for d in parent_dir.iterdir():
        if d.is_dir() and d.name.startswith(f"{base_name}_slice_"):
            custom_param_file = d / "custom_params.json"
            if custom_param_file.exists():
                try:
                    with open(custom_param_file, 'r') as f:
                        data = json.load(f)
                        # 双重确认：这确实是我们生成的虚拟切片
                        if data.get("_is_virtual_slice") is True:
                            slices_to_remove.append(d)
                except Exception:
                    pass

    if not slices_to_remove:
        console.print(f"[yellow]未在 {parent_dir.name} 下找到属于 {base_name} 的虚拟切片。[/yellow]")
        return

    console.print(f"\n[bold yellow]找到 {len(slices_to_remove)} 个虚拟切片目录:[/bold yellow]")
    for d in slices_to_remove[:5]:
        console.print(f"  - {d.name}")
    if len(slices_to_remove) > 5:
        console.print(f"  ... 及其他 {len(slices_to_remove) - 5} 个目录")

    if Confirm.ask("确认要删除以上虚拟目录吗？(仅删除软链接，原数据安全)", default=True):
        for d in slices_to_remove:
            shutil.rmtree(d)
        console.print("[green]✔ 清理完成。[/green]")


def run_interactive_workflow(selected_dirs: List[str]):
    """工具入口点"""
    console.print("[bold magenta]=== 虚拟时间切片工具 (Slicer) ===[/bold magenta]")
    console.print("[dim]原理：通过软链接拦截未来的时间步，并注入 custom_params.json。[/dim]")
    console.print("[dim]这样可以将单次模拟的不同演化时刻，转化为对比分析模块中的参数化扫描 (X轴)。[/dim]\n")

    action = Prompt.ask(
        "请选择操作 - \n[bold cyan]1[/bold cyan] - 创建虚拟时间切片 (Slicer)\n[bold cyan]2[/bold cyan] - 清理已存在的切片 (Unslicer)\n输入: ",
        choices=["1", "2"],
        default="1",
        show_choices=False
    )

    for dir_str in selected_dirs:
        run_dir = Path(dir_str).resolve()
        if action == "1":
            create_virtual_slices(run_dir)
        elif action == "2":
            remove_virtual_slices(run_dir)