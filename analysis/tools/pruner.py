# analysis/tools/pruner.py
import os
from pathlib import Path
from typing import List, Tuple
from rich.table import Table
from rich.prompt import Confirm
from rich.progress import track
from ..core.utils import console


def get_pruning_plan(directory: str) -> Tuple[List[Path], List[Path], List[Path]]:
    """
    分析目录，返回 (待删除列表, 待保留列表, 所有文件列表)
    保留逻辑：First, Middle, Last
    """
    path = Path(directory) / "diags" / "particle_states"
    if not path.exists():
        return [], [], []

    # 按文件名排序，通常文件名包含 step，排序即按时间排序
    files = sorted(list(path.glob("*.h5")))
    total_count = len(files)

    # 如果文件太少，不需要修剪
    if total_count <= 3:
        return [], files, files

    # 确定保留的索引
    first_idx = 0
    last_idx = total_count - 1
    mid_idx = total_count // 2  # 整数除法取中间

    keep_indices = {first_idx, mid_idx, last_idx}

    to_delete = []
    to_keep = []

    for i, f in enumerate(files):
        if i in keep_indices:
            to_keep.append(f)
        else:
            to_delete.append(f)

    return to_delete, to_keep, files


def run_pruner_interactive(selected_dirs: List[str]):
    """
    交互式修剪工具入口
    """
    console.print("\n[bold red]启动时间步修剪工具 (Pruner)[/bold red]")
    console.print("[yellow]警告：此工具将删除绝大多数粒子文件，只保留 [u]最初、中间、最后[/u] 三个时刻！[/yellow]\n")

    all_deletion_tasks = []  # 存储所有待删除文件的路径

    # --- 1. 扫描并展示计划 ---
    table = Table(title="修剪计划概览", show_lines=True)
    table.add_column("目录名", style="cyan")
    table.add_column("总文件数", justify="right")
    table.add_column("将删除", justify="right", style="red")
    table.add_column("将保留 (示例)", style="green")

    total_delete_count = 0

    for d in selected_dirs:
        to_delete, to_keep, all_files = get_pruning_plan(d)

        if not all_files:
            continue

        dir_name = Path(d).name

        # 格式化保留文件的提示 (显示文件名)
        keep_names = [f.name for f in to_keep]
        keep_str = ", ".join(keep_names)

        table.add_row(
            dir_name,
            str(len(all_files)),
            str(len(to_delete)),
            keep_str
        )

        all_deletion_tasks.extend(to_delete)
        total_delete_count += len(to_delete)

    if total_delete_count == 0:
        console.print("[green]没有发现需要修剪的多余文件 (每个目录文件数均 <= 3)。[/green]")
        return

    console.print(table)
    console.print(f"\n[bold white]总计将永久删除 {total_delete_count} 个文件。[/bold white]")

    # --- 2. 最终确认 ---
    # 这是一个危险操作，默认选 No
    if not Confirm.ask("[bold red blink]确定要执行删除吗？此操作无法撤销！[/bold red blink]", default=False):
        console.print("[yellow]操作已取消。[/yellow]")
        return

    # --- 3. 执行删除 ---
    console.print("\n开始执行清理...")

    deleted_count = 0
    errors = 0

    # 使用 track 显示简单进度条
    for file_path in track(all_deletion_tasks, description="Deleting files..."):
        try:
            os.remove(file_path)
            deleted_count += 1

            # 同时尝试删除对应的同名预览图 (如果存在)
            # 假设预览图在 output/slimmer_previews 下，且命名规则已知
            # 这一步是可选的，看你的文件结构，这里只做最保守的同目录清理

        except Exception as e:
            console.print(f"[red]删除失败 {file_path.name}: {e}[/red]")
            errors += 1

    console.print(f"\n[green]清理完成！成功删除: {deleted_count}，失败: {errors}[/green]")

    # 再次提醒剩余文件
    console.print("[dim]提示：剩下的 First/Middle/Last 文件依然完整保留。[/dim]")