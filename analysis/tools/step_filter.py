import re
from pathlib import Path
from typing import List, Dict, Any, Optional

from rich.table import Table
from rich.prompt import Prompt, Confirm
from rich.progress import track

from ..core.utils import console


def _extract_step(filename: str) -> int:
    """从文件名中提取步数，仅匹配 .h5（不含 .disable）"""
    match = re.search(r'_(\d+)\.h5$', filename)
    if match:
        return int(match.group(1))
    return -1


def _extract_step_disabled(filename: str) -> int:
    """从 .disable 文件名中提取步数"""
    match = re.search(r'_(\d+)\.h5\.disable$', filename)
    if match:
        return int(match.group(1))
    return -1


def _discover_timesteps(run_dir: Path) -> Dict[str, Any]:
    """扫描 run 目录，返回活跃和已禁用的时间步信息"""
    particle_dir = run_dir / "diags" / "particle_states"
    field_dir = run_dir / "diags" / "field_states"

    active_steps = set()
    disabled_steps = set()
    disabled_files: List[Path] = []

    for d in [particle_dir, field_dir]:
        if not d.exists():
            continue
        for f in d.iterdir():
            step = _extract_step(f.name)
            if step != -1:
                active_steps.add(step)
                continue
            step_d = _extract_step_disabled(f.name)
            if step_d != -1:
                disabled_steps.add(step_d)
                disabled_files.append(f)

    active_steps_sorted = sorted(active_steps)
    disabled_steps_sorted = sorted(disabled_steps)

    return {
        "particle_dir": particle_dir,
        "field_dir": field_dir,
        "active_steps": active_steps_sorted,
        "disabled_steps": disabled_steps_sorted,
        "disabled_files": disabled_files,
    }


# ---- 过滤策略 ----

def _filter_stride(steps: List[int], n: int) -> List[int]:
    """等距采样：保留每第 N 个（按位置索引），始终保留首尾。返回要禁用的步列表。"""
    if n <= 1:
        return []
    keep_indices = set(range(0, len(steps), n))
    keep_indices.add(len(steps) - 1)
    return [s for i, s in enumerate(steps) if i not in keep_indices]


def _filter_endpoints_sparse(steps: List[int], k: int, middle_stride: int) -> List[int]:
    """首尾各保留 K 步，中间区域按 middle_stride 稀疏采样。始终保留第 0 步。返回要禁用的步列表。"""
    total = len(steps)
    if total <= 2 * k:
        return []

    keep_indices = set()
    # 首 K 步（包含 index 0，即 step 0）
    keep_indices.update(range(min(k, total)))
    # 尾 K 步
    keep_indices.update(range(max(0, total - k), total))
    # 中间区域按 stride
    for i in range(k, total - k, middle_stride):
        keep_indices.add(i)

    return [s for i, s in enumerate(steps) if i not in keep_indices]


def _filter_tail_dense(steps: List[int], k: int, front_stride: int) -> List[int]:
    """尾部保留 K 步，前面区域按 front_stride 稀疏采样。始终保留第 0 步。返回要禁用的步列表。"""
    total = len(steps)
    if total <= k:
        return []

    keep_indices = set()
    # 尾 K 步
    keep_indices.update(range(max(0, total - k), total))
    # 前面区域按 stride（range 从 0 开始，始终包含 index 0 即 step 0）
    for i in range(0, total - k, front_stride):
        keep_indices.add(i)

    return [s for i, s in enumerate(steps) if i not in keep_indices]


# ---- 操作函数 ----

def _apply_disable(run_dir: Path, steps_to_disable: List[int]) -> tuple:
    """对指定步重命名 .h5 -> .h5.disable（particle + field 同步）"""
    success, errors = 0, 0
    step_set = set(steps_to_disable)

    for subdir in ["particle_states", "field_states"]:
        d = run_dir / "diags" / subdir
        if not d.exists():
            continue
        for f in sorted(d.iterdir()):
            step = _extract_step(f.name)
            if step != -1 and step in step_set:
                target = f.parent / (f.name + ".disable")
                try:
                    f.rename(target)
                    success += 1
                except Exception as e:
                    console.print(f"[red]重命名失败 {f.name}: {e}[/red]")
                    errors += 1
    return success, errors


def _apply_enable(run_dir: Path, steps_to_enable: Optional[List[int]] = None) -> tuple:
    """恢复 .h5.disable -> .h5。steps_to_enable=None 表示恢复全部"""
    success, errors = 0, 0
    step_set = set(steps_to_enable) if steps_to_enable is not None else None

    for subdir in ["particle_states", "field_states"]:
        d = run_dir / "diags" / subdir
        if not d.exists():
            continue
        for f in sorted(d.iterdir()):
            step = _extract_step_disabled(f.name)
            if step != -1 and (step_set is None or step in step_set):
                target = f.parent / f.name.replace(".disable", "")
                try:
                    f.rename(target)
                    success += 1
                except Exception as e:
                    console.print(f"[red]恢复失败 {f.name}: {e}[/red]")
                    errors += 1
    return success, errors


# ---- 显示函数 ----

def _show_status(runs_info: List[Dict]):
    """展示各 run 的当前状态"""
    table = Table(title="时间步状态总览", show_lines=True)
    table.add_column("目录", style="cyan")
    table.add_column("活跃步数", justify="right", style="green")
    table.add_column("已禁用步数", justify="right", style="red")
    table.add_column("步范围", style="dim")

    for info in runs_info:
        active = info["active_steps"]
        disabled = info["disabled_steps"]
        name = info["name"]
        step_range = f"{active[0]} - {active[-1]}" if active else "-"
        table.add_row(name, str(len(active)), str(len(disabled)), step_range)

    console.print(table)


def _show_preview(runs_plan: List[Dict]):
    """展示过滤计划预览"""
    table = Table(title="过滤计划预览", show_lines=True)
    table.add_column("目录", style="cyan")
    table.add_column("当前活跃步", justify="right")
    table.add_column("将禁用", justify="right", style="red")
    table.add_column("将保留", justify="right", style="green")

    total_disable_files = 0

    for plan in runs_plan:
        to_disable = plan["steps_to_disable"]
        to_keep = plan["steps_to_keep"]
        table.add_row(
            plan["name"],
            str(len(plan["active_steps"])),
            str(len(to_disable)),
            str(len(to_keep)),
        )
        total_disable_files += len(to_disable) * 2  # particle + field

    console.print(table)
    console.print(f"\n[bold]总计将禁用 {total_disable_files} 个文件 "
                  f"({len(runs_plan)} 个 run)。[/bold]")


# ---- 子流程 ----

def _workflow_disable(selected_dirs: List[str]):
    """禁用时间步的交互流程"""
    # 1. 扫描所有 run
    runs_plan = []
    for d in selected_dirs:
        run_dir = Path(d)
        info = _discover_timesteps(run_dir)
        if not info["active_steps"]:
            console.print(f"[yellow]跳过 {run_dir.name}：无活跃时间步[/yellow]")
            continue
        runs_plan.append({
            "name": run_dir.name,
            "path": run_dir,
            "active_steps": info["active_steps"],
            "steps_to_disable": [],
            "steps_to_keep": list(info["active_steps"]),
        })

    if not runs_plan:
        console.print("[red]没有可处理的运行目录。[/red]")
        return

    # 2. 选择策略
    console.print("\n[bold]请选择过滤策略:[/bold]")
    console.print("  [cyan]1[/] - 等距采样（每 N 步保留 1 个）")
    console.print("  [cyan]2[/] - 首尾密集 + 中间稀疏")
    console.print("  [cyan]3[/] - 尾部密集 + 前面稀疏")
    strategy = Prompt.ask("选择策略", choices=["1", "2", "3"], default="1")

    if strategy == "1":
        n = int(Prompt.ask("保留间隔 N（每 N 步保留 1 个）", default="10"))
        for plan in runs_plan:
            steps = plan["active_steps"]
            plan["steps_to_disable"] = _filter_stride(steps, n)
            plan["steps_to_keep"] = [s for s in steps if s not in set(plan["steps_to_disable"])]
    elif strategy == "2":
        k = int(Prompt.ask("保留首尾各多少步", default="5"))
        stride = int(Prompt.ask("中间区域步长", default="10"))
        for plan in runs_plan:
            steps = plan["active_steps"]
            plan["steps_to_disable"] = _filter_endpoints_sparse(steps, k, stride)
            plan["steps_to_keep"] = [s for s in steps if s not in set(plan["steps_to_disable"])]
    else:
        k = int(Prompt.ask("尾部保留多少步", default="10"))
        stride = int(Prompt.ask("前面区域步长", default="10"))
        for plan in runs_plan:
            steps = plan["active_steps"]
            plan["steps_to_disable"] = _filter_tail_dense(steps, k, stride)
            plan["steps_to_keep"] = [s for s in steps if s not in set(plan["steps_to_disable"])]

    # 3. 预览
    _show_preview(runs_plan)

    if not any(p["steps_to_disable"] for p in runs_plan):
        console.print("[yellow]过滤后无步需要禁用（步数太少或参数过于宽松）。[/yellow]")
        return

    # 4. 确认并执行
    if not Confirm.ask("\n确定要执行吗？", default=False):
        console.print("[yellow]操作已取消。[/yellow]")
        return

    total_success, total_errors = 0, 0
    for plan in track(runs_plan, description="处理中..."):
        if plan["steps_to_disable"]:
            s, e = _apply_disable(plan["path"], plan["steps_to_disable"])
            total_success += s
            total_errors += e

    console.print(f"\n[green]完成！成功禁用: {total_success}，失败: {total_errors}[/green]")
    console.print("[dim]提示：可随时使用「恢复时间步」操作撤销。[/dim]")


def _workflow_enable(selected_dirs: List[str]):
    """恢复已禁用时间步的交互流程"""
    runs_info = []
    for d in selected_dirs:
        run_dir = Path(d)
        info = _discover_timesteps(run_dir)
        if info["disabled_steps"]:
            runs_info.append({**info, "name": run_dir.name, "path": run_dir})

    if not runs_info:
        console.print("[green]没有已禁用的时间步。[/green]")
        return

    _show_status(runs_info)

    console.print("\n[bold]恢复选项:[/bold]")
    console.print("  [cyan]1[/] - 恢复全部已禁用的时间步")
    console.print("  [cyan]2[/] - 指定要恢复的步")
    choice = Prompt.ask("选择", choices=["1", "2"], default="1")

    steps_to_enable = None
    if choice == "2":
        # 收集所有已禁用的步
        all_disabled = sorted(set(s for r in runs_info for s in r["disabled_steps"]))
        console.print(f"[dim]已禁用的步: {', '.join(map(str, all_disabled[:20]))}{'...' if len(all_disabled) > 20 else ''}[/dim]")
        raw = Prompt.ask("输入要恢复的步号（逗号分隔，支持范围如 0-100）")
        steps_to_enable = _parse_step_input(raw, all_disabled)
        if not steps_to_enable:
            console.print("[yellow]未匹配到有效的步号。[/yellow]")
            return

    if not Confirm.ask("确定要恢复吗？", default=True):
        console.print("[yellow]操作已取消。[/yellow]")
        return

    total_success, total_errors = 0, 0
    for info in track(runs_info, description="恢复中..."):
        s, e = _apply_enable(info["path"], steps_to_enable)
        total_success += s
        total_errors += e

    console.print(f"\n[green]恢复完成！成功: {total_success}，失败: {total_errors}[/green]")


def _parse_step_input(raw: str, available: List[int]) -> List[int]:
    """解析用户输入的步号（逗号分隔，支持范围如 0-100）"""
    result = []
    available_set = set(available)
    for part in raw.split(","):
        part = part.strip()
        if "-" in part:
            try:
                a, b = part.split("-", 1)
                for s in range(int(a), int(b) + 1):
                    if s in available_set:
                        result.append(s)
            except ValueError:
                continue
        else:
            try:
                s = int(part)
                if s in available_set:
                    result.append(s)
            except ValueError:
                continue
    return sorted(set(result))


# ---- 入口 ----

def run_interactive_workflow(selected_dirs: List[str]):
    """step_filter 工具入口"""
    console.print("\n[bold underline]时间步过滤工具 (Step Filter)[/bold underline]")
    console.print("[dim]原理：通过重命名 .h5 -> .h5.disable 来排除时间步，不删除任何文件。[/dim]")
    console.print("[dim]分析框架的 glob 模式会自动跳过 .disable 文件。[/dim]\n")

    console.print("[bold]请选择操作:[/bold]")
    console.print("  [cyan]1[/] - 禁用时间步（过滤）")
    console.print("  [cyan]2[/] - 恢复已禁用的时间步")
    console.print("  [cyan]3[/] - 查看当前状态")
    action = Prompt.ask("选择", choices=["1", "2", "3"], default="1")

    if action == "1":
        _workflow_disable(selected_dirs)
    elif action == "2":
        _workflow_enable(selected_dirs)
    else:
        runs_info = []
        for d in selected_dirs:
            run_dir = Path(d)
            info = _discover_timesteps(run_dir)
            runs_info.append({**info, "name": run_dir.name})
        _show_status(runs_info)
