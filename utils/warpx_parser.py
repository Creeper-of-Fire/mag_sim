import re

# WarpX verbose 输出模式
_RE_STEP_ENDS = re.compile(r"STEP\s+(\d+)\s+ends\.")
_RE_STEP_STARTS = re.compile(r"STEP\s+(\d+)\s+starts")
_RE_OPENPMD_PATH = re.compile(r"^\s+/.*?/diags/")
_RE_EVOLVE_TIME = re.compile(
    r"Evolve time\s*=\s*([\d.eE+-]+)\s*s;"
    r"\s*This step\s*=\s*([\d.eE+-]+)\s*s;"
    r"\s*Avg\.\s*per\s*step\s*=\s*([\d.eE+-]+)\s*s"
)

# simulation.py 的 _print_summary() 输出
_RE_TOTAL_STEPS = re.compile(r"总步数\s*=\s*(\d+)")


def parse_step_line(line: str) -> int | None:
    """从 'STEP N ends.' 行提取当前步数。"""
    m = _RE_STEP_ENDS.search(line)
    return int(m.group(1)) if m else None


def is_step_progress_line(line: str) -> bool:
    """判断是否为 WarpX 冗余进度行（STEP / Evolve / re-sorting / Writing openPMD 及其路径续行）。"""
    if _RE_STEP_STARTS.search(line) or _RE_STEP_ENDS.search(line) or _RE_EVOLVE_TIME.search(line):
        return True
    if _RE_OPENPMD_PATH.search(line):
        return True
    return "re-sorting particles" in line or "Writing openPMD file" in line


def parse_evolve_line(line: str) -> tuple[float, float, float] | None:
    """从 Evolve 行提取 (elapsed, this_step, avg_per_step) 秒数。"""
    m = _RE_EVOLVE_TIME.search(line)
    if m:
        return float(m.group(1)), float(m.group(2)), float(m.group(3))
    return None


def parse_total_steps(line: str) -> int | None:
    """从 simulation.py 的 '总步数 = N' 提取总步数。"""
    m = _RE_TOTAL_STEPS.search(line)
    return int(m.group(1)) if m else None
