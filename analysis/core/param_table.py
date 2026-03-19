from collections import Counter
from typing import Dict, Any, List, Tuple, Callable

import matplotlib.pyplot as plt
import numpy as np
import sympy
from matplotlib.axes import Axes
from matplotlib.table import Table as mpl_Table
from sympy.parsing.sympy_parser import parse_expr

from analysis.core.simulation import SimulationRun
from analysis.core.utils import M_E, C, J_PER_MEV
from ..plotting.styles import get_style

# =============================================================================
# 类型别名，增加代码可读性
# =============================================================================

# 定义一个格式化器为一个函数，它接收一个模拟对象并返回一个字符串
Formatter = Callable[[Any], str]
# 参数映射表。键是参数名，值是一个元组，包含(详细格式化器, 简洁格式化器)
ParamMap = Dict[str, Tuple[Formatter, Formatter]]


# =============================================================================
# 生成展示性的物理公式
# =============================================================================

def _get_theoretical_formula(sim: Any) -> str:
    """
    根据模拟类型，返回展示性的物理公式 (LaTeX)，而不是具体的数值展开。
    """
    b_type = getattr(sim, 'B_field_type', 'unknown')

    # 1. 均匀场
    if b_type == 'uniform':
        # 显示 B = (B0, B0, B0)
        val = getattr(sim, 'B_target_rms', 0)
        return rf"$\mathbf{{B}} = ({val}, {val}, {val}) \, \mathrm{{T}}$"

    # 2. 单高斯场
    elif b_type == 'single_gaussian':
        # 显示 B ~ exp(...)
        return (r"$\mathbf{B}(\mathbf{r}) = B_0 \hat{n} \cdot e^{-|\mathbf{r}-\mathbf{r}_0|^2/w^2}$"
                r"$\quad (\hat{n} \text{ random}, |\hat{n}|=1)$")

    # 3. 多高斯叠加 (核心需求)
    elif b_type == 'multi_gaussian':
        n = getattr(sim, 'num_gaussians', 'N')
        # 显示求和公式
        return (rf"$\mathbf{{B}}(\mathbf{{r}}) = \sum_{{i=1}}^{{{n}}} \mathbf{{B}}_i e^{{-|\mathbf{{r}}-\mathbf{{r}}_i|^2/w^2}}$"
                rf"$\quad (|\mathbf{{B}}_i| \equiv B_0, \text{{random dir}})$")

    return r"$\text{Custom Field}$"

# =============================================================================
# 辅助函数：物理公式生成与格式化
# =============================================================================

def _parse_and_latex(srepr_str: str, max_length: int = 800) -> str:
    """
    解析 srepr 并转换为 LaTeX。允许更长的长度，因为我们现在单独展示它。
    """
    try:
        if not srepr_str: return "N/A"
        expr = parse_expr(srepr_str)
        # 即使很复杂，也尝试渲染，因为现在有专门的空间
        lat = sympy.latex(expr)
        # 只有极其巨大的时候才截断
        if len(lat) > max_length:
            return r"\text{公式过长 (Length > " + str(max_length) + r")}"
        return f"${lat}$"
    except Exception:
        return r"\text{解析错误}"


# =============================================================================
# 参数映射表定义 (核心修正)
# =============================================================================

def _get_param_map(run: SimulationRun) -> ParamMap:
    """
    定义参数名称到格式化函数的映射。
    每个条目是一个元组：(用于单表的详细格式化器, 用于对比表的简洁格式化器)
    """
    s = run.sim
    m_e_c2_MeV = (M_E * C ** 2) / J_PER_MEV
    is_3d = hasattr(s, 'NY') and hasattr(s, 'Ly') and s.NY > 1

    # 为每个参数定义一对格式化函数 (详细, 简洁)
    fmt_norm_B = (
        lambda x: f"$\\sqrt{{4 \\mu_0 n_0 k_B T}} \\approx {x.B_norm:.2e} \\, \\text{{T}}$",
        lambda x: f"{x.B_norm:.2e}"
    )
    fmt_norm_J = (
        lambda x: f"$n_0 e c \\approx {x.J_norm:.2e} \\, \\text{{A/m}}^2$",
        lambda x: f"{x.J_norm:.2e}"
    )
    fmt_sigma = (
        lambda x: f"$\\frac{{B_0^2}}{{\\mu_0 n_0 \\varepsilon_{{tot}}}} \\approx {x.sigma:.3f}$",
        lambda x: f"{x.sigma:.3f}"
    )

    def fmt_field_structure_detail(s_obj):
        if s_obj.B_field_type == 'uniform': return "均匀场"
        if s_obj.B_field_type == 'single_gaussian': return "单高斯"
        if s_obj.B_field_type == 'multi_gaussian':
            n = getattr(s_obj, 'num_gaussians', '?')
            return f"多高斯叠加 (N={n})"
        return s_obj.B_field_type

    # 准备映射字典
    mapping: ParamMap = {}

    # --- 归一化与尺度 ---
    mapping["--- 归一化尺度 ---"] = (None, None)
    mapping["热磁场 $B_{th}$ (T)"] = fmt_norm_B
    mapping["热电流 $J_{rel}$ (A/m²)"] = fmt_norm_J
    mapping["趋肤深度 $d_e$ (m)"] = (lambda x: f"${x.d_e:.2e}$", lambda x: f"{x.d_e:.2e}")
    mapping["等离子体周期 $\\omega_{pe}^{-1}$ (s)"] = (lambda x: f"${1.0 / x.w_pe:.2e}$", lambda x: f"{1.0 / x.w_pe:.2e}")

    # --- 核心物理参数 ---
    mapping["--- 等离子体物理 ---"] = (None, None)
    mapping["磁能占比 $\sigma$"] = fmt_sigma
    mapping["相对论温度 $\Theta$"] = (lambda x: f"${x.theta:.3f}$ ({x.T_plasma / 1e3:.1f} keV)", lambda x: f"{x.theta:.3f}")
    mapping["数密度 $n_0$ ($\mathrm{m}^{-3}$)"] = (lambda x: f"${x.n_plasma:.2e}$", lambda x: f"{x.n_plasma:.2e}")

    # --- 磁场模型 ---
    mapping["--- 磁场模型 ---"] = (None, None)
    mapping["模型类型"] = (fmt_field_structure_detail, fmt_field_structure_detail)

    if hasattr(s, 'B_field_type') and 'gaussian' in s.B_field_type:
        mapping["高斯包数量 $N$"] = (lambda x: f"${getattr(x, 'num_gaussians', 1)}$", lambda x: f"{getattr(x, 'num_gaussians', 1)}")
        mapping["相对宽度 $w/L$"] = (lambda x: f"{getattr(x, 'gaussian_width_L_ratio', 'N/A'):.2f}",
                                     lambda x: f"{getattr(x, 'gaussian_width_L_ratio', 'N/A'):.2f}")
        mapping["峰值场强 $B_0$ (T)"] = (lambda x: f"{x.B_target_rms:.2f}", lambda x: f"{x.B_target_rms:.2f}")

    # 使用特殊键名 "EXPR_HIDDEN" 标记它，这样我们在生成主表格数据时可以识别并跳过它
    mapping["EXPR_HIDDEN"] = (_get_theoretical_formula, _get_theoretical_formula)

    # --- 束流与能量 ---
    if hasattr(s, 'beam_fraction') and s.beam_fraction > 0:
        mapping["--- 束流参数 ---"] = (None, None)
        mapping["束流占比 (%)"] = (lambda x: f"{x.beam_fraction * 100:.1f}", lambda x: f"{x.beam_fraction * 100:.1f}")
        mapping["束流归一化动量 $u_z$"] = (lambda x: f"{x.beam_u_drift:.2f} ($\\approx${x.beam_u_drift * m_e_c2_MeV:.1f} MeV/c)",
                                           lambda x: f"{x.beam_u_drift:.2f}")

    # --- 模拟域 ---
    mapping["--- 网格与模拟域 ---"] = (None, None)
    grid_str_detail = f"${s.NX} \\times {s.NY} \\times {s.NZ}$" if is_3d else f"${s.NX} \\times {s.NZ}$"
    grid_str_simple = f"{s.NX}x{s.NY}x{s.NZ}" if is_3d else f"{s.NX}x{s.NZ}"
    size_str_detail = f"${s.LX:.0f}d_e \\times {s.LY:.0f}d_e \\times {s.LZ:.0f}d_e$" if is_3d else f"${s.LX:.0f}d_e \\times {s.LZ:.0f}d_e$"
    size_str_simple = f"{s.LX:.0f}x{s.LY:.0f}x{s.LZ:.0f}" if is_3d else f"{s.LX:.0f}x{s.LZ:.0f}"

    mapping["网格数"] = (lambda _: grid_str_detail, lambda _: grid_str_simple)
    mapping["模拟域尺寸 ($d_e$)"] = (lambda _: size_str_detail, lambda _: size_str_simple)
    mapping["每单元粒子数 (NPPC)"] = (lambda x: f"${x.NPPC}$", lambda x: f"{x.NPPC}")

    return mapping


# =============================================================================
# 单参数表绘制
# =============================================================================

def _create_parameter_table_data(run: SimulationRun) -> Tuple[List[List[str]], str]:
    """
    返回: (表格行数据列表, 独立的公式字符串)
    """
    param_map = _get_param_map(run)
    table_data = []
    expr_str = ""

    for param_name, (formatter, _) in param_map.items():
        # 如果是特殊标记的公式，提取出来但不放入表格
        if param_name == "EXPR_HIDDEN":
            try:
                expr_str = formatter(run.sim)
            except:
                expr_str = r"\text{Error Parsing Expr}"
            continue

        if formatter is None:
            table_data.append([param_name, ''])
            continue
        try:
            value_str = formatter(run.sim)
        except Exception:
            value_str = "Error"
        table_data.append([f"  {param_name}", value_str])

    return table_data, expr_str


def plot_parameter_table(ax: Axes, run: SimulationRun) -> mpl_Table:
    """
    绘制单个模拟的参数表。
    特点：表格只包含基本信息，底部单独显示复杂的合成公式。
    """
    style = get_style()
    ax.axis('off')

    # 获取数据
    table_data, expr_str = _create_parameter_table_data(run)

    # 1. 绘制主表格 (只有参数名和简单的值)
    table = ax.table(
        cellText=table_data, colLabels=['参数', '值'], loc='upper center',
        cellLoc='left', colWidths=[0.35, 0.65])

    table.auto_set_font_size(False)
    table.set_fontsize(style.font_size_base)

    # 行高设置为 3.0，稍微大一点，舒适
    table.scale(1, 3.0)

    # 美化表格
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor('lightgray')
        if row == 0:
            cell.set_text_props(weight='bold', ha='center')
            cell.set_facecolor('#B0C4DE')
        else:
            if "---" in table_data[row - 1][0]:
                cell.set_text_props(weight='bold', ha='center')
                cell.set_facecolor('#E0E0E0')
                if col == 1: cell.set_text_props(text='')
            if col == 0: cell.set_text_props(ha='left')
            if row % 2 == 0: cell.set_facecolor('#F5F5F5')

    # 2. 在表格下方单独绘制具体的合成公式
    if expr_str:
        display_text = (
            "磁场分布公式: " + expr_str
        )

        ax.text(0.0, -0.50,  # 放在表格下方 (y < 0)
                display_text,
                transform=ax.transAxes,
                fontsize=style.font_size_base + 2,  # 公式稍微大一点
                ha='left', va='top',
                bbox=dict(boxstyle="round,pad=0.6", fc='#F0F8FF', ec="#4682B4", alpha=1.0, lw=1.5))

        return table


# =============================================================================
# 对比参数表绘制
# =============================================================================

def _create_comparison_table_data(runs: List[SimulationRun]) -> Tuple[List[str], List[List[str]], List[List[bool]], Dict[str, str]]:
    """
    返回: (表头, 表格数据, 高亮掩码, {run_name: expr_str})
    """
    if not runs: return [], [], [], {}
    param_map = _get_param_map(runs[0])
    header = ["参数"] + [run.name for run in runs]
    table_data, highlight_mask = [], []
    expr_dict = {}

    # 初始化公式字典
    for run in runs:
        expr_dict[run.name] = r"\text{N/A}"

    for param_name, (_, formatter) in param_map.items():
        # 提取公式
        if param_name == "EXPR_HIDDEN":
            for run in runs:
                try:
                    # 使用详细格式化器 (tuple index 0) 来获取完整公式
                    detailed_fmt = param_map[param_name][0]
                    expr_dict[run.name] = detailed_fmt(run.sim)
                except:
                    pass
            continue

        if formatter is None:
            table_data.append([param_name] + [''] * len(runs))
            highlight_mask.append([False] * (len(runs) + 1))
            continue

        row_values = []
        for run in runs:
            try:
                value_str = formatter(run.sim)
            except Exception:
                value_str = "Error"
            row_values.append(value_str)

        if "IGNORE" in row_values: continue

        # 高亮逻辑
        if len(row_values) > 1:
            counts = Counter(row_values)
            majority_value, _ = counts.most_common(1)[0]
            row_mask = [val != majority_value for val in row_values]
        else:
            row_mask = [False] * len(row_values)

        table_data.append([f"  {param_name}"] + row_values)
        highlight_mask.append([False] + row_mask)

    return header, table_data, highlight_mask, expr_dict


def plot_comparison_parameter_table(ax: Axes, runs: List[SimulationRun]) -> mpl_Table:
    """
    绘制多模拟对比表。
    公式在下方按 Run 分列显示。
    """
    style = get_style()
    ax.axis('off')
    header, table_data, highlight_mask, expr_dict = _create_comparison_table_data(runs)
    if not table_data: return None

    num_runs = len(runs)
    col_widths = [0.40] + [0.60 / num_runs] * num_runs if num_runs > 0 else [0.4, 0.6]

    # 1. 绘制主表
    table = ax.table(
        cellText=table_data, colLabels=header, loc='upper center',
        cellLoc='center', colWidths=col_widths)

    table.auto_set_font_size(False)
    table.set_fontsize(style.font_size_legend)

    # 行高加大
    table.scale(1, 3.0)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor('lightgray')
        if row == 0:
            cell.set_text_props(weight='bold', ha='center')
            cell.set_facecolor('#B0C4DE')
            continue
        if col == 0: cell.set_text_props(ha='left')
        if "---" in table_data[row - 1][0]:
            cell.set_text_props(weight='bold', ha='center')
            cell.set_facecolor('#E0E0E0')
            if col > 0: cell.set_text_props(text='')
            continue
        if (row - 1) % 2 == 0: cell.set_facecolor('#F5F5F5')
        if highlight_mask[row - 1][col]:
            cell.set_facecolor('#FFDDC1')
            cell.set_text_props(weight='bold')

    # 底部公式展示
    combined_text = "磁场模型公式:\n"

    # 简单去重：如果所有 run 的公式都一样，只显示一次
    unique_exprs = {}
    for run in runs:
        name = run.name
        expr = expr_dict.get(name, "")
        if expr not in unique_exprs:
            unique_exprs[expr] = []
        unique_exprs[expr].append(name)

    # 如果只有一种公式，显示该公式即可
    if len(unique_exprs) == 1:
        expr = list(unique_exprs.keys())[0]
        combined_text += expr
    else:
        # 如果有多种公式，按组显示
        for expr, names in unique_exprs.items():
            # names_str = ", ".join(names)
            # combined_text += f"• [{names_str}]: {expr}\n"
            # 为了美观，这里还是逐个显示比较稳妥，避免名字太长换行
            for name in names:
                combined_text += f"• {name}:\n   {expr}\n"

    ax.text(0.0, -0.5,
            combined_text,
            transform=ax.transAxes,
            fontsize=style.font_size_legend + 1,
            ha='left', va='top',
            linespacing=1.8,
            bbox=dict(boxstyle="round,pad=0.6", fc='#F0F8FF', ec="#4682B4", alpha=1.0, lw=1.5))

    return table