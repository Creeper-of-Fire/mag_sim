# analysis/core/params_display_names.py

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Union, List


@dataclass
class ParamInfo:
    """
    存储参数的显示信息，包括中文名、物理符号和单位。
    """
    name_cn: str
    symbol: str
    unit: str
    description: str = ""

    @property
    def ToLabel(self):
        if self.unit:
            return f"{self.name_cn} ({self.symbol}) [{self.unit}]"
        else:
            return f"{self.name_cn} ({self.symbol})"

    def format_axis(self, values: Union[np.ndarray, List[float]]) -> Tuple[np.ndarray, str]:
        """
        输入原始数值，输出：
        1. 缩放后的数值（用于绘图）
        2. 格式化后的坐标轴标签（包含倍率，如 10^-15）
        """
        vals = np.array(values, dtype=float)
        if vals.size == 0 or np.all(vals == 0):
            return vals, self.ToLabel

        # 自动检测数量级 (10^k)
        max_val = np.max(np.abs(vals))
        exponent = int(np.floor(np.log10(max_val)))

        # 如果数量级在常用范围之外 (比如太小 10^-3 以下，或太大 10^4 以上)
        # 或者单位本身就是秒 's'，且值很小
        if abs(exponent) >= 3:
            scale = 10 ** exponent
            scaled_vals = vals / scale
            # 构造带倍率的标签，例如：时间 (t) [10^-15 s]
            if self.unit:
                new_label = f"{self.name_cn} ({self.symbol}) [$10^{{{exponent}}}$ {self.unit}]"
            else:
                new_label = f"{self.name_cn} ({self.symbol}) [$10^{{{exponent}}}$]"
            return scaled_vals, new_label

        return vals, self.ToLabel


# 创建一个默认的未知参数信息
def UNKNOWN_PARAM(key: str):
    return ParamInfo(key, "?", "", "未在映射表中定义的参数")


# --- 参数显示映射表 ---
# 将 config.py, simulation.py, slicer.py 中所有可能作为变量的参数都在这里注册
PARAM_DISPLAY_MAP = {
    # 来自 slicer.py 的虚拟参数
    "slice_step": ParamInfo("时间切片步数", r'$t_{step}$', "steps", "虚拟时间切片对应的模拟步数"),
    "current_time": ParamInfo("时间", r'$t$', "s", "从模拟开始的物理时间"),

    # 来自 config.py 的核心物理参数
    "run_id": ParamInfo("运行ID", r'$ID_{run}$', "", "用于区分统计性重复运行的标识"),
    "target_sigma": ParamInfo("磁能占比", r'$\sigma$', "", "磁能密度与粒子热焓密度之比"),
    "B0": ParamInfo("背景磁场", r'$B_0$', "T", "初始背景磁场的绝对强度"),
    "n_plasma": ParamInfo("等离子体数密度", r'$n_e$', r'm$^{-3}$', "电子或正电子的数密度"),
    "T_plasma_eV": ParamInfo("等离子体温度", r'$T_e$', "eV", "等离子体背景温度"),
    "beam_fraction": ParamInfo("束流粒子占比", r'$f_{beam}$', "", "非热束流粒子占总数的比例"),
    "beam_energy_eV": ParamInfo("束流动能", r'$E_{beam}$', "eV", "非热束流粒子的动能"),

    # 来自 config.py 的模拟设置参数
    "LX": ParamInfo("模拟域长度X", r'$L_x$', r'$d_e$', "模拟域X方向长度（电子趋肤深度归一化）"),
    "LT": ParamInfo("模拟总时长", r'$T_{sim}$', r'$\omega_{pe}^{-1}$', "模拟总时长（等离子体周期归一化）"),
    "NX": ParamInfo("网格数X", r'$N_x$', "", "X方向的网格数量"),
    "NPPC": ParamInfo("每单元粒子数", "NPPC", "", "每个单元的宏粒子数"),
}


def get_param_display(key: str) -> ParamInfo:
    """
    安全地从映射表中获取参数的显示信息。
    如果找不到，则返回原始信息。
    """
    return PARAM_DISPLAY_MAP.get(key, UNKNOWN_PARAM(key))
