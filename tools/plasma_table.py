# -*- coding: utf-8 -*-

# =============================================================================
# 导入依赖
# =============================================================================
import re
from dataclasses import dataclass
from functools import cached_property

import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import c, e, epsilon_0, hbar, k, m_e, mu_0
# 重命名 sigma 防止与磁化参数混淆
from scipy.constants import sigma as SIGMA_SB

from analysis.utils import setup_chinese_font

# 初始化中文字体
setup_chinese_font()


# =============================================================================
# 核心上下文 (复用并增强)
# =============================================================================
class PlasmaScenario:
    """
    等离子体场景上下文，负责物理计算。
    """

    def __init__(self, name: str, n_e: float, T_eV: float, NX: int, LX: float, LT: float, DT: float, dims: int = 3):
        self.name = name
        self.n_e = n_e
        self.T_eV = T_eV
        self.NX = NX
        self.LX = LX  # in d_e
        self.LT = LT  # in 1/w_pe
        self.DT = DT  # in 1/w_pe
        self.dims = dims

    # --- 基础热力学 ---
    @cached_property
    def T_J(self) -> float:
        return self.T_eV * e

    @cached_property
    def T_K(self) -> float:
        return self.T_J / k

    @cached_property
    def lorentz_gamma(self) -> float:
        return 1 + self.T_J / (m_e * c ** 2)

    @cached_property
    def v_thermal(self) -> float:
        """相对论热速度"""
        return c * np.sqrt(1 - 1 / self.lorentz_gamma ** 2)

    @cached_property
    def total_energy_density(self) -> float:
        """等离子体总能量密度 (用于计算 Sigma)"""
        return 2 * self.n_e * (m_e * c ** 2 + 3 * self.T_J)

    # --- 基础频率与长度 ---
    @cached_property
    def omega_pe(self) -> float:
        return np.sqrt(self.n_e * e ** 2 / (epsilon_0 * m_e))

    @cached_property
    def skin_depth_e(self) -> float:
        """趋肤深度 d_e (归一化基准长度)"""
        return c / self.omega_pe

    @cached_property
    def debye_length(self) -> float:
        return np.sqrt(epsilon_0 * self.T_J / (self.n_e * e ** 2))

    # --- 碰撞与辐射 ---
    @cached_property
    def mean_free_path(self) -> float:
        coulomb_log = 15
        v_th = self.v_thermal
        if v_th == 0: return 0.0
        (self.n_e * e ** 4 * coulomb_log) / (4 * np.pi * epsilon_0 ** 2 * m_e ** 2 * v_th ** 3)
        nu_coll = (self.n_e * e ** 4 * coulomb_log) / (4 * np.pi * epsilon_0 ** 2 * m_e ** 2 * v_th ** 3)
        return v_th / nu_coll if nu_coll > 0 else np.inf

    @cached_property
    def tau_collision(self) -> float:
        """平均碰撞时间 (s)"""
        v_th = self.v_thermal
        if v_th == 0: return np.inf
        return self.mean_free_path / v_th

    @cached_property
    def tau_cooling(self) -> float:
        """辐射冷却时间 (s)"""
        U_rad = (4 * SIGMA_SB / c) * self.T_K ** 4
        sigma_Thomson = 6.6524e-29
        # 冷却功率 dE/dt
        power = (4 / 3) * sigma_Thomson * c * U_rad * self.lorentz_gamma ** 2
        total_energy = self.lorentz_gamma * m_e * c ** 2
        return total_energy / power if power > 0 else np.inf

    # --- 模拟参数转换 ---
    @cached_property
    def dx_m(self) -> float:
        return (self.LX * self.skin_depth_e) / self.NX

    @cached_property
    def dt_s(self) -> float:
        return self.DT / self.omega_pe

    @cached_property
    def box_length_m(self) -> float:
        return self.LX * self.skin_depth_e


# =============================================================================
# 表格生成器
# =============================================================================
class TableGenerator:
    """
    生成详细的参数表格，支持LaTeX公式渲染。
    """

    def __init__(self):
        # 设置绘图参数以支持更好的LaTeX显示
        plt.rcParams.update({
            'text.usetex': False,  # 使用内置mathtext引擎，不需要外部latex安装
            'mathtext.fontset': 'cm',  # Computer Modern 字体
            'font.family': 'sans-serif'
        })

    def calculate_magnetic_params(self, s: PlasmaScenario, target_sigma: float):
        """
        根据目标 Sigma 计算磁场和回旋半径
        Sigma = B^2 / (2 * mu_0 * U_p)
        """
        U_p = s.total_energy_density
        B_field = np.sqrt(target_sigma * 2 * mu_0 * U_p)

        # 相对论回旋半径 rho = (gamma * m * v) / (q * B)
        if B_field > 0:
            rho_e = (s.lorentz_gamma * m_e * s.v_thermal) / (e * B_field)
            freq_ce = (e * B_field) / (s.lorentz_gamma * m_e)  # 回旋频率
        else:
            rho_e = np.inf
            freq_ce = 0.0

        return B_field, rho_e, freq_ce
    def generate(self, scenario: PlasmaScenario, target_sigma: float):
        """
        为单个场景生成详细表格图
        """
        # 1. 准备数据
        de = scenario.skin_depth_e
        wpe = scenario.omega_pe
        t_pe = 1.0 / wpe  # 等离子体时间周期基准

        # 计算磁学量
        B_val, rho_val, wce_val = self.calculate_magnetic_params(scenario, target_sigma)

        # 定义表格行数据结构
        # 格式: (Category, Name, Symbol/Formula, Value(SI), Value(Norm))
        data_rows = []

        # --- 基础参量 ---
        data_rows.append(["输入", "电子密度", r"$n_e$",
                          f"{scenario.n_e:.2e} m$^{{-3}}$", "-"])
        data_rows.append(["输入", "电子温度", r"$T_e$",
                          f"{scenario.T_eV / 1e3:.1f} keV", f"$\gamma$={scenario.lorentz_gamma:.2f}"])
        data_rows.append(["输入", "目标磁化率", r"$\sigma = \frac{B^2}{2\mu_0 U_p}$",
                          "-", f"{target_sigma}"])

        # --- 空间尺度 (归一化至 d_e) ---
        data_rows.append(["长度", "趋肤深度 (参考)", r"$d_e = c/\omega_{pe}$",
                          f"{de:.2e} m", "1.0"])

        data_rows.append(["长度", "德拜长度", r"$\lambda_D = v_{th}/\omega_{pe}$",
                          f"{scenario.debye_length:.2e} m",
                          f"{scenario.debye_length / de:.2e} $d_e$"])

        data_rows.append(["长度", "回旋半径", r"$\rho_e = \frac{\gamma m v}{e B}$",
                          f"{rho_val:.2e} m",
                          f"{rho_val / de:.2e} $d_e$"])

        data_rows.append(["长度", "平均自由程", r"$\lambda_{mfp} \approx v_{th}/\nu_{ei}$",
                          f"{scenario.mean_free_path:.2e} m",
                          f"{scenario.mean_free_path / de:.1e} $d_e$"])

        # --- 时间尺度 (归一化至 1/w_pe) ---
        data_rows.append(["时间", "等离子体周期 (参考)", r"$\omega_{pe}^{-1}$",
                          f"{t_pe:.2e} s", "1.0"])

        data_rows.append(["时间", "回旋周期", r"$\omega_{ce}^{-1} = (\frac{eB}{\gamma m})^{-1}$",
                          f"{(1 / wce_val if wce_val > 0 else np.inf):.2e} s",
                          f"{(wpe / wce_val if wce_val > 0 else np.inf):.2f} $\omega_{{pe}}^{{-1}}$"])

        data_rows.append(["时间", "碰撞时间", r"$\tau_{coll} = 1/\nu_{ei}$",
                          f"{scenario.tau_collision:.2e} s",
                          f"{scenario.tau_collision / t_pe:.1e} $\omega_{{pe}}^{{-1}}$"])

        data_rows.append(["时间", "冷却时间", r"$\tau_{cool} \propto 1/T^4$",
                          f"{scenario.tau_cooling:.2e} s",
                          f"{scenario.tau_cooling / t_pe:.1e} $\omega_{{pe}}^{{-1}}$"])

        # --- 模拟设置 ---
        data_rows.append(["模拟", "盒子尺寸 (L)", r"$L_{box}$",
                          f"{scenario.box_length_m:.2e} m",
                          f"{scenario.LX} $d_e$"])

        data_rows.append(["模拟", "网格步长 (dx)", r"$\Delta x$",
                          f"{scenario.dx_m:.2e} m",
                          f"{scenario.dx_m / de:.3f} $d_e$"])

        data_rows.append(["模拟", "时间步长 (dt)", r"$\Delta t$",
                          f"{scenario.dt_s:.2e} s",
                          f"{scenario.DT} $\omega_{{pe}}^{{-1}}$"])

        data_rows.append(["结果", "推导磁场", r"$B_{target}$",
                          f"{B_val:.2f} T", "-"])

        # 2. 开始绘图
        fig_height = len(data_rows) * 0.6 + 2
        fig, ax = plt.subplots(figsize=(10, fig_height))
        ax.axis('off')

        # 标题
        title_text = f"模拟参数表: {scenario.name}"
        ax.set_title(title_text, fontsize=16, weight='bold', pad=20)

        # 构建表格内容
        cell_text = []
        for row in data_rows:
            # 移除第一列 Category，用于颜色分组或忽略
            cell_text.append(row[1:])

        col_labels = ["参数", "公式 / 符号", "国际单位制数值", "归一化数值"]

        # 创建表格
        the_table = ax.table(cellText=cell_text,
                             colLabels=col_labels,
                             loc='center',
                             cellLoc='center',
                             colWidths=[0.2, 0.3, 0.25, 0.25])

        the_table.auto_set_font_size(False)
        the_table.set_fontsize(11)
        the_table.scale(1, 2.0)  # 拉伸行高

        # 美化表格样式
        for (row, col), cell in the_table.get_celld().items():
            cell.set_edgecolor('black')
            if row == 0:
                cell.set_text_props(weight='bold', color='white')
                cell.set_facecolor('#40466e')  # Header color
                cell.set_fontsize(12)
            else:
                # 斑马纹或根据类别着色
                category = data_rows[row - 1][0]
                if category == "输入":
                    cell.set_facecolor('#e3f2fd')
                elif category == "长度":
                    cell.set_facecolor('#f1f8e9')
                elif category == "时间":
                    cell.set_facecolor('#fff3e0')
                elif category == "模拟":
                    cell.set_facecolor('#f3e5f5')
                else:
                    cell.set_facecolor('white')

        plt.tight_layout()

        # 保存
        safe_name = re.sub(r'[\\/*?:"<>|()]', "", scenario.name).replace(' ', '_')
        filename = f"table_{safe_name}.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"[Generated] {filename}")
        plt.close(fig)  # 释放内存


# =============================================================================
# 主程序
# =============================================================================
if __name__ == "__main__":

    # 1. 配置参数扫描 (与之前逻辑一致)
    plasma_regimes = [
        {"name": "氘瓶颈", "n_e": 7.28e33, "T_eV": 84480.0, "NX_scale": 1.0, "DT": 0.2},
        {"name": "中温情况", "n_e": 3.598e32, "T_eV": 58650.0, "NX_scale": 1.25, "DT": 0.15},
        {"name": "低温情况", "n_e": 8.43e23, "T_eV": 12230.0, "NX_scale": 2.75, "DT": 0.075},
    ]

    # 简化：只取一个典型Box配置演示，避免生成太多图片
    # 如果你需要全部，可以恢复列表
    box_config = {"LX": 100.0, "NX_base": 256, "LT": 100.0}

    target_sigmas = [0.01, 0.1]  # 关注的磁化率

    generator = TableGenerator()

    print("--- 开始生成参数表格 ---")

    for regime in plasma_regimes:
        for sigma in target_sigmas:
            # 构建场景名称
            suffix = f"_{int(regime['T_eV'] / 1000)}keV_Sigma{sigma}"
            name = f"{regime['name']}{suffix}"

            # 实例化
            scenario = PlasmaScenario(
                name=name,
                n_e=regime['n_e'],
                T_eV=regime['T_eV'],
                NX=int(box_config['NX_base'] * regime['NX_scale']),
                LX=box_config['LX'],
                LT=box_config['LT'],
                DT=regime['DT']
            )

            # 生成表格
            generator.generate(scenario, sigma)

    print("--- 全部完成 ---")