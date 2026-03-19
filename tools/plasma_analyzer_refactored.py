# -*- coding: utf-8 -*-

# =============================================================================
# 导入依赖
# =============================================================================
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from functools import cached_property

import numpy as np
from scipy.constants import c, e, epsilon_0, hbar, k, m_e, mu_0, sigma, alpha

from analysis.utils import setup_chinese_font

setup_chinese_font()


# =============================================================================
# 核心架构: 第1部分 - 上下文 (Context)
# 描述一个完整的物理和模拟场景
# =============================================================================
class PlasmaScenario:
    """
    数据上下文类 (Context)。
    作为单一数据源，存储所有输入参数并提供所有衍生基础物理量的计算。
    使用 @cached_property 确保每个值只计算一次。
    """

    def __init__(self, name: str, n_e: float, T_eV: float, NX: int, LX: float, LT: float, DT: float, dims: int = 1):
        # 场景标识
        self.name = name
        # 物理输入
        self.n_e = n_e  # 电子数密度 (m^-3)
        self.T_eV = T_eV  # 电子温度 (eV)
        # 无量纲模拟输入
        self.NX = NX  # 空间网格数
        self.LX = LX  # 模拟盒尺寸 (in d_e)
        self.LT = LT  # 模拟总时长 (in 1/w_pe)
        self.DT = DT  # 模拟时间步长 (in 1/w_pe)
        self.dims = dims  # 模拟维度

    # --- 基础热力学属性 ---
    @cached_property
    def T_J(self) -> float:
        """温度 (焦耳)"""
        return self.T_eV * e

    @cached_property
    def T_K(self) -> float:
        """温度 (开尔文)"""
        return self.T_J / k

    @cached_property
    def lorentz_gamma(self) -> float:
        """热洛伦兹因子"""
        return 1 + self.T_J / (m_e * c ** 2)

    @cached_property
    def v_thermal(self) -> float:
        """热速度 (相对论)"""
        return c * np.sqrt(1 - 1 / self.lorentz_gamma ** 2)

    @cached_property
    def total_energy_density(self) -> float:
        """
        等离子体总能量密度 U_p (J/m^3)。
        用于计算磁化参数 Sigma = B^2 / (2 * mu_0 * U_p)。
        假设包含正反粒子对或电子+离子背景: 2 * n_e * (rest_mass + thermal)
        """
        # 注意: 这里系数取2是假设正负电荷对等的能量贡献 (如正负电子对或电子-离子)
        # 且 3 * T_J 是相对论热能的一个近似
        return 2 * self.n_e * (m_e * c ** 2 + 3 * self.T_J)

    # --- 能耗/停止时间计算 (Stopping Time) ---
    @cached_property
    def coulomb_log(self) -> float: return 15.0

    @cached_property
    def collision_time_s(self) -> float:
        """经典的动量散射碰撞时间 (Spitzer)"""
        v_th = self.v_thermal
        if v_th == 0: return np.inf
        nu_coll = (self.n_e * e ** 4 * self.coulomb_log) / (4 * np.pi * epsilon_0 ** 2 * m_e ** 2 * v_th ** 3)
        return 1.0 / nu_coll if nu_coll > 0 else np.inf

    def stopping_time_A19(self, E_eV: float) -> float:
        """
        严格根据物理评论D 第37卷 第12期 Eq. (A19) 推导的能损(停止)时间计算。
        适用于 T < m_e 且粒子的 beta > beta_th (快粒子近似)。
        dE/dt = 4 * pi * alpha^2 * (hbar * c)^2 * Lambda * n_e / (beta * c * m_e)
        tau = E / (dE/dt)
        """
        E_J = E_eV * e
        gamma = 1 + E_J / (m_e * c ** 2)
        if gamma <= 1: return np.inf
        beta = np.sqrt(1 - 1 / gamma ** 2)

        # SI 单位系下严格转换的 Eq. (A19)
        dE_dt = (4 * np.pi * alpha ** 2 * (hbar * c) ** 2 * self.coulomb_log * self.n_e) / (beta * c * m_e)
        return E_J / dE_dt

    # --- 等离子体特征尺度 ---
    @cached_property
    def omega_pe(self) -> float:
        """电子等离子体频率 (rad/s)"""
        return np.sqrt(self.n_e * e ** 2 / (epsilon_0 * m_e))

    @cached_property
    def skin_depth_e(self) -> float:
        """电子趋肤深度 (m)"""
        return c / self.omega_pe

    @cached_property
    def debye_length(self) -> float:
        """德拜长度 (m)"""
        return np.sqrt(epsilon_0 * self.T_J / (self.n_e * e ** 2))

    @cached_property
    def mean_free_path(self) -> float:
        """电子平均自由程 (m) - 估算值"""
        # 使用强碰撞公式估算 (Coulomb Log ~ 15)
        coulomb_log = 15
        v_th = self.v_thermal
        if v_th == 0: return 0.0

        # 碰撞频率 nu_ei
        nu_coll = (self.n_e * e ** 4 * coulomb_log) / (4 * np.pi * epsilon_0 ** 2 * m_e ** 2 * v_th ** 3)
        return v_th / nu_coll if nu_coll > 0 else np.inf

    @cached_property
    def box_length_m(self) -> float:
        """模拟盒物理总长 (m)"""
        return self.LX * self.skin_depth_e

    # --- 模拟参数的物理量转换 ---
    @cached_property
    def dx_m(self) -> float:
        """物理网格尺寸 (m)"""
        return (self.LX * self.skin_depth_e) / self.NX

    @cached_property
    def dt_s(self) -> float:
        """物理时间步长 (s)"""
        return self.DT / self.omega_pe

    @cached_property
    def total_sim_time_s(self) -> float:
        """物理模拟总时长 (s)"""
        return self.LT / self.omega_pe


# =============================================================================
# 核心架构: 第2部分 - 验证器框架 (Validator Framework)
# 定义验证的标准流程和数据结构
# =============================================================================
class ValidationStatus(Enum):
    """验证结果的状态枚举"""
    SUCCESS = auto()
    WARNING = auto()
    FAILURE = auto()
    INFO = auto()


@dataclass
class ValidationResult:
    """
    标准化的验证结果数据类。
    所有验证器都必须返回这个类型的对象。
    """
    validator_name: str
    status: ValidationStatus
    title: str
    message: str
    formula: str = ""


class Validator(ABC):
    """验证器抽象基类 (接口)"""

    @abstractmethod
    def validate(self, scenario: PlasmaScenario) -> list[ValidationResult]:
        """
        对给定的场景执行验证。
        Args:
            scenario (PlasmaScenario): 包含所有数据的上下文对象。
        Returns:
            list[ValidationResult]: 一个或多个验证结果。
        """
        pass


# =============================================================================
# 核心架构: 第3部分 - 具体验证器实现 (Concrete Validators)
# 每个类实现一个独立的物理检查
# =============================================================================
class TimescaleValidatorV2(Validator):
    """
    检查碰撞和能量损失时间尺度。
    无碰撞 PIC 模拟要求：碰撞时间和能耗时间都应远大于模拟总时长。
    """

    def validate(self, scenario: PlasmaScenario) -> list[ValidationResult]:
        t_sim = scenario.total_sim_time_s
        results = []

        # 定义需要检查的时间尺度及其名称
        check_list = [
            ("动量散射时间", scenario.collision_time_s, "描述整体偏转/动量交换频率"),
            ("主体能损时间 (T_e)", scenario.stopping_time_A19(scenario.T_eV), "Eq.(A19): 热粒子能量弛豫时间"),
            ("高能尾部能损时间 (2T_e)", scenario.stopping_time_A19(scenario.T_eV * 2), "Eq.(A19): 快粒子能量弛豫时间"),
            ("高能尾部能损时间 (10T_e)", scenario.stopping_time_A19(scenario.T_eV * 10), "Eq.(A19): 快粒子能量弛豫时间")
        ]

        for title, tau, desc in check_list:
            ratio = tau / t_sim if t_sim > 0 else np.inf

            # 判定标准：通常大于 10 倍模拟时长认为可以忽略该效应（无碰撞近似成立）
            if ratio > 10:
                status = ValidationStatus.SUCCESS
            elif ratio > 1:
                status = ValidationStatus.WARNING
            else:
                status = ValidationStatus.FAILURE  # 物理过程在模拟时间内发生，不可忽略

            # 格式化消息：同时显示比例和物理秒数
            # 使用科学计数法，因为 BBN 场景下的时间尺度可能在 10^-20 秒量级
            msg = f"Ratio={ratio:.2e} | 物理时长={tau:.2e} s"

            results.append(ValidationResult(
                validator_name="TimescaleCheck",
                status=status,
                title=title,
                message=msg,
                formula=desc
            ))

        # 4. 辐射冷却时间 (单独算，因为涉及公式不同)
        U_rad = (4 * sigma / c) * scenario.T_K ** 4
        sigma_T = 6.6524e-29
        dE_dt_cool = (4 / 3) * sigma_T * c * U_rad * scenario.lorentz_gamma ** 2
        tau_cool = (scenario.lorentz_gamma * m_e * c ** 2) / dE_dt_cool if dE_dt_cool > 0 else np.inf

        cool_ratio = tau_cool / t_sim
        msg_cool = f"Ratio={cool_ratio:.2e} | 物理时长={tau_cool:.2e} s"

        results.append(ValidationResult(
            "TimescaleCheck",
            ValidationStatus.SUCCESS if cool_ratio > 10 else ValidationStatus.WARNING,
            "辐射冷却时间",
            msg_cool,
            "评估康普顿/同步辐射损耗时间尺度"
        ))

        return results


class RelativisticCollisionAnalyzer:
    def __init__(self, n_e: float, T_eV: float, coulomb_log: float = 15.0):
        self.n_e = n_e
        self.T_eV = T_eV
        self.T_J = T_eV * e
        self.coulomb_log = coulomb_log

        # 1. 相对论温度参数 z = m_e c^2 / T_b
        self.z = (m_e * c ** 2) / self.T_J

        # 2. 预计算贝塞尔函数 K_0(z), K_1(z), K_2(z)
        from scipy.special import kve as bessel_kve

        self.K0 = bessel_kve(0, self.z)
        self.K1 = bessel_kve(1, self.z)
        self.K2 = bessel_kve(2, self.z)

        # 3. 碰撞强度常数 \Gamma_{ab} (对于 e-e 碰撞)
        # 论文定义: \Gamma_{ab} = n_b q_a^2 q_b^2 \ln\Lambda / (4 \pi \epsilon_0^2 m_a^2)
        self.Gamma_ee = (self.n_e * e ** 4 * self.coulomb_log) / (4 * np.pi * epsilon_0 ** 2 * m_e ** 2)

        # 4. 背景热速度参数 u_{tb}^2 = T_b / m_b
        self.u_tb_sq = self.T_J / m_e

    def evaluate_relaxation(self, E_test_eV: float):
        """
        严格根据 Braams & Karney (1987) 计算 D_uu, F_u 和弛豫时间。
        E_test_eV: 测试粒子的动能 (eV)
        """
        E_test_J = E_test_eV * e

        # 粒子的相对论运动学参数
        gamma = 1.0 + E_test_J / (m_e * c ** 2)
        if gamma <= 1.0:
            return float('inf'), 0.0, 0.0

        v = c * np.sqrt(1.0 - 1.0 / gamma ** 2)
        u = gamma * v  # 动量/质量比 (论文中的变量 u)

        # ==========================================
        # 严格执行文献公式计算 D_uu
        # D_{uu} = \Gamma_{ab} * (K_1/K_2) * (u_{tb}^2 / v^3) * [1 - (K_0/K_1) * (u_{tb}^2 / \gamma^2 c^2)]
        # ==========================================
        term_bessel_1 = self.K1 / self.K2
        term_bessel_2 = self.K0 / self.K1

        bracket = 1.0 - term_bessel_2 * (self.u_tb_sq / (gamma ** 2 * c ** 2))

        D_uu = self.Gamma_ee * term_bessel_1 * (self.u_tb_sq / v ** 3) * bracket

        # ==========================================
        # 严格执行文献公式计算 F_u (动摩擦力)
        # F_u = - (m_a v / T_b) * D_{uu}
        # ==========================================
        F_u = - (m_e * v / self.T_J) * D_uu

        # ==========================================
        # 计算弛豫时间 \tau
        # \tau = u / |F_u| = (\gamma v) / |F_u|
        # ==========================================
        tau = u / abs(F_u) if F_u != 0 else float('inf')

        return {
            "tau_s": tau,
            "D_uu": D_uu,
            "F_u": F_u,
            "gamma": gamma,
            "v_over_c": v / c
        }

class QuantumDegeneracyValidator(Validator):
    """检查系统是否处于经典状态，避免量子简并。"""

    def validate(self, scenario: PlasmaScenario) -> list[ValidationResult]:
        E_F = (hbar ** 2 / (2 * m_e)) * (3 * np.pi ** 2 * scenario.n_e) ** (2 / 3)
        T_F_eV = E_F / e
        ratio = scenario.T_eV / T_F_eV

        is_classical = ratio > 0.4
        status = ValidationStatus.SUCCESS if is_classical else ValidationStatus.FAILURE
        message = f"T_e / T_F = {ratio:.2f}. {'经典等离子体，适用PIC。' if is_classical else '量子简并，经典PIC失效！'}"

        return [ValidationResult(
            validator_name="QuantumCheck",
            status=status,
            title="量子效应审查",
            message=message,
            formula="经典区域要求: Tₑ / T_F > 0.4"
        )]


class PICMethodValidator(Validator):
    """检查德拜球内粒子数，判断PIC方法的有效性。"""

    def validate(self, scenario: PlasmaScenario) -> list[ValidationResult]:
        N_D = scenario.n_e * (4 / 3) * np.pi * scenario.debye_length ** 3
        is_valid = N_D > 10

        status = ValidationStatus.SUCCESS if is_valid else ValidationStatus.FAILURE
        message = f"N_D = {N_D:.2e}. {'集体效应主导，PIC有效。' if is_valid else '强耦合，标准PIC失效！'}"

        return [ValidationResult(
            validator_name="PICCheck",
            status=status,
            title="PIC方法有效性 (集体效应)",
            message=message,
            formula=r"要求: N_D = nₑ(4/3)πλ_D³ ≫ 1"
        )]


class SpatialResolutionValidator(Validator):
    """检查德拜长度是否能被网格解析，避免数值加热。"""

    def validate(self, scenario: PlasmaScenario) -> list[ValidationResult]:
        ratio = scenario.debye_length / scenario.dx_m
        is_resolved = ratio > 1.0

        status = ValidationStatus.SUCCESS if is_resolved else ValidationStatus.FAILURE
        message = f"λ_D / Δx = {ratio:.2f}. {'可分辨德拜长度。' if is_resolved else '数值加热风险！'}"

        return [ValidationResult(
            validator_name="ResolutionCheck",
            status=status,
            title="空间分辨率审查",
            message=message,
            formula="避免数值加热要求: λ_D / Δx > 1"
        )]


class TimescaleValidator(Validator):
    """检查碰撞和冷却时间尺度，判断无碰撞近似是否成立。"""

    def validate(self, scenario: PlasmaScenario) -> list[ValidationResult]:
        coulomb_log = 15
        v_th = scenario.v_thermal
        if v_th == 0:
            tau_coll_s = np.inf
        else:
            nu_coll = (scenario.n_e * e ** 4 * coulomb_log) / (4 * np.pi * epsilon_0 ** 2 * m_e ** 2 * v_th ** 3)
            tau_coll_s = 1.0 / nu_coll

        U_rad = (4 * sigma / c) * scenario.T_K ** 4
        sigma_T = 6.6524e-29
        dE_dt = (4 / 3) * sigma_T * c * U_rad * scenario.lorentz_gamma ** 2
        tau_cool_s = (scenario.lorentz_gamma * m_e * c ** 2) / dE_dt if dE_dt > 0 else np.inf

        coll_ratio = tau_coll_s / scenario.total_sim_time_s
        cool_ratio = tau_cool_s / scenario.total_sim_time_s

        status_coll = ValidationStatus.SUCCESS if coll_ratio > 10 else ValidationStatus.WARNING
        status_cool = ValidationStatus.SUCCESS if cool_ratio > 10 else ValidationStatus.WARNING

        results = [
            ValidationResult("TimescaleCheck", status_coll, r"碰撞时间 $\tau_{coll} / T_{sim}$", f"{coll_ratio:.1e}"),
            ValidationResult("TimescaleCheck", status_cool, r"冷却时间 $\tau_{cool} / T_{sim}$", f"{cool_ratio:.1e}",
                             r"无碰撞/辐射近似要求: $\tau / T_{sim} \gg 1$")
        ]
        return results


class CFLValidator(Validator):
    """校验CFL条件，确保数值稳定性。"""

    def validate(self, scenario: PlasmaScenario) -> list[ValidationResult]:
        dt_max_s = scenario.dx_m / (c * np.sqrt(scenario.dims))
        dt_max_norm = dt_max_s * scenario.omega_pe
        is_stable = scenario.DT < dt_max_norm
        courant_number = scenario.DT / dt_max_norm

        status = ValidationStatus.SUCCESS if is_stable else ValidationStatus.FAILURE
        message = f"Courant Number = {courant_number:.3f}. {'稳定。' if is_stable else '不稳定！'}"

        return [ValidationResult(
            validator_name="CFLCheck",
            status=status,
            title="CFL数值稳定性",
            message=message,
            formula=f"要求: ΔT < ΔT_max = {dt_max_norm:.4f} (for {scenario.dims}D)"
        )]


class MagnetizationValidator(Validator):
    """计算磁化所需的Sigma参数范围。"""

    def validate(self, scenario: PlasmaScenario) -> list[ValidationResult]:
        coulomb_log = 15
        U_p = scenario.total_energy_density
        v_th = scenario.v_thermal

        if v_th > 0:
            nu_coll = (scenario.n_e * e ** 4 * coulomb_log) / (4 * np.pi * epsilon_0 ** 2 * m_e ** 2 * v_th ** 3)
        else:
            nu_coll = np.inf

        # 磁化下限: Omega_ce > nu_coll
        B_min = (scenario.lorentz_gamma * m_e * nu_coll) / e
        sigma_min = (B_min ** 2) / (2 * mu_0 * U_p)

        # 分辨率上限: r_L > dx
        B_max = (scenario.lorentz_gamma * m_e * v_th) / (e * scenario.dx_m) if scenario.dx_m > 0 else 0.0
        sigma_max = (B_max ** 2) / (2 * mu_0 * U_p)

        is_possible = B_max > B_min
        status = ValidationStatus.SUCCESS if is_possible else ValidationStatus.FAILURE

        msg = f"推荐范围: {sigma_min:.1e} < σ < {sigma_max:.1e}" if is_possible else "无 可行 Sigma 范围！需加密网格。"

        return [ValidationResult(
            validator_name="Magnetization",
            status=status,
            title="磁化可行域 (Sigma)",
            message=msg,
            formula=r"要求: B_min(磁化) < B < B_max(分辨)"
        )]


class ScaleWindowValidator(Validator):
    """
    尺度窗口验证器 (Engineering View)
    包含 SI 单位和归一化单位 (normalized to d_e) 的双重输出。
    """

    def validate(self, scenario: PlasmaScenario) -> list[ValidationResult]:
        results = []

        # 准备基础数据
        de = scenario.skin_depth_e
        L_box = scenario.box_length_m
        mfp = scenario.mean_free_path
        lambda_D = scenario.debye_length
        U_p = scenario.total_energy_density

        # 归一化值 (Normalized values)
        L_box_norm = L_box / de if de > 0 else 0
        mfp_norm = mfp / de if de > 0 else np.inf
        lambda_D_norm = lambda_D / de if de > 0 else 0

        # --- 1. 碰撞尺度验证 (Collision Scale) ---
        # 准则: Box < MFP / 10
        is_collisionless = mfp > (L_box * 10.0)

        status_coll = ValidationStatus.SUCCESS if is_collisionless else ValidationStatus.WARNING
        # 同时显示物理值和归一化值
        msg_coll = (f"MFP(平均自由程) = {mfp:.2e} m ({mfp_norm:.1f} d_e), "
                    f"Box(盒子大小) = {L_box:.2e} m ({L_box_norm:.1f} d_e).")

        results.append(ValidationResult(
            validator_name="ScaleWindow",
            status=status_coll,
            title="碰撞尺度 (L vs MFP)",
            message=msg_coll,
            formula=r"Check: L_{box} \ll \lambda_{mfp}"
        ))

        # --- 2. 波动尺度验证 (Wave Scale) ---
        is_wave_resolvable = L_box_norm > 5.0
        status_wave = ValidationStatus.SUCCESS if is_wave_resolvable else ValidationStatus.WARNING

        results.append(ValidationResult(
            validator_name="ScaleWindow",
            status=status_wave,
            title="波动尺度 (Box vs SkinDepth)",
            message=f"盒长 = {L_box_norm:.1f} d_e (物理尺寸 {L_box:.2e} m)",
            formula=r"Check: L_{box} > 5 d_e"
        ))

        # --- 3. 磁化尺度窗口 (Magnetic Scale Window) ---
        v_perp = scenario.v_thermal
        gamma = scenario.lorentz_gamma

        # 计算几何限制下的回旋半径限制
        # 下限 (Grid Limit): 回旋半径不能小于 1.5 倍网格 (否则数值加热/插值错误)
        rho_min_limit = scenario.dx_m * 1.5
        # 上限 (Box Limit): 回旋半径不能大于 1/4 盒子 (否则粒子直接跑出边界，看不出磁化)
        rho_max_limit = L_box / 4.0

        # 将回旋半径限制转换为磁场限制 (B 与 rho 成反比)
        # rho = (gamma m v) / (e B)  =>  B = (gamma m v) / (e rho)
        # 因此: rho_min 对应 B_max; rho_max 对应 B_min
        if rho_min_limit > 0 and rho_max_limit > 0:
            B_max_grid = (gamma * m_e * v_perp) / (e * rho_min_limit)
            B_min_box = (gamma * m_e * v_perp) / (e * rho_max_limit)
        else:
            B_max_grid, B_min_box = 0, 0

        valid_window = B_max_grid > B_min_box

        if valid_window:
            # === 计算对应的 Sigma ===
            # Sigma = B^2 / (2 * mu_0 * Up)
            sigma_min = B_min_box ** 2 / (2 * mu_0 * U_p)
            sigma_max = B_max_grid ** 2 / (2 * mu_0 * U_p)

            status_mag = ValidationStatus.INFO
            # 格式化输出: 第一行是 Sigma，第二行是 Tesla，方便查阅
            msg_mag = (f"Sigma可行域: [{sigma_min:.2e}, {sigma_max:.2e}]\n"
                       f"       对应磁场 B : [{B_min_box:.2e} T, {B_max_grid:.2e} T]")
            formula_mag = r"Grid(B_max) > Box(B_min)"
        else:
            status_mag = ValidationStatus.FAILURE
            msg_mag = f"窗口关闭! Grid限 ({B_max_grid:.2e} T) < Box限 ({B_min_box:.2e} T)"
            formula_mag = "需增加 NX 或 LX"

        results.append(ValidationResult(
            validator_name="ScaleWindow",
            status=status_mag,
            title="磁场工程窗口 (B Field)",
            message=msg_mag,
            formula=formula_mag
        ))

        # --- 4. 特征长度清单 ---
        # 这里我们列出绝对值，但使用科学计数法避免 0.0
        # 同时附带归一化长度
        summary_msg = (
            f"d_e={de:.2e}m | "
            f"λ_D={lambda_D:.2e}m ({lambda_D_norm:.2f} d_e) | "
            f"MFP={mfp:.2e}m ({mfp_norm:.1f} d_e)"
        )

        results.append(ValidationResult(
            validator_name="ScaleWindow",
            status=ValidationStatus.INFO,
            title="特征长度绝对值 (SI)",
            message=summary_msg,
            formula="括号内为归一化单位 ( / d_e)"
        ))

        return results


# =============================================================================
# 核心架构: 第4部分 - 报告器 (Reporters)
# 负责展示验证结果
# =============================================================================
class Reporter(ABC):
    """报告器抽象基类"""

    @abstractmethod
    def generate(self, scenario: PlasmaScenario, results: list[ValidationResult]):
        pass


class ConsoleReporter(Reporter):
    """将验证结果打印到控制台。"""

    STATUS_COLORS = {
        ValidationStatus.SUCCESS: '\033[92m',  # Green
        ValidationStatus.WARNING: '\033[93m',  # Yellow
        ValidationStatus.FAILURE: '\033[91m',  # Red
        ValidationStatus.INFO: '\033[94m',  # Blue
    }
    END_COLOR = '\033[0m'

    def generate(self, scenario: PlasmaScenario, results: list[ValidationResult]):
        print("\n" + f" 分析场景: {scenario.name} ".center(70, "="))

        current_validator = ""
        for res in results:
            if res.validator_name != current_validator:
                # print(f"\n--- {res.title.split(' ')[0]} ---") # 简单分组
                current_validator = res.validator_name

            color = self.STATUS_COLORS.get(res.status, '')
            status_tag = f"[{res.status.name}]"
            print(f"{color}{status_tag:<11} | {res.title:<30} | {res.message}{self.END_COLOR}")
            if res.formula:
                print(f"{'':<14}| {'':<30} | {res.formula}")
        print("=" * 70)


class MatplotlibReporter(Reporter):
    """将验证结果生成为一张总结图片。"""

    STATUS_MAP = {
        ValidationStatus.SUCCESS: ("#2E7D32", "[OK]"),
        ValidationStatus.WARNING: ("#F9A825", "[WARN]"),
        ValidationStatus.FAILURE: ("#C62828", "[FAIL]"),
        ValidationStatus.INFO: ("#1565C0", "[INFO]"),
    }

    def generate(self, scenario: PlasmaScenario, results: list[ValidationResult]):
        fig, ax = plt.subplots(figsize=(12, 16), facecolor='white')
        fig.suptitle(f"等离子体物理参数与模拟有效性审查\n— {scenario.name} —", fontsize=24, weight='bold', y=0.98)

        ax.set_facecolor('white')
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        y_pos = 0.95
        dy = 0.04

        def add_header(title):
            nonlocal y_pos
            ax.text(0.5, y_pos, title, ha='center', va='top', fontsize=18, weight='bold', color='#333333')
            ax.axhline(y_pos - 0.02, xmin=0.1, xmax=0.9, color='gray', linestyle='--')
            y_pos -= 0.05

        def add_result(res: ValidationResult):
            nonlocal y_pos
            color, status_text = self.STATUS_MAP.get(res.status, ('black', ''))

            ax.text(0.05, y_pos, f"{res.title}:", ha='left', va='top', fontsize=14, weight='bold')
            ax.text(0.95, y_pos, f"{status_text} {res.message}", ha='right', va='top', fontsize=14, color=color)

            if res.formula:
                y_pos -= 0.025
                ax.text(0.5, y_pos, res.formula, ha='center', va='top', fontsize=12, color='gray', style='italic')
                y_pos -= (dy - 0.025)
            else:
                y_pos -= dy

        # 渲染输入参数
        add_header("1. 基础输入参数")
        add_result(ValidationResult("Inputs", ValidationStatus.INFO, r"电子密度 $n_e$", f"{scenario.n_e:.2e} m⁻³"))
        add_result(ValidationResult("Inputs", ValidationStatus.INFO, r"电子温度 $T_e$", f"{scenario.T_eV / 1e3:.2f} keV"))
        add_result(ValidationResult("Inputs", ValidationStatus.INFO, r"网格/尺寸/时长", f"NX={scenario.NX}, LX={scenario.LX}, LT={scenario.LT}"))

        # 渲染验证结果
        add_header("2. 模拟有效性审查")
        for result in results:
            add_result(result)

        # 保存文件
        safe_filename = re.sub(r'[\\/*?:"<>|()]', "", scenario.name).replace(' ', '_')
        filename = f"physics_summary_{safe_filename}.png"

        plt.tight_layout(rect=[0, 0.02, 1, 0.93])
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"\n[成功] 场景 '{scenario.name}' 的总结图已保存至: {filename}")
        plt.show()


# =============================================================================
# 核心架构: 第5部分 - 主程序 (Main Execution)
# 组装并运行整个分析流程
# =============================================================================
if __name__ == "__main__":

    test_scenario = PlasmaScenario(
        name="Deuterium Bottleneck (84keV)",
        n_e=7.28e33,
        T_eV=84480.0,
        NX=256, LX=100.0, LT=100.0, DT=0.2, dims=3
    )

    validators = [TimescaleValidator(),TimescaleValidatorV2(),]  # 仅测试修改的部分

    print(f"\n分析场景: {test_scenario.name}")
    print("-" * 70)
    for v in validators:
        for res in v.validate(test_scenario):
            print(f"[{res.status.name:7}] {res.title:30} : {res.message}")
            if res.formula: print(f"{'':10} 说明: {res.formula}")

    exit()

    # --- 1. 定义参数扫描的各个维度 ---

    # 维度一: 核心的物理场景
    plasma_regimes = [
        {
            "name": "Deuterium Bottleneck (84keV)",
            "n_e": 7.28e33,
            "T_eV": 84480.0,
            "extra_NX_times": 1.0,
            "DT":0.2,
        },
        {
            "name": "Mid Temp Case (58keV)",
            "n_e": 3.598e32,
            "T_eV": 58650.0,
            "extra_NX_times": 1.25,
            "DT": 0.15,
        },
        {
            "name": "Low Temp Case (12keV)",
            "n_e": 8.43e23,
            "T_eV": 12230.0,
            "extra_NX_times": 2.75,
            "DT": 0.075,
        },
    ]

    # 维度二: 模拟盒的尺寸、分辨率和时长
    box_setups = [
        # 你的数据中 LX, LY, LZ 和 NX, NY, NZ 都是相等的，所以我们简化处理
        # {"name": "Box_S", "LX": 25.0, "NX": 64, "LT": 300.0},
        # {"name": "Box_M", "LX": 50.0, "NX": 128, "LT": 300.0},
        {"name": "Box_L", "LX": 100.0, "NX": 256, "LT": 100.0},
    ]

    # 维度三: 目标磁化参数 Sigma
    target_sigmas = [0.01, 0.1]

    # --- 2. 通过循环生成所有场景实例 ---

    scenarios_to_run = []
    for regime in plasma_regimes:
        for box in box_setups:
            for target_sigma in target_sigmas:
                temp_str = f"_{int(regime['T_eV'] / 1000)}keV" if regime['T_eV'] < 80000 else ""

                scenario_name = (
                    f"{box['name']}_LT{int(box['LT'])}_sigma{target_sigma}"
                    f"{temp_str}"
                )

                # 创建 PlasmaScenario 实例
                # 假设你的模拟都是3D的
                scenario = PlasmaScenario(
                    name=scenario_name,
                    n_e=regime['n_e'],
                    T_eV=regime['T_eV'],
                    NX=box['NX'] * regime['extra_NX_times'],
                    LX=box['LX'],
                    LT=box['LT'],
                    DT=regime['DT'],
                    dims=3  # 你的数据中 NX=NY=NZ, 假设为3D
                )
                scenarios_to_run.append(scenario)

    print(f"--- 共生成了 {len(scenarios_to_run)} 个模拟场景进行分析 ---")

    # 2. 定义本次分析需要使用的验证器链
    validator_chain = [
        QuantumDegeneracyValidator(),
        PICMethodValidator(),
        SpatialResolutionValidator(),
        TimescaleValidator(),
        CFLValidator(),
        MagnetizationValidator(),
        ScaleWindowValidator()
    ]

    # 3. 定义本次分析需要使用的报告器
    reporters: list[Reporter] = [
        ConsoleReporter(),
    ]
    # 尝试添加 MatplotlibReporter，如果失败则跳过
    try:
        # 我们把 import 语句也放在这里，这样如果库不存在，代码甚至不会尝试去解析 MatplotlibReporter 类
        from matplotlib import pyplot as plt

        # 只有在 import 成功后，才将绘图报告器加入列表
        reporters.append(MatplotlibReporter())
        print("[信息] 检测到 matplotlib 库，绘图功能已启用。")
    except ImportError:
        print("[信息] 未找到 matplotlib 库，绘图功能已禁用。将仅输出到控制台。")

    # 4. 循环执行所有场景分析
    for scenario_to_run in scenarios_to_run:
        # 运行所有验证器
        all_results = []
        for validator in validator_chain:
            validate_results = validator.validate(scenario_to_run)
            all_results.extend(validate_results)

        # 使用所有报告器生成报告
        for reporter in reporters:
            reporter.generate(scenario_to_run, all_results)

    print("\n" + " 所有分析任务完成 ".center(70, "="))
