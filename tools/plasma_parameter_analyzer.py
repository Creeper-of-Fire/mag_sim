# -*- coding: utf-8 -*-

import re

import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import e, m_e, c, hbar, k, epsilon_0, sigma

# 尝试导入字体设置工具，如果失败则忽略（确保代码在无特定环境时也能运行）
try:
    from analysis.utils import setup_chinese_font

    setup_chinese_font()
except ImportError:
    print("[警告] 未找到 'analysis.utils.setup_chinese_font'。将使用默认字体，中文可能无法正常显示。")


# =============================================================================
# 物理计算核心类
# =============================================================================
class PlasmaParameterCalculator:
    """
    计算给定密度和温度下的核心等离子体物理参数。
    所有公式都使用 SI 单位制。
    """

    def __init__(self, n_e, T_eV):
        """
        初始化计算器。
        Args:
            n_e (float): 电子数密度 (m^-3)。
            T_eV (float): 电子温度 (eV)。
        """
        self.n_e = n_e
        self.T_eV = T_eV
        self.T_J = T_eV * e  # 温度 (焦耳)
        self.T_K = self.T_J / k  # 温度 (开尔文)
        self.m_e_c2_eV = m_e * c ** 2 / e

    def get_fermi_energy(self):
        """ 计算费米能量/温度，并判断系统是否进入量子简并状态。 """
        E_F_nr = (hbar ** 2 / (2 * m_e)) * (3 * np.pi ** 2 * self.n_e) ** (2 / 3)
        if E_F_nr / (m_e * c ** 2) > 0.1:
            is_relativistic = True
            E_F = hbar * c * (3 * np.pi ** 2 * self.n_e) ** (1 / 3)
        else:
            is_relativistic = False
            E_F = E_F_nr
        T_F_eV = E_F / e
        T_ratio = self.T_eV / T_F_eV
        is_classical = T_ratio > 0.4
        return {"T_F_eV": T_F_eV, "T_div_TF": T_ratio, "is_classical": is_classical, "is_relativistic": is_relativistic}

    def get_debye_length(self):
        """ 计算德拜长度和德拜球内粒子数。 """
        lambda_D = np.sqrt(epsilon_0 * self.T_J / (self.n_e * e ** 2))
        N_D = self.n_e * (4 / 3) * np.pi * lambda_D ** 3
        is_valid_pic = N_D > 10
        return {"lambda_D_m": lambda_D, "N_D": N_D, "is_valid_pic": is_valid_pic}

    def get_timescales(self):
        """ 估算碰撞时间和康普顿冷却时间。 """
        coulomb_log = 15
        gamma = 1 + self.T_J / (m_e * c ** 2)
        v_th = c * np.sqrt(1 - 1 / gamma ** 2)
        if v_th == 0: return {"tau_coll_s": np.inf, "tau_cool_s": np.inf}
        nu_coll = (self.n_e * e ** 4 * coulomb_log) / (4 * np.pi * epsilon_0 ** 2 * m_e ** 2 * v_th ** 3)
        tau_coll_s = 1.0 / nu_coll
        U_rad = (4 * sigma / c) * self.T_K ** 4
        sigma_T = 6.6524e-29
        dE_dt = (4 / 3) * sigma_T * c * U_rad * gamma ** 2
        tau_cool_s = (gamma * m_e * c ** 2) / dE_dt if dE_dt > 0 else np.inf
        return {"tau_coll_s": tau_coll_s, "tau_cool_s": tau_cool_s}

    def get_coupling_parameter(self):
        """
        计算等离子体耦合参数 Gamma = E_pot / E_kin。
        Gamma << 1: 理想气体 (PIC适用)
        Gamma >= 1: 强耦合/非理想 (PIC失效)
        """
        # Wigner-Seitz 半径 (平均粒子间距)
        a_ws = (3 / (4 * np.pi * self.n_e)) ** (1 / 3)
        # 平均库仑势能
        E_pot = e ** 2 / (4 * np.pi * epsilon_0 * a_ws)
        # 耦合参数 (相对于热能)
        Gamma = E_pot / self.T_J

        is_weakly_coupled = Gamma < 1.0
        return {"Gamma": Gamma, "is_weakly_coupled": is_weakly_coupled}

    def get_magnetization_specs(self, dx_m):
        """
        计算磁化相关的详细参数，包括中间物理量、B场边界以及对应的Sigma边界。
        """
        # --- 0. 准备基础物理量 ---
        from scipy.constants import mu_0
        coulomb_log = 15
        # 洛伦兹因子 (热)
        gamma = 1 + self.T_J / (m_e * c ** 2)
        # 热速度
        v_th = c * np.sqrt(1 - 1 / gamma ** 2)

        # 粒子能量密度 (对应 simulation.py 中的定义: 2*n * (rest + enthalpy))
        # 注意：这里假设 n_e 是单种粒子的密度，因为是对等离子体，所以总能量密度乘2
        U_p = 2 * self.n_e * (m_e * c ** 2 + 3 * self.T_J)

        # --- 1. 计算碰撞频率 nu_coll ---
        if v_th > 0:
            # 相对论修正的碰撞频率估算
            nu_coll = (self.n_e * e ** 4 * coulomb_log) / (4 * np.pi * epsilon_0 ** 2 * m_e ** 2 * v_th ** 3)
        else:
            nu_coll = np.inf

        # --- 2. 推导 B_min (磁化阈值: Omega_ce > nu_coll) ---
        # Omega_ce = eB / (gamma * m) > nu_coll  => B > gamma * m * nu_coll / e
        B_min = (gamma * m_e * nu_coll) / e

        # 对应的 Sigma_min
        sigma_min = (B_min ** 2) / (2 * mu_0 * U_p)

        # --- 3. 推导 B_max (分辨率阈值: r_L > dx) ---
        # r_L = (gamma * m * v_th) / (e * B) > dx => B < (gamma * m * v_th) / (e * dx)
        if dx_m > 0:
            B_max = (gamma * m_e * v_th) / (e * dx_m)
        else:
            B_max = 0.0

        # 对应的 Sigma_max
        sigma_max = (B_max ** 2) / (2 * mu_0 * U_p)

        # --- 4. 计算在临界点处的中间物理量 (用于展示) ---
        # 在 B_min 处的参数 (刚好磁化)
        Omega_ce_at_min = (e * B_min) / (gamma * m_e)
        r_L_at_min = (gamma * m_e * v_th) / (e * B_min) if B_min > 0 else np.inf

        # 在 B_max 处的参数 (刚好能分辨)
        Omega_ce_at_max = (e * B_max) / (gamma * m_e)
        r_L_at_max = (gamma * m_e * v_th) / (e * B_max) if B_max > 0 else 0

        return {
            "vals": {
                "U_p": U_p,
                "v_th": v_th,
                "gamma": gamma,
                "nu_coll": nu_coll
            },
            "limits": {
                "B_min": B_min, "sigma_min": sigma_min,
                "B_max": B_max, "sigma_max": sigma_max
            },
            "debug_at_min": {
                "Omega_ce": Omega_ce_at_min, "r_L": r_L_at_min
            },
            "debug_at_max": {
                "Omega_ce": Omega_ce_at_max, "r_L": r_L_at_max
            }
        }

    def get_cooling_break_even(self):
        """
        计算能量竞争的盈亏平衡点 (Break-even Point)。
        定义：磁重联加速功率 = 康普顿冷却功率 时的磁场强度。
        假设：
          1. 加速效率: E_rec ~ 0.1 * c * B (相对论重联) -> P_acc ~ e * 0.1 * c * B * c
          2. 冷却机制: 逆康普顿 (IC) 占主导
          3. 粒子能量: 取热分布粒子的平均能量 (gamma_th) 作为基准。
             (如果连热粒子都跑不过冷却，高能粒子更跑不过，因为 P_cool ~ gamma^2)
        """
        from scipy.constants import mu_0

        # 1. 基础物理量
        sigma_T = 6.6524e-29  # 汤姆逊散射截面
        # Stefan-Boltzmann constant (imported as sigma usually)
        # U_ph = 4 * sigma_SB * T^4 / c
        U_ph = 4 * sigma * self.T_K ** 4 / c
        gamma = 1 + self.T_J / (m_e * c ** 2)

        # 2. 冷却功率 (单个电子, Watts)
        # P_cool = (4/3) * sigma_T * c * U_ph * gamma^2
        P_cool = (4 / 3) * sigma_T * c * U_ph * gamma ** 2

        # 3. 加速功率 (P_acc = 0.1 * e * c^2 * B)
        # 令 P_acc = P_cool，求解 B
        # B_crit = P_cool / (0.1 * e * c^2)
        B_crit = P_cool / (0.1 * e * c ** 2)

        # 4. 对应的 Sigma 值
        # U_p = 2 * n * (m_e c^2 + 3 kT) (总热焓密度)
        U_p = 2 * self.n_e * (m_e * c ** 2 + 3 * self.T_J)
        sigma_crit = B_crit ** 2 / (2 * mu_0 * U_p)

        return {
            "P_cool_W": P_cool,
            "B_crit": B_crit,
            "sigma_crit": sigma_crit,
            "U_ph": U_ph
        }

    def get_cfl_specs(self, dx_m, dt_sim_norm, dims=1):
        """
        计算 CFL 条件并校验数值稳定性。
        CFL 条件: c * dt < dx / sqrt(dims)

        Args:
            dx_m (float): 物理网格尺寸 (meters)
            dt_sim_norm (float): 模拟输入的时间步长 (无量纲, normalized to 1/w_pe)
            dims (int): 模拟维度 (1, 2, or 3)
        """
        # 1. 物理上的最大允许时间步长 (seconds)
        # dt_max_s = dx / (c * sqrt(dims))
        dt_max_s = dx_m / (c * np.sqrt(dims))

        # 预计算等离子体频率 (rad/s)
        self.w_pe = np.sqrt(self.n_e * e ** 2 / (epsilon_0 * m_e))

        # 2. 转换为无量纲时间步长
        # dt_norm = dt_s * w_pe
        dt_max_norm = dt_max_s * self.w_pe

        # 3. 推荐值 (Courant Number = 0.95 for stability)
        dt_recommend_norm = dt_max_norm * 0.95

        # 4. 判定
        is_stable = dt_sim_norm < dt_max_norm
        courant_number = dt_sim_norm / dt_max_norm

        return {
            "dt_max_s": dt_max_s,
            "dt_max_norm": dt_max_norm,
            "dt_sim_norm": dt_sim_norm,
            "is_stable": is_stable,
            "courant_number": courant_number,
            "dt_recommend_norm": dt_recommend_norm,
            "dims": dims
        }

    def get_gyro_check(self, dt_sim_norm, B_target=None, B_max_res=None):
        """
        检查回旋频率分辨率。
        如果未提供 B_target，则使用 B_max_res (网格能分辨的最大磁场) 进行压力测试。
        """
        dt_s = dt_sim_norm / self.w_pe

        self.gamma = 1 + self.T_J / (m_e * c ** 2)

        # 如果没有指定实际磁场，默认用分辨率上限来做最坏情况分析
        B_eval = B_target if B_target else B_max_res

        # 计算回旋频率 (带相对论修正)
        w_ce = (e * B_eval) / (self.gamma * m_e)

        phase_rotation = w_ce * dt_s

        # 反推当前 DT 允许的最大磁场 (Limit = 0.2)
        max_allowed_B = (0.2 * self.gamma * m_e) / (e * dt_s)

        return {
            "B_eval": B_eval,
            "phase_rotation": phase_rotation,
            "is_accurate": phase_rotation < 0.2,
            "is_stable": phase_rotation < 2.0,
            "max_allowed_B": max_allowed_B
        }


# =============================================================================
# 绘图函数 (已重写)
# =============================================================================
def create_scenario_figure(scenario_name, results, sim_params, filename="physics_validation_summary.png"):
    """
    为单个场景生成一张信息详尽的总结图，清晰展示所有参数、计算和结论。
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 18), facecolor='white')
    fig.suptitle(f"等离子体物理参数与模拟有效性审查\n— {scenario_name} —", fontsize=28, weight='bold', y=0.98)

    res = results
    sim = sim_params

    ax.set_facecolor('white')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    y_pos = 0.96
    dy = 0.045
    header_dy = 0.065

    def add_header(title):
        nonlocal y_pos
        ax.text(0.5, y_pos, title, ha='center', va='top', fontsize=22, weight='bold', color='#333333')
        ax.axhline(y_pos - 0.025, xmin=0.1, xmax=0.9, color='gray', linestyle='--')
        y_pos -= header_dy

    def add_row(label, value, status=None, formula=None):
        nonlocal y_pos
        color = 'black'
        status_text = ""
        if status == "OK":
            color, status_text = '#2E7D32', "[OK] "
        elif status == "FAIL":
            color, status_text = '#C62828', "[FAIL] "
        elif status == "WARN":
            color, status_text = '#F9A825', "[WARN] "
        ax.text(0.05, y_pos, f"{label}:", ha='left', va='top', fontsize=18, weight='bold')
        ax.text(0.95, y_pos, f"{status_text}{value}", ha='right', va='top', fontsize=18, color=color)
        if formula:
            y_pos -= 0.025
            ax.text(0.5, y_pos, formula, ha='center', va='top', fontsize=16, color='gray')
            y_pos -= (dy - 0.02)
        else:
            y_pos -= dy

    # --- Section 1: Inputs ---
    add_header("1. 基础输入参数")
    add_row(r"电子密度 $n_e$", f"{res['inputs']['n_e']:.2e} m$^{{-3}}$")
    add_row(r"电子温度 $T_e$", f"{res['inputs']['T_eV'] / 1e3:.2f} keV")

    # --- Section 2: Simulation Setup ---
    add_header("2. 无量纲模拟参数")
    add_row(r"模拟盒尺寸 $L_X$", f"{sim['LX']:.1f} ($d_e$)")
    add_row(r"模拟总时长 $L_T$", f"{sim['LT']:.1f} ($1/\omega_{{pe}}$)")
    add_row(r"空间网格数 $N_X$", f"{sim['NX']}")

    # --- Section 3: Intermediate Calculations ---
    add_header("3. 中间物理量与尺度换算")
    dx_m = (sim['LX'] * sim['d_e_m']) / sim['NX']
    total_sim_time_s = sim['LT'] / sim['w_pe_s']
    add_row(r"等离子体频率 $\omega_{pe}$", f"{sim['w_pe_s']:.2e} rad/s", formula=r"$\omega_{pe} = \sqrt{n_e e^2 / (\epsilon_0 m_e)}$")
    add_row(r"电子趋肤深度 $d_e$", f"{sim['d_e_m']:.2e} m", formula=r"$d_e = c / \omega_{pe}$")
    add_row(r"物理网格尺寸 $\Delta x$", f"{dx_m:.2e} m", formula=r"$\Delta x = (L_X \cdot d_e) / N_X$")
    add_row(r"物理模拟时长 $T_{sim}$", f"{total_sim_time_s:.2e} s", formula=r"$T_{sim} = L_T / \omega_{pe}$")
    add_row(r"德拜长度 $\lambda_D$", f"{res['debye']['lambda_D_m']:.2e} m", formula=r"$\lambda_D = \sqrt{\epsilon_0 k_B T_e / (n_e e^2)}$")
    add_row(r"费米温度 $T_F$", f"{res['fermi']['T_F_eV'] / 1e3:.2f} keV", formula=r"$E_F \approx \frac{\hbar^2}{2m_e}(3\pi^2 n_e)^{2/3}$")

    # --- Section 4: Validity Checks ---
    add_header("4. 模拟有效性审查")
    status_q = "OK" if res['fermi']['is_classical'] else "FAIL"
    add_row("量子效应审查", f"$T_e / T_F = {res['fermi']['T_div_TF']:.2f}$", status_q, formula="经典区域要求: $T_e / T_F > 0.4$")
    add_row("结论", "经典等离子体" if status_q == 'OK' else "量子简并，经典PIC失效！")
    y_pos -= 0.02
    status_pic = "OK" if res['debye']['is_valid_pic'] else "FAIL"
    add_row("PIC方法有效性", f"$N_D = {res['debye']['N_D']:.2e}$", status_pic, formula=r"集体效应主导要求: $N_D = n_e \cdot \frac{4}{3}\pi \lambda_D^3 \gg 1$")
    add_row("结论", "集体效应主导 (PIC有效)" if status_pic == 'OK' else "强耦合，标准PIC失效！")
    y_pos -= 0.02
    res_ratio = res['debye']['lambda_D_m'] / dx_m
    status_res = "OK" if res_ratio > 1.0 else "FAIL"
    add_row("空间分辨率审查", f"$\\lambda_D / \\Delta x = {res_ratio:.2f}$", status_res, formula="避免数值加热要求: $\lambda_D / \Delta x > 1$")
    add_row("结论", "可分辨德拜长度" if status_res == 'OK' else "数值加热风险！")
    y_pos -= 0.02
    coll_ratio = res['timescales']['tau_coll_s'] / total_sim_time_s
    cool_ratio = res['timescales']['tau_cool_s'] / total_sim_time_s
    status_coll = "OK" if coll_ratio > 10 else "WARN"
    status_cool = "OK" if cool_ratio > 10 else "WARN"
    add_row(r"碰撞/冷却审查", "", formula=r"无碰撞/辐射近似要求: $\tau / T_{sim} \gg 1$")
    add_row(r"  $\tau_{coll} / T_{sim}$", f"{coll_ratio:.1e}", status_coll,
            formula=r"$\tau_{coll} \approx \frac{4\pi \epsilon_0^2 m_e^2 v_{th}^3}{n_e e^4 \ln\Lambda}$")
    add_row(r"  $\tau_{cool} / T_{sim}$", f"{cool_ratio:.1e}", status_cool,
            formula=r"$\tau_{cool} = \frac{\gamma m_e c^2}{P_{Compton}}$, with $P_{Compton} \propto \gamma^2 T^4$")
    add_row("结论", "无碰撞/辐射近似成立" if (status_coll == 'OK' and status_cool == 'OK') else "需考虑碰撞或辐射效应！")

    # --- Section 5: Magnetization & Sigma Range ---
    # 先获取数据
    # 注意：需要先计算 dx_m
    d_e_temp = c / np.sqrt(res['inputs']['n_e'] * e ** 2 / (epsilon_0 * m_e))
    dx_m_temp = (sim['LX'] * d_e_temp) / sim['NX']

    # 重新调用新的计算方法 (假设你已经把 calculator 实例传进来了，或者在外部算好传给 results)
    # 这里为了演示，假设 results 字典里已经有了 'mag' 键，存储了上面 get_magnetization_specs 的返回
    mag = res['mag']

    add_header("5. 磁化参数反推与可行域")

    # 5.1 基础物理量
    v_th_c = mag['vals']['v_th'] / c
    nu_coll = mag['vals']['nu_coll']
    add_row(r"热速度 $v_{th}/c$", f"{v_th_c:.3f}")
    add_row(r"碰撞频率 $\nu_{coll}$", f"{nu_coll:.1e} Hz", formula=r"Spitzer频率 (相对论修正)")

    y_pos -= 0.02

    # 5.2 下限 (磁化要求)
    B_min = mag['limits']['B_min']
    sig_min = mag['limits']['sigma_min']
    omg_min = mag['debug_at_min']['Omega_ce']
    rl_min = mag['debug_at_min']['r_L']

    ax.text(0.05, y_pos, "A. 磁化下限 (Magnetization Limit):", ha='left', va='top', fontsize=16, weight='bold', color='#1565C0')
    y_pos -= 0.03
    add_row(r"  阈值 $B_{min}$", f"{B_min:.1e} T")
    add_row(r"  对应 $\sigma_{min}$", f"{sig_min:.1e}", status="OK", formula=r"要求 $\Omega_{ce} > \nu_{coll}$")
    # 显示这时的中间量
    ax.text(0.5, y_pos, f"(此时 $\Omega_{{ce}} \\approx \\nu_{{coll}} = {omg_min:.1e}$, $r_L = {rl_min:.1e}$ m)", ha='center', va='top', fontsize=14,
            color='gray')
    y_pos -= 0.03

    # 5.3 上限 (分辨率要求)
    B_max = mag['limits']['B_max']
    sig_max = mag['limits']['sigma_max']
    rl_max = mag['debug_at_max']['r_L']

    ax.text(0.05, y_pos, "B. 分辨率上限 (Resolution Limit):", ha='left', va='top', fontsize=16, weight='bold', color='#1565C0')
    y_pos -= 0.03

    status_res = "OK" if B_max > B_min else "FAIL"
    add_row(r"  阈值 $B_{max}$", f"{B_max:.1e} T")
    add_row(r"  对应 $\sigma_{max}$", f"{sig_max:.1e}", status=status_res, formula=r"要求 $r_L > \Delta x$")
    ax.text(0.5, y_pos, f"(此时 $r_L \\approx \\Delta x = {rl_max:.1e}$ m)", ha='center', va='top', fontsize=14, color='gray')
    y_pos -= 0.04

    # 5.4 最终结论
    if B_max > B_min:
        final_msg = f"推荐 Sigma 范围:\n{sig_min:.1e} < σ < {sig_max:.1e}"
        final_color = '#2E7D32'
    else:
        final_msg = "无可行 Sigma 范围！\n(需加密网格或放弃模拟)"
        final_color = '#C62828'

    # ax.text(0.5, y_pos - 0.01, final_msg, ha='center', va='top', fontsize=20, weight='bold', color=final_color)

    # --- Section 6: Energy Competition (New Added) ---
    # 获取盈亏平衡点数据 (假设 calculator 已经计算并传入 results，或者这里现场算)
    # 为了方便集成，建议在主循环里算好放入 results['energy_comp']
    comp = res.get('energy_comp')

    y_pos -= 0.02  # 增加一点间距
    add_header("6. 能量竞争审查 (加速 vs 冷却)")

    B_crit = comp['B_crit']
    sig_crit = comp['sigma_crit']
    P_cool = comp['P_cool_W']

    # 显示冷却功率基准
    add_row(r"单粒子冷却功率 $P_{cool}$", f"{P_cool:.1e} W", formula=r"康普顿冷却 (Thermal $\gamma$)")

    y_pos -= 0.02

    # 显示临界值
    ax.text(0.05, y_pos, "加速胜出的最低门槛 (Break-even):", ha='left', va='top', fontsize=16, weight='bold', color='#E65100')
    y_pos -= 0.03

    add_row(r"  临界磁场 $B_{crit}$", f"{B_crit:.1e} T")
    add_row(r"  临界 $\sigma_{crit}$", f"{sig_crit:.1e}", formula=r"要求 $P_{acc} (0.1 v_A B) > P_{cool}$")

    y_pos -= 0.02

    # --- 最终判定 (Grand Finale) ---
    # 逻辑：我们需要找到一个 B，既要 > B_crit (能加速)，又要 < B_max (网格能分辨)
    # 如果 B_crit > B_max，说明要在网格允许范围内实现加速是不可能的。

    B_max_res = mag['limits']['B_max']  # 分辨率上限

    if B_crit < B_max_res:
        # 存在可行窗口
        win_msg = f"物理可行!\n加速窗口: {B_crit:.1e} T < B < {B_max_res:.1e} T\n(对应 σ > {sig_crit:.1e})"
        win_color = '#2E7D32'  # Green
    else:
        # 窗口关闭
        win_msg = f"物理不可行!\n即使在分辨率极限({B_max_res:.1e} T)下\n加速仍跑不过冷却({B_crit:.1e} T)"
        win_color = '#D32F2F'  # Red

    # ax.text(0.5, y_pos - 0.02, win_msg, ha='center', va='top', fontsize=18, weight='bold', color=win_color,
    #         bbox=dict(boxstyle="round,pad=0.5", fc='#F5F5F5', ec=win_color, lw=2))

    # --- Section 7: CFL & Numerical Stability (New) ---
    add_header("7. 数值稳定性 (CFL Condition)")
    cfl = res['cfl']

    # 显示维度
    dims_str = f"{cfl['dims']}D"
    add_row("模拟维度", dims_str)

    # 显示计算出的最大DT
    add_row(r"理论最大 $\Delta T_{max}$", f"{cfl['dt_max_norm']:.4f}", formula=r"$\Delta T < \Delta X \cdot (d_e / c) \cdot \omega_{pe} / \sqrt{D}$")

    # 显示当前DT和状态
    status_cfl = "OK" if cfl['is_stable'] else "FAIL"
    add_row(r"当前设定 $\Delta T$", f"{cfl['dt_sim_norm']:.4f}", status_cfl)

    # 显示 Courant 数
    courant_color = "green" if cfl['courant_number'] < 1.0 else "red"
    ax.text(0.5, y_pos, f"Courant Number = {cfl['courant_number']:.3f} (Limit: 1.0)",
            ha='center', va='top', fontsize=16, color=courant_color, weight='bold')
    y_pos -= 0.035

    # 推荐值
    if not cfl['is_stable']:
        rec_val = cfl['dt_recommend_norm']
        ax.text(0.5, y_pos, f"⚠️ 推荐修改 $\Delta T$ 至: {rec_val:.4f} 或更小",
                ha='center', va='top', fontsize=18, weight='bold', color='#D32F2F',
                bbox=dict(boxstyle="round,pad=0.3", fc='#FFEBEE', ec='#D32F2F'))
    else:
        ax.text(0.5, y_pos, "数值参数稳定", ha='center', va='top', fontsize=16, color='#2E7D32')

    # --- Section 8: Gyro-Resolution Check ---
    # 假设我们画在最下面
    # 实际整合时请放在 CFL Section 之后

    # 这里是一个独立的展示逻辑，你可以把它插入到之前的函数中
    gyro = results['gyro']

    ax.text(0.5, y_pos, "8. 磁场时间分辨率 (Gyro-Resolution)", ha='center', va='top', fontsize=22, weight='bold', color='#333333')
    ax.axhline(y_pos - 0.025, xmin=0.1, xmax=0.9, color='gray', linestyle='--')
    y_pos -= 0.06

    ax.text(0.05, y_pos, "评估磁场 B:", ha='left', va='top', fontsize=18, weight='bold')
    ax.text(0.95, y_pos, f"{gyro['B_eval']:.1e} T (分辨率上限)", ha='right', va='top', fontsize=18)
    y_pos -= 0.04

    val = gyro['phase_rotation']
    status = "OK" if gyro['is_accurate'] else ("WARN" if gyro['is_stable'] else "FAIL")
    color = "#2E7D32" if status == "OK" else ("#F9A825" if status == "WARN" else "#C62828")

    ax.text(0.05, y_pos, r"回旋相位 $\omega_{ce}\Delta t$:", ha='left', va='top', fontsize=18, weight='bold')
    ax.text(0.95, y_pos, f"{status} [{val:.3f}]", ha='right', va='top', fontsize=18, color=color)

    ax.text(0.5, y_pos - 0.025, r"要求: $< 0.2$ (高精度), $< 2.0$ (稳定)", ha='center', va='top', fontsize=14, color='gray')
    y_pos -= 0.06

    # 反推建议
    ax.text(0.05, y_pos, "当前 $\Delta T$ 支持最大 B:", ha='left', va='top', fontsize=18, weight='bold')
    ax.text(0.95, y_pos, f"{gyro['max_allowed_B']:.1e} T", ha='right', va='top', fontsize=18, color='#1565C0')

    plt.tight_layout(rect=[0, 0.02, 1, 0.93])
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"\n[成功] 场景 '{scenario_name}' 的增强版总结图已保存至: {filename}")
    plt.show()


# =============================================================================
# 主程序
# =============================================================================
if __name__ == "__main__":

    scenarios_def = {
        "氘瓶颈 (Deuterium Bottleneck)": {"n_e": 7.28e33, "T_eV": 84480.0},
        "高能区 (High-Energy Regime)": {"n_e": 9.59e40, "T_eV": 8617000.0}
    }
    sim_params_base = {"NX": 64, "LX": 25.0, "LT": 1000.0, "DT": 0.20}

    print("=" * 70)
    print(" 开始进行等离子体物理参数分析 ".center(70, "="))
    print("=" * 70)

    for name, params in scenarios_def.items():
        print(f"\n{'':-^70}")
        print(f"--- 分析场景: {name} ---".center(70))
        print(f"{'':-^70}")

        calc = PlasmaParameterCalculator(n_e=params['n_e'], T_eV=params['T_eV'])
        fermi_res = calc.get_fermi_energy()
        debye_res = calc.get_debye_length()
        time_res = calc.get_timescales()

        w_pe = np.sqrt(params['n_e'] * e ** 2 / (epsilon_0 * m_e))
        d_e = c / w_pe

        coupling_res = calc.get_coupling_parameter()

        # 计算 dx
        dx_m = (sim_params_base['LX'] * d_e) / sim_params_base['NX']

        # 调用新的磁化计算
        mag_res = calc.get_magnetization_specs(dx_m=dx_m)

        # 5. CFL 数值稳定性校验
        cfl_res = calc.get_cfl_specs(dx_m=dx_m, dt_sim_norm=sim_params_base['DT'], dims=3)

        energy_comp_res = calc.get_cooling_break_even()

        # 我们用 B_max 来做压力测试，看当前 DT 是否能扛得住网格能分辨的最大磁场
        B_max = mag_res['limits']['B_max']
        gyro_res = calc.get_gyro_check(dt_sim_norm=sim_params_base['DT'], B_target=None, B_max_res=B_max)

        # 详细打印所有参数和中间结果 (保留控制台输出)
        print("\n[控制台详细输出]")
        print(f"  - 输入: n_e={params['n_e']:.2e} m^-3, T_e={params['T_eV']:.2e} eV")
        print(f"  - 计算: λ_D={debye_res['lambda_D_m']:.2e} m, T_F={fermi_res['T_F_eV']:.2e} eV")
        dx_m = (sim_params_base['LX'] * d_e) / sim_params_base['NX']
        total_sim_time_s = sim_params_base['LT'] / w_pe
        print(f"  - 模拟尺度: Δx={dx_m:.2e} m, T_sim={total_sim_time_s:.2e} s")
        print(f"  - 有效性: T/T_F={fermi_res['T_div_TF']:.2f}, N_D={debye_res['N_D']:.2e}, λ_D/Δx={debye_res['lambda_D_m'] / dx_m:.2f}")
        print(f"  - 耦合参数: Gamma={coupling_res['Gamma']:.2e} ({'Weak' if coupling_res['is_weakly_coupled'] else 'Strong'})")

        # 打印给控制台看看
        print(f"  - 磁化下限: B > {mag_res['limits']['B_min']:.1e} T (Sigma > {mag_res['limits']['sigma_min']:.1e})")
        print(f"  - 分辨上限: B < {mag_res['limits']['B_max']:.1e} T (Sigma < {mag_res['limits']['sigma_max']:.1e})")

        print(f"  - 能量竞争: P_cool={energy_comp_res['P_cool_W']:.1e} W")
        print(f"  - 盈亏平衡: B > {energy_comp_res['B_crit']:.1e} T (Sigma > {energy_comp_res['sigma_crit']:.1e})")

        print("-" * 30)
        print(f"设定 DT: {sim_params_base['DT']}")
        print(f"网格分辨率上限 B_max: {B_max:.1e} T")
        print(f"回旋相位 w_ce * dt: {gyro_res['phase_rotation']:.4f}")
        if gyro_res['is_accurate']:
            print(f"\033[92m[OK] 时间步长足够小，可以精确模拟高达 {gyro_res['max_allowed_B']:.1e} T 的磁场。\033[0m")
        else:
            print(f"\033[91m[Warning] 时间步长过大！模拟 B_max 时会失真。\033[0m")

        results_for_plot = {
            "inputs": params,
            "fermi": fermi_res,
            "debye": debye_res,
            "timescales": time_res,
            "coupling": coupling_res,
            "mag": mag_res,
            "energy_comp": energy_comp_res,
            "cfl": cfl_res,
            "gyro": gyro_res
        }
        current_sim_params = sim_params_base.copy()
        current_sim_params['d_e_m'] = d_e
        current_sim_params['w_pe_s'] = w_pe

        safe_filename = re.sub(r'[\\/*?:"<>|()]', "", name).replace(' ', '_')
        filename = f"physics_summary_{safe_filename}.png"

        create_scenario_figure(name, results_for_plot, current_sim_params, filename)

    print(f"\n{'':-^70}")
    print(" 分析完成 ".center(70, "="))
    print(f"{'':-^70}")
