import numpy as np
from scipy.constants import c, m_e, e, epsilon_0


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
        严格根据 Braams & Karney (1987) P4 的公式 计算 D_uu, F_u 和弛豫时间。
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


if __name__ == "__main__":
    n_e = 7.28e33  # 密度 m^-3
    T_eV = 84480.0  # 背景温度 84.48 keV

    analyzer = RelativisticCollisionAnalyzer(n_e=n_e, T_eV=T_eV)

    print(f"背景参数: T_e = {T_eV / 1000:.2f} keV, z = {analyzer.z:.4f}\n" + "=" * 50)

    energies = [1.0, 1.5, 2.0, 5.0, 10.0]  # E/T 比例

    tau_1T = analyzer.evaluate_relaxation(T_eV)["tau_s"]

    for mult in energies:
        res = analyzer.evaluate_relaxation(mult * T_eV)
        tau = res["tau_s"]
        ratio_to_1T = tau / tau_1T

        print(f"E = {mult:>4.1f} T_e | v/c = {res['v_over_c']:.3f} | \gamma = {res['gamma']:.3f}")
        print(f"  -> D_uu = {res['D_uu']:.4e} m^2/s^3")
        print(f"  -> F_u  = {res['F_u']:.4e} m/s^2")
        print(f"  -> 弛豫时间 \tau = {tau:.4e} s (是 1T_e 的 {ratio_to_1T:.2f} 倍)\n")