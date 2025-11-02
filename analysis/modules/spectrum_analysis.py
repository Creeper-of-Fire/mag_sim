# modules/spectrum_analysis.py

from typing import List, Set, Optional

import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import k as kB, c, m_e, e  # 物理常量
from scipy.optimize import root_scalar
from scipy.special import kn as bessel_k  # 第二类修正贝塞尔函数 K_n

from .base_module import BaseAnalysisModule
from ..core.simulation import SimulationRun
from ..core.utils import console, plot_parameter_table, save_figure
from ..plotting.layout import create_analysis_figure
from ..plotting.spectrum_plotter import SpectrumPlotter

# 为了清晰和效率，在模块级别定义常量
ME_C2_J = m_e * c ** 2  # 电子静能量 (单位: 焦耳)
MEV_TO_J = e * 1e6  # MeV 到 焦耳 的转换因子
KEV_TO_J = e * 1e3  # keV 到 焦耳 的转换因子
J_TO_KEV = 1.0 / KEV_TO_J  # 焦耳 到 keV 的转换因子


class SpectrumAnalysisModule(BaseAnalysisModule):
    @property
    def name(self) -> str:
        return "粒子能谱分析"

    @property
    def description(self) -> str:
        return "绘制初始/最终能谱，并与用户输入的理论热谱对比。"

    @property
    def required_data(self) -> Set[str]:
        return {'initial_spectrum', 'final_spectrum'}

    @staticmethod
    def _calculate_temperature_from_avg_energy(avg_ek_mev: float) -> Optional[float]:
        """
        根据给定的平均动能，数值求解对应的麦克斯韦-朱特纳(M-J)分布的温度。
        该函数是 Mathematica 脚本逻辑的 Python 实现。

        Args:
            avg_ek_mev (float): 从模拟中计算出的粒子平均动能 (单位: MeV)。

        Returns:
            Optional[float]: 计算出的等效温度 (单位: keV)，如果求解失败则返回 None。
        """
        if avg_ek_mev <= 0:
            console.print(f"[yellow]  警告: 平均动能 ({avg_ek_mev:.4f} MeV) 无效，无法计算温度。[/yellow]")
            return None

        target_avg_ek_j = avg_ek_mev * MEV_TO_J

        def avg_kinetic_energy_MJ(T_K: float) -> float:
            """
            根据温度 T (开尔文) 计算 M-J 分布的理论平均动能 (焦耳)。
            <Ek> = (<γ> - 1) * m_e * c^2
            <γ> = 3*θ + K_1(1/θ) / K_2(1/θ)
            θ = k_B * T / (m_e * c^2)
            """
            if T_K <= 0:
                return -np.inf  # 物理上无效的温度
            theta = (kB * T_K) / ME_C2_J
            # 避免 theta 过小导致 1/theta 溢出
            if theta < 1e-9:  # 极低温下，近似于经典气体
                return 1.5 * kB * T_K

            one_over_theta = 1.0 / theta
            avg_gamma = 3 * theta + bessel_k(1, one_over_theta) / bessel_k(2, one_over_theta)
            return (avg_gamma - 1.0) * ME_C2_J

        # 方程: F(T) = avg_kinetic_energy_MJ(T) - target_avg_ek_j = 0
        equation_to_solve = lambda T_K: avg_kinetic_energy_MJ(T_K) - target_avg_ek_j

        # 使用非相对论公式估算一个初始温度，为求解器提供一个好的起点
        # <Ek> ≈ (3/2) * k_B * T  => T_guess = (2/3) * <Ek> / k_B
        T_guess = (2.0 / 3.0) * target_avg_ek_j / kB

        # 增加一个小的下界防止温度为负
        bracket = [max(1.0, T_guess * 0.1), T_guess * 10.0]

        try:
            # 使用 root_scalar 进行数值求解
            # f: 要求解的函数; x0: 初始猜测; bracket: 求解区间
            solution = root_scalar(equation_to_solve, x0=T_guess, bracket=bracket, method='brentq')

            if solution.converged:
                T_final_K = solution.root
                # 将最终温度从开尔文 (K) 转换为千电子伏特 (keV)
                T_final_keV = (kB * T_final_K) * J_TO_KEV
                return T_final_keV
            else:
                console.print(f"[red]  错误: 温度求解器未能收敛。[/red]")
                return None
        except Exception as e:
            console.print(f"[red]  错误: 在求解温度时发生异常: {e}[/red]")
            return None

    def run(self, loaded_runs: List[SimulationRun]):
        console.print("\n[bold magenta]执行: 粒子能谱分析...[/bold magenta]")

        valid_runs = [r for r in loaded_runs if r.initial_spectrum or r.final_spectrum]
        if not valid_runs:
            console.print("[yellow]警告: 没有加载到有效的能谱数据，跳过此分析。[/yellow]")
            return

        # --- 自动计算等效温度 ---
        console.print("\n" + "=" * 50)
        console.print("[bold yellow]      自动计算等效温度 (能谱分析)[/bold yellow]")
        console.print("=" * 50)

        for run in valid_runs:
            console.print(f"\n[bold]处理模拟: [cyan]{run.name}[/cyan][/bold]")
            if run.final_spectrum and run.final_spectrum.weights.size > 0:
                # 1. 计算加权平均动能
                avg_energy_MeV = np.average(run.final_spectrum.energies_MeV, weights=run.final_spectrum.weights)
                console.print(f"  [green]➔ 计算出的最终加权平均动能: [bold white]{avg_energy_MeV:.6f} MeV[/bold white][/green]")

                # 2. 根据平均动能求解等效温度
                console.print("  [dim]  正在求解麦克斯韦-朱特纳分布的等效温度...[/dim]")
                calculated_T_keV = self._calculate_temperature_from_avg_energy(avg_energy_MeV)

                if calculated_T_keV is not None:
                    run.user_T_keV = calculated_T_keV
                    console.print(f"  [green]✔ 计算出的等效温度: [bold white]{run.user_T_keV:.3f} keV[/bold white][/green]")
                else:
                    console.print("[yellow]⚠ 温度计算失败，将不绘制理论谱。[/yellow]")
            else:
                console.print("[yellow]⚠ 最终能谱为空，跳过温度计算。[/yellow]")

        console.print("=" * 50 + "\n")

        # --- 循环绘制每个模拟的结果 ---
        for i, run in enumerate(valid_runs):
            output_name = f"analysis_spectrum_{run.name}.png"
            console.print(f"  ({i + 1}/{len(valid_runs)}) 正在绘制 [bold]{run.name}[/bold]...")
            self._generate_single_run_plot(run, output_name)

    def _generate_single_run_plot(self, run: SimulationRun, output_name: str):
        # --- 实例化绘图器 ---
        spectrum_plotter = SpectrumPlotter()

        # 使用布局管理器
        with create_analysis_figure(run, "analysis_spectrum", num_plots=1, figsize=(10, 6)) as (fig, ax_plot):
            # --- 使用绘图器在 ax_plot 上绘图 ---
            spectrum_plotter.plot(ax_plot, run, run.name)
            spectrum_plotter.setup_axes(ax_plot)