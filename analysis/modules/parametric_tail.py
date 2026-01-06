# analysis/modules/parametric_tail.py

import json
from pathlib import Path
from typing import List, Set, Dict, Any, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
from mpmath import quad
from scipy.constants import k as kB, c, m_e, e
from scipy.special import kn as bessel_k
from scipy.optimize import root_scalar

from utils.project_config import FILENAME_HISTORY
from .base_module import BaseComparisonModule
from ..core.simulation import SimulationRun
from ..core.utils import console
from ..plotting.layout import create_analysis_figure
from ..plotting.styles import get_style

# --- 物理常量 ---
ME_C2_J = m_e * c ** 2
J_PER_MEV = e * 1e6
J_TO_KEV = 1.0 / (e * 1e3)


class ParametricTailModule(BaseComparisonModule):
    @property
    def name(self) -> str:
        return "参数扫描：高能尾巴与加热效率"

    @property
    def description(self) -> str:
        return "自动识别变化的输入参数(X轴)，绘制高能尾部能量占比和等效温度的变化趋势。"

    @property
    def required_data(self) -> Set[str]:
        return {'final_spectrum'}

    # =========================================================================
    # 1. 物理计算核心 (温度拟合 & 尾部积分)
    # =========================================================================

    def _solve_temperature_kev(self, avg_ek_mev: float) -> float:
        """根据平均动能反推 Maxwell-Juttner 温度 (keV)"""
        if avg_ek_mev <= 0: return 0.0
        target_avg_ek_j = avg_ek_mev * J_PER_MEV

        def mj_avg_energy(T_K):
            if T_K <= 0: return -1.0
            theta = (kB * T_K) / ME_C2_J
            if theta < 1e-9: return 1.5 * kB * T_K
            # <E_k> = mc^2 * ( 3*theta + K1(1/th)/K2(1/th) - 1 )
            return ME_C2_J * (3 * theta + bessel_k(1, 1.0 / theta) / bessel_k(2, 1.0 / theta) - 1.0)

        # 估算初值
        T_guess = (2.0 / 3.0) * target_avg_ek_j / kB
        try:
            sol = root_scalar(lambda t: mj_avg_energy(t) - target_avg_ek_j,
                              x0=T_guess, bracket=[T_guess * 0.1, T_guess * 10.0], method='brentq')
            return (sol.root * kB) * J_TO_KEV
        except:
            return 0.0

    def _integrate_theoretical_energy(self, T_keV: float, E_cut_MeV: float, total_weight: float) -> float:
        """
        对 Maxwell-Juttner 分布进行数值积分，计算 E > E_cut 部分的理论总能量。
        """
        if T_keV <= 0: return 0.0

        T_J = T_keV * 1e3 * e
        theta = T_J / ME_C2_J

        # MJ PDF (Normalized to 1)
        # f(E) = A * gamma * p * exp(-gamma/theta)
        # 我们这里直接写被积函数: E * f(E)
        norm_factor = 1.0 / (ME_C2_J * theta * bessel_k(2, 1.0 / theta))

        def integrand(E_MeV):
            E_J = E_MeV * J_PER_MEV
            gamma = 1.0 + E_J / ME_C2_J
            pc_J = np.sqrt(E_J * (E_J + 2 * ME_C2_J))
            # f(E) part:
            pdf_val = norm_factor * (pc_J / ME_C2_J) * gamma * np.exp(-gamma / theta) * J_PER_MEV  # Jacobian
            # Return E * f(E)
            return E_MeV * pdf_val

        # 积分范围: [E_cut, Infinity]
        try:
            integral, _ = quad(integrand, E_cut_MeV, np.inf)
            return integral * total_weight  # 乘以总粒子数得到总能量 (MeV)
        except Exception:
            return 0.0

    def _calculate_non_thermal_metrics(self, run: SimulationRun) -> Dict[str, float]:
        """
        核心逻辑：模拟值 - 理论值
        """
        spec = run.final_spectrum
        if spec is None or spec.weights.size == 0:
            return {'T_keV': 0.0, 'non_thermal_fraction': 0.0, 'amplification': 1.0}

        # 1. 基础统计
        total_energy_MeV = np.sum(spec.energies_MeV * spec.weights)
        total_count = np.sum(spec.weights)
        avg_energy_MeV = total_energy_MeV / total_count

        # 2. 拟合温度
        T_keV = self._solve_temperature_kev(avg_energy_MeV)
        T_MeV = T_keV / 1000.0

        # 3. 设定阈值 (4倍温度，通常认为是热分布的边界)
        # 如果是非相对论情况 4T 足够，相对论可能需要更高，这里取 4.0 比较通用
        CUTOFF_FACTOR = 4.0
        E_cut_MeV = CUTOFF_FACTOR * T_MeV if T_MeV > 1e-6 else CUTOFF_FACTOR * avg_energy_MeV

        # 4. 计算模拟的尾部能量
        is_tail = spec.energies_MeV > E_cut_MeV
        U_sim_tail = np.sum(spec.energies_MeV[is_tail] * spec.weights[is_tail])

        # 5. 计算理论的尾部能量 (积分)
        U_th_tail = self._integrate_theoretical_energy(T_keV, E_cut_MeV, total_count)

        # 6. 核心指标：非热能量增益
        # 多出来的能量 = Sim - Theory
        U_excess = max(0.0, U_sim_tail - U_th_tail)

        # 指标A: 非热能量占比 (Excess / Total_System_Energy)
        # 这代表了系统中有多少能量被转化为了非热形式
        non_thermal_frac = U_excess / total_energy_MeV if total_energy_MeV > 0 else 0.0

        # 指标B: 放大因子 (Sim / Theory)
        # 代表尾部比热分布强多少倍
        amplification = U_sim_tail / U_th_tail if U_th_tail > 0 else 1.0

        return {
            'T_keV': T_keV,
            'non_thermal_fraction': non_thermal_frac,
            'amplification': amplification,
            'E_cut_MeV': E_cut_MeV
        }

    # =========================================================================
    # 2. 参数探测核心
    # =========================================================================

    def _get_input_params(self, run: SimulationRun) -> Dict[str, Any]:
        """
        尝试从 FILENAME_HISTORY 中找到该 run 对应的输入参数。
        如果找不到，回退到 run.sim (但 run.sim 可能包含导出参数，不如 history 纯净)。
        """
        run_path = Path(run.path).resolve()

        # 假设结构: job_dir / sim_results / run_dir
        # FILENAME_HISTORY在 job_dir 下
        history_path = run_path.parent.parent / FILENAME_HISTORY

        if history_path.exists():
            try:
                # 倒序读取，匹配最新的记录
                with open(history_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    for line in reversed(lines):
                        try:
                            record = json.loads(line)
                            rec_path = Path(record.get('output_dir', '')).resolve()
                            # 路径匹配 (处理 WSL/Windows 路径差异，这里简化为 name 匹配)
                            if rec_path.name == run_path.name:
                                return record.get('params', {})
                        except:
                            continue
            except Exception as e:
                console.print(f"[yellow]读取 {FILENAME_HISTORY} 出错: {e}[/yellow]")

        # 回退策略：从 dill 对象中提取看似输入的参数
        # 这是一个简单的 heuristic
        params = {}
        for k, v in vars(run.sim).items():
            if isinstance(v, (int, float, str, bool)) and not k.startswith('_'):
                params[k] = v
        return params

    def _detect_variable_parameter(self, runs: List[SimulationRun]) -> Tuple[str, List[Any], List[SimulationRun]]:
        """
        对比所有 runs 的参数，找出哪个参数在变化。
        返回: (参数名, 参数值列表, 排序后的runs)
        """
        # 1. 收集所有参数
        run_params_map = []
        for run in runs:
            p = self._get_input_params(run)
            run_params_map.append({'run': run, 'params': p})

        if not run_params_map:
            return "Unknown", [], runs

        # 2. 找出所有键
        all_keys = set()
        for item in run_params_map:
            all_keys.update(item['params'].keys())

        # 3. 寻找变化量
        varying_keys = []
        for k in all_keys:
            values = set()
            for item in run_params_map:
                # 转换为字符串以便比较 (避免 float 精度问题)
                val = item['params'].get(k, None)
                values.add(str(val))
            if len(values) > 1:
                varying_keys.append(k)

        # 4. 确定 X 轴
        if len(varying_keys) == 0:
            console.print("[yellow]警告: 所有模拟的输入参数似乎都相同？将使用模拟名称作为 X 轴。[/yellow]")
            x_key = "Run Name"
            # 这种情况下无需排序，直接返回
            return x_key, [r.name for r in runs], runs

        elif len(varying_keys) == 1:
            x_key = varying_keys[0]
            console.print(f"[green]✔ 检测到单一扫描变量: [bold]{x_key}[/bold][/green]")
        else:
            # 多个变量在变，让用户选，或者默认选第一个
            # 这里为了自动化，我们优先排除 'task_name' 之类的非物理参数
            physics_keys = [k for k in varying_keys if k not in ['task_name', 'description']]
            x_key = physics_keys[0] if physics_keys else varying_keys[0]
            console.print(f"[yellow]⚠ 检测到多个变量在变化 {varying_keys}。自动选择: [bold]{x_key}[/bold][/yellow]")

        # 5. 根据 X 轴的值对 runs 进行排序
        def sort_key(item):
            val = item['params'].get(x_key, 0)
            # 尝试转为 float 排序，否则按 str
            try:
                return float(val)
            except:
                return str(val)

        run_params_map.sort(key=sort_key)

        sorted_runs = [item['run'] for item in run_params_map]
        sorted_values = [item['params'].get(x_key, "N/A") for item in run_params_map]

        return x_key, sorted_values, sorted_runs

    # =========================================================================
    # 3. 运行与绘图
    # =========================================================================

    def run(self, loaded_runs: List[SimulationRun]):
        console.print("\n[bold magenta]执行: 参数扫描非热能量分析...[/bold magenta]")

        valid_runs = [r for r in loaded_runs if r.final_spectrum]
        if len(valid_runs) < 2:
            console.print("[red]错误: 需要至少 2 个模拟来进行对比。[/red]")
            return

        x_label, x_vals, sorted_runs = self._detect_variable_parameter(valid_runs)

        y_frac = []
        y_amp = []
        y_temp = []

        console.print(f"  正在进行理论积分对比 (Cutoff = 4 * T_fit)...")
        for run in sorted_runs:
            m = self._calculate_non_thermal_metrics(run)
            y_frac.append(m['non_thermal_fraction'])
            y_amp.append(m['amplification'])
            y_temp.append(m['T_keV'])

            console.print(f"    - {run.name}: {x_label}={self._get_input_params(run).get(x_label)} -> "
                          f"Excss={m['non_thermal_fraction'] * 100:.2f}%, Amp={m['amplification']:.1f}x")

        # 绘图数据准备
        try:
            x_num = [float(v) for v in x_vals]
            is_num = True
        except:
            x_num = range(len(x_vals))
            is_num = False

        style = get_style()
        fname = f"scan_nonthermal_{x_label}"

        # 创建 2 个子图：一个画非热占比，一个画温度
        with create_analysis_figure(sorted_runs, "scan_nonthermal", num_plots=2, figsize=(9, 8), override_filename=fname) as (fig, (ax1, ax2)):

            # --- 图1: 非热能量占比 (真正的高能部分) ---
            # 这是一个非常严格的指标
            ax1.plot(x_num, np.array(y_frac) * 100, 'o-', color='crimson', lw=2, label='非热能量占比')
            ax1.set_ylabel(r"非热能量占比 (%)" + "\n" + r"$(U_{sim} - U_{th}) / U_{total}$")
            ax1.set_title(f"非热加速效率 vs {x_label}", fontsize=14)
            ax1.grid(True, linestyle='--', alpha=0.5)

            # 可以在右轴画放大因子 (Amplification)
            ax1b = ax1.twinx()
            ax1b.plot(x_num, y_amp, 'd--', color='navy', alpha=0.6, label='尾部放大倍数')
            ax1b.set_ylabel(r"尾部放大倍数 ($U_{sim}/U_{th}$)", color='navy')
            ax1b.tick_params(axis='y', labelcolor='navy')
            ax1b.set_yscale('log')  # 放大倍数通常跨度很大

            # --- 图2: 整体温度 (加热效果) ---
            ax2.plot(x_num, y_temp, 's-', color='darkorange', lw=2)
            ax2.set_ylabel("整体等效温度 $T_{fit}$ (keV)")
            ax2.set_title(f"整体加热效果 vs {x_label}", fontsize=14)
            ax2.set_xlabel(x_label if is_num else "Simulation Case", fontsize=12)
            ax2.grid(True, linestyle='--', alpha=0.5)

            if not is_num:
                ax1.set_xticks(x_num)
                ax1.set_xticklabels(x_vals, rotation=45)
                ax2.set_xticks(x_num)
                ax2.set_xticklabels(x_vals, rotation=45)

            plt.subplots_adjust(hspace=0.3)