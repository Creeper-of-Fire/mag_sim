# analysis/modules/parametric_tail.py

import hashlib
import json
from pathlib import Path
from typing import List, Set, Dict, Any, Tuple

import matplotlib.pyplot as plt
import numpy as np
from rich.prompt import Prompt
from scipy.constants import k as kB, c, m_e, e
from scipy.optimize import root_scalar
from scipy.special import kn as bessel_k

from utils.project_config import FILENAME_HISTORY
from .base_module import BaseComparisonModule
from ..core.simulation import SimulationRun
from ..core.utils import console
from ..plotting.layout import create_analysis_figure

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

    def _calculate_mj_pdf(self, E_MeV: np.ndarray, T_keV: float) -> np.ndarray:
        """计算 Maxwell-Juttner 概率密度 f(E)"""
        if T_keV <= 0: return np.zeros_like(E_MeV)
        T_J = T_keV * 1e3 * e
        theta = T_J / ME_C2_J

        # 归一化系数
        norm = 1.0 / (ME_C2_J * theta * bessel_k(2, 1.0 / theta))

        E_J = E_MeV * J_PER_MEV
        gamma = 1.0 + E_J / ME_C2_J
        pc_J = np.sqrt(E_J * (E_J + 2 * ME_C2_J))

        # PDF (per Joule) -> 需要乘以 J_PER_MEV 转为 per MeV
        pdf = norm * (pc_J / ME_C2_J) * gamma * np.exp(-gamma / theta) * J_PER_MEV
        return pdf

    def _calculate_excess_energy(self, run: SimulationRun) -> Dict[str, float]:
        """
        核心算法改进：
        1. 建立对数分箱。
        2. 计算每个箱内的 模拟粒子数 N_sim 和 理论粒子数 N_th。
        3. 计算差值 ΔN = max(0, N_sim - N_th)。
        4. 加权求和：Sum( ΔN * E_center )。
        """
        spec = run.final_spectrum
        if spec is None or spec.weights.size == 0:
            return {'T_keV': 0.0, 'excess_ratio': 0.0, 'total_excess_MeV': 0.0}

        # 1. 基础统计与温度拟合
        total_energy_MeV = np.sum(spec.energies_MeV * spec.weights)
        total_weight = np.sum(spec.weights)
        avg_energy_MeV = total_energy_MeV / total_weight
        T_keV = self._solve_temperature_kev(avg_energy_MeV)

        # 2. 建立分箱 (覆盖整个范围，从极低到极高)
        min_e = max(1e-4, spec.energies_MeV.min())
        max_e = max(10.0, spec.energies_MeV.max() * 1.5)  # 确保覆盖高能尾
        bins = np.logspace(np.log10(min_e), np.log10(max_e), 200)
        centers = np.sqrt(bins[:-1] * bins[1:])
        widths = np.diff(bins)

        # 3. 模拟数据的直方图
        counts_sim, _ = np.histogram(spec.energies_MeV, bins=bins, weights=spec.weights)

        # 4. 理论数据的直方图
        # N_th(bin) ≈ PDF(center) * width * total_weight
        pdf_vals = self._calculate_mj_pdf(centers, T_keV)
        counts_th = pdf_vals * widths * total_weight

        # 5. 核心：计算加权正向差值 (Weighted Positive Excess)
        # 只有当 sim > th 时才计入，且乘以能量 E 进行加权
        diff_counts = counts_sim - counts_th

        # 过滤掉负值 (即模拟 < 理论的部分不扣分)
        positive_diff = np.maximum(0.0, diff_counts)

        # 能量加权积分：Sum ( ΔN * E )
        excess_energy_MeV = np.sum(positive_diff * centers)

        # 6. 归一化指标
        # 非热能量占比 = 溢出的能量 / 总能量
        excess_ratio = excess_energy_MeV / total_energy_MeV

        return {
            'T_keV': T_keV,
            'excess_ratio': excess_ratio,
            'total_excess_MeV': excess_energy_MeV,
            'total_energy_MeV': total_energy_MeV
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

        console.print(f"[red]无法在历史记录 ({history_path.name}) 中找到模拟 run: {run.name}[/red]")

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
        console.print("\n[bold magenta]执行: 参数扫描非热能量分析 (正向差值法)...[/bold magenta]")

        valid_runs = [r for r in loaded_runs if r.final_spectrum]
        if len(valid_runs) < 2:
            console.print("[red]错误: 需要至少 2 个模拟来进行对比。[/red]")
            return

        # 1. 检测变量并排序
        x_label, x_vals, sorted_runs = self._detect_variable_parameter(valid_runs)

        # 2. 生成默认唯一标识符 (Hash)
        # 将所有参与分析的 Run 名字拼起来做哈希，确保只要选的文件不一样，文件名就不一样
        run_names_concat = "".join(sorted([r.name for r in sorted_runs]))
        short_hash = hashlib.md5(run_names_concat.encode('utf-8')).hexdigest()[:6]

        default_filename = f"scan_{x_label}_{short_hash}"

        # 3. 交互式确认文件名 (给你一个改名的机会)
        console.print(f"\n[cyan]默认输出文件名: {default_filename}.png[/cyan]")
        user_suffix = Prompt.ask(
            "请输入文件名后缀 (用于区分实验批次)",
            default=short_hash,
            show_default=True
        )

        # 构造最终文件名
        # 格式: scan_{变量名}_{用户后缀/哈希}
        # 例如: scan_target_sigma_batch1.png
        final_filename = f"scan_{x_label}_{user_suffix}"

        y_ratio = []
        y_energy = []
        y_temp = []

        console.print(f"  正在逐个能箱计算 (Sim - Theory) * Energy ...")
        for run in sorted_runs:
            m = self._calculate_excess_energy(run)
            y_ratio.append(m['excess_ratio'])
            y_energy.append(m['total_excess_MeV'])
            y_temp.append(m['T_keV'])

            console.print(f"    - {run.name}: {x_label}={self._get_input_params(run).get(x_label)} -> "
                          f"Excess Ratio={m['excess_ratio'] * 100:.3f}% (T={m['T_keV']:.1f} keV)")

        try:
            x_num = [float(v) for v in x_vals]
            is_num = True
        except:
            x_num = range(len(x_vals))
            is_num = False

        with create_analysis_figure(sorted_runs, "scan_excess", num_plots=2, figsize=(9, 8), override_filename=final_filename) as (fig, (ax1, ax2)):

            # --- 图1: 非热能量占比 ---
            # 这是最有物理意义的图：到底有多少比例的能量进入了非热部分
            ax1.plot(x_num, np.array(y_ratio) * 100, 'o-', color='crimson', lw=2, markersize=6)
            ax1.set_ylabel(r"非热能量占比 (%)" + "\n" + r"$\sum (N_{sim}-N_{th})E / E_{total}$")
            ax1.set_title(f"非热加速效率 vs {x_label}", fontsize=14)
            ax1.grid(True, linestyle='--', alpha=0.5)

            # --- 图2 (放在下半部分): 背景温度 ---
            # 用于区分是“整体加热”还是“尾部加速”
            ax2.plot(x_num, y_temp, 's-', color='darkorange', lw=2, markersize=6)
            ax2.set_ylabel("整体等效温度 $T_{fit}$ (keV)")
            ax2.set_title(f"背景加热效果 vs {x_label}", fontsize=14)
            ax2.set_xlabel(x_label if is_num else "Simulation Case", fontsize=12)
            ax2.grid(True, linestyle='--', alpha=0.5)

            if not is_num:
                ax1.set_xticks(x_num)
                ax1.set_xticklabels(x_vals, rotation=45)
                ax2.set_xticks(x_num)
                ax2.set_xticklabels(x_vals, rotation=45)

            plt.subplots_adjust(hspace=0.3)
