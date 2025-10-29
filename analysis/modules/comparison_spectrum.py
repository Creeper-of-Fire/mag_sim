# analysis/modules/comparison_spectrum.py

from typing import List, Set

import matplotlib.pyplot as plt

from .base_module import BaseComparisonModule
from ..core.simulation import SimulationRun
from ..core.utils import console, save_figure
from ..plotting.spectrum_plotter import SpectrumComparisonPlotter


class SpectrumComparisonModule(BaseComparisonModule):
    """
    对比分析模块：对比不同模拟的最终粒子能谱。
    """
    @property
    def name(self) -> str:
        return "能谱对比分析"

    @property
    def description(self) -> str:
        return "将多个模拟的最终能谱绘制在同一张图上进行对比。"

    @property
    def required_data(self) -> Set[str]:
        return {'initial_spectrum', 'final_spectrum'}

    def run(self, loaded_runs: List[SimulationRun]):
        """
        执行对比分析和绘图。
        模块负责创建画布并遍历模拟，调用绘图器进行绘制。
        """
        console.print("\n[bold magenta]执行: 能谱对比分析...[/bold magenta]")

        valid_runs = [r for r in loaded_runs if r.final_spectrum and r.final_spectrum.weights.size > 0]

        if len(valid_runs) < 2:
            console.print(f"[yellow]警告: 找到的有效能谱数据不足两个 ({len(valid_runs)}个)，无法进行对比分析。[/yellow]")
            return

        # 1. 获取所有参与对比的模拟的名称并排序
        run_names = tuple(sorted([r.name for r in valid_runs]))
        # 2. 计算哈希值并转换为简短的十六进制字符串
        short_hash = hex(abs(hash(run_names)))[2:8]  # 取6位十六进制哈希
        output_name = f"comparison_spectrum_{short_hash}.png"

        console.print(f"  -> 正在为 {len(valid_runs)} 个模拟生成对比图...")

        try:
            # 1. 实例化绘图器，注入所有数据以计算公共分箱
            plotter = SpectrumComparisonPlotter(valid_runs)
        except ValueError as e:
            console.print(f"[red]错误: 初始化绘图器失败: {e}[/red]")
            return

        # 2. 创建画布
        fig, ax = plt.subplots(figsize=(12, 8))
        fig.suptitle("最终能谱对比分析", fontsize=18)

        # 3. 模块负责遍历，在同一个 ax 上绘制每个模拟的谱线
        for run in valid_runs:
            plotter.plot(ax, run, label=run.name)

        # 4. 设置坐标轴样式
        plotter.setup_axes(ax)

        # 5. 保存图像
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        save_figure(fig, output_name)