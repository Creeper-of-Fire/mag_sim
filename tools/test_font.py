# test_font.py
import matplotlib.pyplot as plt
from rich.console import Console


# =============================================================================
# 模拟 analysis/core/utils.py 中的核心函数
# =============================================================================
def setup_chinese_font_for_final_test():
    """
    字体设置函数。思源黑体是正确的选择，因为它能处理中文。
    数学符号将由 mathtext 引擎处理。
    """
    from matplotlib import font_manager as fm

    console = Console()

    # 思源黑体依然是我们的最佳选择，用于渲染所有中文部分
    font_name = 'Source Han Sans SC'

    if fm.findfont(font_name, fontext='ttf'):
        plt.rcParams['font.sans-serif'] = [font_name]
        console.print(f"[bold green]✔ 中文字体已正确设置为：{font_name}[/bold green]")
    else:
        console.print(f"[bold red]错误：找不到中文字体 '{font_name}'。[/bold red]")

    # 解决常规负号显示问题
    plt.rcParams['axes.unicode_minus'] = False


# =============================================================================
# 主测试逻辑
# =============================================================================
def run_final_font_test():
    console = Console()
    console.print("\n[bold cyan]--- 最终解决方案测试 (mathtext 引擎) ---[/bold cyan]")

    # 1. 应用字体设置
    setup_chinese_font_for_final_test()

    # 2. 创建测试图形
    fig, ax = plt.subplots(figsize=(8, 4.5))

    # 3. [核心修改] 使用 mathtext 语法 (r"$...$") 来书写所有公式
    title_text = "最终字体测试：中文与 mathtext 混合渲染"
    label_text_x = r"X轴标签 (单位: $m^{-3}$)"  # 正确的写法
    label_text_y = r"Y轴标签 (物理量: $\sigma$)"  # 正确的写法
    main_text = (
            "中文部分由“思源黑体”渲染。\n"
            "公式部分由 Matplotlib 的数学引擎渲染。\n"
            r"科学符号: $\alpha, \beta, \gamma, \omega, \Omega, \pm, \approx$" + "\n"
                                                                                 r"关键的上标负号: $10^{-19}$"  # 正确的写法
    )

    ax.set_title(title_text, fontsize=16)
    ax.set_xlabel(label_text_x, fontsize=12)
    ax.set_ylabel(label_text_y, fontsize=12)

    ax.text(0.5, 0.5, main_text, ha='center', va='center', fontsize=11, linespacing=1.5)

    ax.set_xticks([])
    ax.set_yticks([])

    # 4. 保存测试图片
    output_file = "font_test_final_solution.png"
    try:
        fig.savefig(output_file, dpi=150)
        plt.close(fig)
        console.print(f"\n[bold green]✔ 测试完成！[/bold green]")
        console.print(f"请检查生成的图片文件：[cyan]'{output_file}'[/cyan]")
        console.print("这一次，所有内容都将完美显示。")
    except Exception as e:
        console.print(f"\n[bold red]错误：在保存图片时发生异常！[/bold red]")
        console.print(f"异常信息: {e}")


if __name__ == "__main__":
    run_final_font_test()