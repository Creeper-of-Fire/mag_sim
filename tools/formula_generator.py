# formula_generator.py
import matplotlib.pyplot as plt
from rich import print as rprint
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

from analysis.utils import setup_chinese_font

# =============================================================================
# 全局配置
# =============================================================================
plt.rcParams['mathtext.fontset'] = 'cm'  # Computer Modern LaTeX 字体
plt.rcParams['axes.unicode_minus'] = False

console = Console()
setup_chinese_font()


class FormulaState:
    def __init__(self):
        # 这里存储基准值的物理描述，例如 "1.0 T", "5 d_e"
        self.base_B0 = "1.0 T"
        self.base_w0 = "1.0 m"


state = FormulaState()


def render_formula_image(N, B_mult, w_mult, filename="formula_output.png"):
    """
    渲染逻辑：
    1. 主公式显示倍数符号 (例如 10 B_0)
    2. 底部批注显示基准值的具体大小 (例如 B_0 = 1.0 T)
    """

    # --- 1. 准备显示文本 ---
    if B_mult == "1" or B_mult == "1.0":
        B_coeff = ""
    elif "B" not in B_mult:
        B_coeff = f"{B_mult} "
    else:
        B_coeff = f"{B_mult} "

    if w_mult == "1" or w_mult == "1.0":
        w_denom = "w_0"
    elif "w" not in w_mult:
        w_denom = f"{w_mult} w_0"
    else:
        w_denom = w_mult

    # --- 2. 拼装单一大字符串 ---

    # 第一部分：主公式 (纯数学公式)
    part_1_formula = (
            r"$\mathbf{B}(\mathbf{r}) = \sum_{i=1}^{" + str(N) + r"} "
            + B_coeff + r"\mathbf{B}_i \cdot \exp\left( - \frac{|\mathbf{r}-\mathbf{r}_i|^2}{(" + w_denom + r")^2} \right)$"
    )

    # 第二部分：中文说明 (核心修改：中文在 $ 外面，公式在 $ 里面)
    part_2_note = (
        r"说明：$|\mathbf{B}_i| \equiv B_0$, 方向任意"
    )

    # 第三部分：基准值 (核心修改：中文在 $ 外面)
    part_3_base = (
            r"基准值：$B_0 = \mathrm{" + state.base_B0 + r"}, \quad "
                                                        r"w_0 = \mathrm{" + state.base_w0 + r"}$"
    )

    # 组合
    combined_text = part_1_formula + "\n" + part_2_note + "\n" + part_3_base

    # --- 2. 绘图 ---
    # 这里的 figsize 不重要，因为我们会用 bbox_inches='tight' 裁剪
    fig = plt.figure(figsize=(1, 1), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')

    # 核心：只创建一个 text 对象，让 matplotlib 处理排版
    ax.text(0.5, 0.5,
            combined_text,
            ha='center', va='center',
            fontsize=20,  # 统一字号
            linespacing=2.0,  # 行间距 (倍数)，让公式不拥挤
            color='black'
            # 移除了边框 bbox，因为你可能想要透明或者纯净的背景，
            # 如果想要边框，可以加 bbox=dict(boxstyle="round,pad=0.5", fc='white', ec="black")
            )

    try:
        # 核心：tight 裁剪，pad_inches 控制边缘留白
        fig.savefig(filename, transparent=False, bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)
        return True
    except Exception as e:
        console.print(f"[bold red]保存失败: {e}[/bold red]")
        return False


def configure_base_values():
    """设置基准值界面"""
    console.clear()
    console.print(Panel("[bold white on blue] 阶段 1: 设置基准值 (Base Values) [/bold white on blue]", expand=False))
    rprint("[dim]输入物理数值，例如 '1.0 T' 或 '0.5 m'。这些值将作为注脚显示。[/dim]\n")

    # 这里有默认值，方便快速确认
    val_b = Prompt.ask("基准磁场 [bold cyan]B0[/bold cyan]", default=state.base_B0)
    val_w = Prompt.ask("基准宽度 [bold cyan]w0[/bold cyan]", default=state.base_w0)

    state.base_B0 = val_b
    state.base_w0 = val_w
    rprint(f"\n[green]✔ 基准值已锁定: B0={val_b}, w0={val_w}[/green]")


def generate_loop():
    """生成循环界面"""
    while True:
        console.rule("[bold yellow]阶段 2: 生成公式 (直接回车退出程序, 'c' 修改基准)[/bold yellow]")

        # 显示当前上下文
        rprint(f"当前基准: [cyan]B0 = {state.base_B0}[/cyan], [cyan]w0 = {state.base_w0}[/cyan]")

        # 1. 获取 N
        n_input = Prompt.ask("\n求和个数 [bold green]N[/bold green] (输入 'c' 修改基准)")
        if n_input.lower() == 'c':
            configure_base_values()
            continue
        if not n_input:
            break  # 退出

        # 2. 获取倍数 - 关键修改：没有 default，必须输入
        # Prompt.ask 如果不给 default，默认行为就是阻塞直到有输入，这正是你想要的
        b_mult = Prompt.ask("磁场倍数 [bold green]k_B[/bold green] (例如输入 10 代表 10B0)")
        while not b_mult.strip():
            b_mult = Prompt.ask("[red]倍数不能为空！[/red] 请输入磁场倍数")

        w_mult = Prompt.ask("宽度倍数 [bold green]k_w[/bold green] (例如输入 2 代表 2w0)")
        while not w_mult.strip():
            w_mult = Prompt.ask("[red]倍数不能为空！[/red] 请输入宽度倍数")

        # 3. 生成
        filename = f"formula_N{n_input}_B{b_mult}_w{w_mult}.png".replace(" ", "")

        with console.status("[bold green]正在渲染...[/bold green]"):
            if render_formula_image(n_input, b_mult, w_mult, filename):
                rprint(f"[bold green]✔ 图片已生成: {filename}[/bold green]")
                # 预览一下生成的 LaTeX，方便核对
                rprint(f"[dim]   -> {b_mult} B0, {w_mult} w0[/dim]")


def main():
    # 1. 先配置一次基准值
    configure_base_values()
    # 2. 进入生成循环
    generate_loop()
    rprint("[bold cyan]程序结束。[/bold cyan]")


if __name__ == "__main__":
    main()
