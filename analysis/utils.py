# =============================================================================
# Matplotlib & 字体
# =============================================================================

def setup_chinese_font():
    """自动查找并设置支持中文的字体。"""

    # 检查并导入核心绘图库 Matplotlib
    try:
        import matplotlib.pyplot as plt
        from matplotlib import font_manager as fm
        from analysis.core.utils import console
    except ImportError:
        print("[警告] 绘图库 Matplotlib 未安装。已跳过所有与绘图相关的字体设置。")
        print("       提示：如需绘图功能，请通过 'pip install matplotlib' 安装。")
        return  # Matplotlib 不存在，后续操作无意义，直接返回

    chinese_fonts_priority = ['WenQuanYi Micro Hei', 'Source Han Sans SC', 'Noto Sans CJK SC', 'SimHei', 'Microsoft YaHei']
    found_font = next((font for font in chinese_fonts_priority if fm.findfont(font, fontext='ttf')), None)
    if found_font:
        plt.rcParams['font.sans-serif'] = [found_font]
        console.print(f"[green]✔ Matplotlib 字体已设置为：{found_font}[/green]")
    else:
        console.print("[yellow]⚠ 警告：未能找到支持中文的字体。图表中的中文可能无法正常显示。[/yellow]")
    plt.rcParams['axes.unicode_minus'] = False
