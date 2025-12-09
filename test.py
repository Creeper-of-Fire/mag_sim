import numpy as np
import matplotlib.pyplot as plt

# --- 设置 ---
# 使用中文显示
plt.rcParams['font.sans-serif'] = ['Source Han Sans SC']  # 'SimHei' 是黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 1. 定义数据范围 (对数空间)
# 能量范围：0.01 到 100 (任意单位)
E = np.logspace(-2, 2.2, 1000)

# 2. 定义函数
# T = 1 (特征温度)
T = 1.0

# (A) 纯热平衡谱 (Maxwell-Boltzmann 核心形态: E^0.5 * e^(-E/T))
# 为了画图好看，做了归一化处理
f_thermal = E**0.5 * np.exp(-E/T)
norm = np.max(f_thermal)
f_thermal = f_thermal / norm

# (B) 幂律尾巴 (Power Law: E^-alpha)
# 我们需要它在某个能量点 (E_cross) 与热谱平滑/相交
alpha = 2.5  # 幂律指数 (越小尾巴越平，越大尾巴越陡)
E_cross = 8.0 # 衔接点 (你可以改这个数来决定尾巴从哪里开始翘起来)

# 计算系数 A，使得在该点 f_thermal(E_cross) == A * E_cross^-alpha
A_factor = (E_cross**0.5 * np.exp(-E_cross/T) / norm) / (E_cross**(-alpha))
f_powerlaw = A_factor * E**(-alpha)

# (C) 组合谱 (取两者中的最大值，或者简单的平滑过渡)
# 这里用 max 模拟 "非热部分盖过了热部分的指数衰减"
f_combined = np.maximum(f_thermal, f_powerlaw)

# 3. 绘图
plt.figure(figsize=(8, 6), dpi=150) # 高清图

# 设置Log-Log坐标
plt.xscale('log')
plt.yscale('log')

# -- 画原本的热谱向下延伸 (蓝色虚线) --
# 只画 E_cross 之后的部分，或者画全长表示参照
plt.plot(E, f_thermal, linestyle='--', color='blue', linewidth=2, alpha=0.6, label='Maxwellian (Thermal)')

# -- 画主体热谱 (蓝色实线) --
# 我们只画 E < E_cross 的部分，表示这是主要分布
mask_thermal = E <= E_cross
plt.plot(E[mask_thermal], f_thermal[mask_thermal], color='blue', linewidth=3)

# -- 画非热尾巴 (红色实线) --
# 画 E > E_cross 的部分
mask_tail = E >= E_cross
plt.plot(E[mask_tail], f_powerlaw[mask_tail], color='red', linewidth=3, label='Non-thermal Tail')

# 4. 调整样式 (去刻度，加标签)
plt.xlabel(r"Particle Kinetic Energy $\rightarrow$", fontsize=14)
plt.ylabel(r"Particle Number Density ($dN/dE$) $\rightarrow$", fontsize=14)

# 去掉具体的刻度数字，变成纯示意图
plt.xticks([])
plt.yticks([])

# 加上文字说明 (可选，不想用代码加也可以PPT里自己加文本框)
plt.text(0.02, 0.2, "Thermal Core\n(Bulk Plasma)", color='blue', fontsize=12, fontweight='bold')
plt.text(15, 0.005, "Non-thermal\nHigh-E Tail", color='red', fontsize=12, fontweight='bold')

# 限制一下Y轴范围，避免显示极小值
plt.ylim(1e-4, 1.5)
plt.xlim(1e-2, 150)

# 加个边框
plt.gca().spines['top'].set_visible(True)
plt.gca().spines['right'].set_visible(True)

plt.tight_layout()
plt.show()

# 如果想保存
# plt.savefig("spectrum_schematic.png", transparent=True)