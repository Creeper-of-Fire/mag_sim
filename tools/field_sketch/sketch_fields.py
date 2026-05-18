#!/usr/bin/env python3
"""
绘制 OT 涡旋场和高斯磁场的示意图。
不对上真实参数，仅展示场的拓扑结构。
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# ── 字体 ──
rcParams["font.family"] = ["WenQuanYi Micro Hei", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False

N = 300  # 网格分辨率
L = 1.0  # 归一化域大小 [-L/2, L/2]


def make_grid():
    x = np.linspace(-L / 2, L / 2, N)
    z = np.linspace(-L / 2, L / 2, N)
    X, Z = np.meshgrid(x, z)
    return X, Z


# ═══════════════════════════════════════════════════════════
# 1. Orszag-Tang 涡旋 (多组模式数)
# ═══════════════════════════════════════════════════════════

def ot_field(X, Z, n=1):
    """2D OT 涡旋: Bx = -sin(n*kz*z), Bz = sin(2n*kx*x)"""
    k = 2 * np.pi / L
    Bx = -np.sin(n * k * Z)
    Bz = np.sin(2 * n * k * X)
    return Bx, Bz


def draw_ot():
    X, Z = make_grid()

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Orszag-Tang 涡旋场 (示意图)", fontsize=16, fontweight="bold")

    configs = [(1, "n=1 (标准 OT)"), (2, "n=2"), (4, "n=4"), (8, "n=8")]

    for ax, (n, title) in zip(axes.flat, configs):
        Bx, Bz = ot_field(X, Z, n=n)
        B_mag = np.sqrt(Bx ** 2 + Bz ** 2)

        # 磁场强度填色
        ax.pcolormesh(X, Z, B_mag, cmap="inferno", shading="auto", alpha=0.6)

        # 流线
        speed = B_mag.max()
        lw = 1.5 * B_mag / speed if speed > 0 else 1.0
        ax.streamplot(
            X[0], Z[:, 0], Bx, Bz,
            color="white", linewidth=lw, density=1.8, arrowsize=0.8,
        )

        ax.set_title(title, fontsize=13)
        ax.set_xlabel("x")
        ax.set_ylabel("z")
        ax.set_aspect("equal")

    plt.tight_layout()
    plt.savefig("ot_vortex_sketch.png", dpi=150, bbox_inches="tight")
    print("✔ ot_vortex_sketch.png")
    plt.close()


# ═══════════════════════════════════════════════════════════
# 2. 高斯磁场 (不同数量高斯包)
# ═══════════════════════════════════════════════════════════

def gaussian_field(X, Z, num_blobs, seed=42):
    """随机位置和方向的高斯包叠加。"""
    rng = np.random.RandomState(seed)
    w = 0.06 * L  # 高斯宽度
    Bx = np.zeros_like(X)
    Bz = np.zeros_like(Z)

    for _ in range(num_blobs):
        x0 = rng.uniform(-L / 2.4, L / 2.4)
        z0 = rng.uniform(-L / 2.4, L / 2.4)
        angle = rng.uniform(0, 2 * np.pi)
        blob = np.exp(-((X - x0) ** 2 + (Z - z0) ** 2) / w ** 2)
        Bx += np.cos(angle) * blob
        Bz += np.sin(angle) * blob

    return Bx, Bz


def draw_gaussian():
    X, Z = make_grid()

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("高斯磁场 (示意图)", fontsize=16, fontweight="bold")

    configs = [
        (1, "N=1 (单个高斯包)", 42),
        (10, "N=10", 42),
        (50, "N=50", 42),
        (100, "N=100", 42),
    ]

    for ax, (n, title, seed) in zip(axes.flat, configs):
        Bx, Bz = gaussian_field(X, Z, n, seed=seed)
        B_mag = np.sqrt(Bx ** 2 + Bz ** 2)

        ax.pcolormesh(X, Z, B_mag, cmap="viridis", shading="auto", alpha=0.7)

        # 矢量箭头 (稀疏采样)
        skip = max(1, N // 20)
        ax.quiver(
            X[::skip, ::skip], Z[::skip, ::skip],
            Bx[::skip, ::skip], Bz[::skip, ::skip],
            color="white", alpha=0.8, scale=25, width=0.003,
        )

        ax.set_title(title, fontsize=13)
        ax.set_xlabel("x")
        ax.set_ylabel("z")
        ax.set_aspect("equal")

    plt.tight_layout()
    plt.savefig("gaussian_field_sketch.png", dpi=150, bbox_inches="tight")
    print("✔ gaussian_field_sketch.png")
    plt.close()


if __name__ == "__main__":
    draw_ot()
    draw_gaussian()
    print("完成。")
