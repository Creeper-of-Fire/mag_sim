#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# --- 3D磁场切片演化MP4视频生成脚本 (支持批量处理) ---
#
# 功能:
# 1. 交互式选择一个或多个模拟运行目录。
# 2. 对每个选定的目录，独立生成一个MP4视频。
# 3. 视频包含可拖动的进度条，方便分析。
# 4. 自动为每次模拟确定统一的颜色条范围，保证视觉一致性。
# 5. 输出文件以模拟目录命名 (e.g., video_测试_高斯磁场.mp4)。
#
import glob
import os
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Tuple

import dill
import imageio
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from tqdm import tqdm

from warpx_analysis_utils import (
    console,
    setup_chinese_font,
    select_directories,
)

# =============================================================================
# 用户可配置参数
# =============================================================================
# 切片轴 ('x', 'y', 或 'z')
SLICE_AXIS = 'z'
# 视频帧率 (Frames Per Second)
FPS = 15
# 视频质量 (0-10, 10为最高质量)。对于科学可视化，8-9通常很好。
QUALITY = 9
# 颜色图
CMAP = 'plasma'


# =============================================================================
# 核心数据加载与绘图函数 (与 plot_field_evolution_3d.py 中部分函数类似)
# =============================================================================

def _center_field(field: np.ndarray, target_shape: tuple) -> np.ndarray:
    """将一个在3D交错网格上的场分量插值到单元中心。"""
    if field.shape == target_shape:
        return field

    nx, ny, nz = target_shape
    if field.shape == (nx + 1, ny, nz):  # Bx
        return 0.5 * (field[:-1, :, :] + field[1:, :, :])
    elif field.shape == (nx, ny + 1, nz):  # By
        return 0.5 * (field[:, :-1, :] + field[:, 1:, :])
    elif field.shape == (nx, ny, nz + 1):  # Bz
        return 0.5 * (field[:, :, :-1] + field[:, :, 1:])
    else:
        console.print(f"[yellow]警告: 场形状 {field.shape} 无法插值到 {target_shape}。尝试裁剪。[/yellow]")
        return field[:nx, :ny, :nz]


def get_centered_magnetic_field(fpath: str, target_shape: tuple) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """从单个 npz 文件加载并返回中心化的 Bx, By, Bz。"""
    with np.load(fpath) as data:
        Bx_stag = data.get('Bx', np.zeros(target_shape))
        By_stag = data.get('By', np.zeros(target_shape))
        Bz_stag = data.get('Bz', np.zeros(target_shape))
        Bx = _center_field(Bx_stag, target_shape)
        By = _center_field(By_stag, target_shape)
        Bz = _center_field(Bz_stag, target_shape)
        return Bx, By, Bz


def find_global_b_max(field_files: list, target_shape: tuple) -> float:
    """预扫描所有文件以找到全局最大磁场强度，用于统一颜色条。"""
    console.print("[cyan]预扫描数据以确定颜色条范围...[/cyan]")
    global_max = 0.0
    for fpath in tqdm(field_files, desc="预扫描进度"):
        Bx, By, Bz = get_centered_magnetic_field(fpath, target_shape)
        b_magnitude = np.sqrt(Bx ** 2 + By ** 2 + Bz ** 2)
        local_max = np.max(b_magnitude)
        if local_max > global_max:
            global_max = local_max
    console.print(f"[green]扫描完成。全局最大 |B| / B_norm = {global_max:.3e}[/green]")
    return global_max


def generate_gif_frames(run_path: str, sim_obj: SimpleNamespace) -> list:
    """为每个时间步生成一帧图像，并保存在临时目录中。"""
    field_files = sorted(glob.glob(os.path.join(run_path, "diags/fields", "fields_*.npz")))
    if not field_files:
        console.print(f"  [red]✗ 在 'diags/fields/' 目录下找不到任何 .npz 文件。[/red]")
        return []

    # 创建一个临时目录来存放帧图像
    temp_dir = Path("./temp_gif_frames")
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir()

    target_shape = (sim_obj.NX, sim_obj.NY, sim_obj.NZ)

    # 预扫描以获得统一的颜色条范围
    global_b_max = find_global_b_max(field_files, target_shape)

    frame_paths = []

    console.print("[cyan]开始生成GIF帧...[/cyan]")
    for fpath in tqdm(field_files, desc="帧生成进度"):
        step = int(os.path.basename(fpath).split('_')[-1].split('.')[0])
        time = step * sim_obj.dt

        Bx, By, Bz = get_centered_magnetic_field(fpath, target_shape)
        b_magnitude = np.sqrt(Bx ** 2 + By ** 2 + Bz ** 2)

        # 根据 SLICE_AXIS 提取2D切片
        if SLICE_AXIS == 'z':
            slice_index = sim_obj.NZ // 2
            data_slice = b_magnitude[:, :, slice_index]
            extent = [-sim_obj.Lx / 2, sim_obj.Lx / 2, -sim_obj.Ly / 2, sim_obj.Ly / 2]
            xlabel, ylabel = "x (m)", "y (m)"
            title = f"|B| @ z=0, t = {time:.2e} s"
        elif SLICE_AXIS == 'y':
            slice_index = sim_obj.NY // 2
            data_slice = b_magnitude[:, slice_index, :]
            extent = [-sim_obj.Lx / 2, sim_obj.Lx / 2, -sim_obj.Lz / 2, sim_obj.Lz / 2]
            xlabel, ylabel = "x (m)", "z (m)"
            title = f"|B| @ y=0, t = {time:.2e} s"
        elif SLICE_AXIS == 'x':
            slice_index = sim_obj.NX // 2
            data_slice = b_magnitude[slice_index, :, :]
            extent = [-sim_obj.Ly / 2, sim_obj.Ly / 2, -sim_obj.Lz / 2, sim_obj.Lz / 2]
            xlabel, ylabel = "y (m)", "z (m)"
            title = f"|B| @ x=0, t = {time:.2e} s"
        else:
            raise ValueError("SLICE_AXIS 必须是 'x', 'y', 或 'z'")

        # 绘图
        fig, ax = plt.subplots(figsize=(8, 6.5))
        # 使用对数色标可以更好地看清弱场区的结构
        im = ax.imshow(data_slice.T, origin='lower', extent=extent, cmap=CMAP,
                       norm=LogNorm(vmin=max(1e-4 * global_b_max, 1e-9), vmax=global_b_max))

        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('|B| / B_norm', fontsize=12)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=14)
        ax.set_aspect('equal')

        frame_filename = temp_dir / f"frame_{step:06d}.png"
        plt.savefig(frame_filename, dpi=150, bbox_inches='tight')
        plt.close(fig)
        frame_paths.append(frame_filename)

    return frame_paths, temp_dir


def create_video_from_frames(frame_paths: list, temp_dir: Path, output_filename: str):
    """
    使用 imageio 将所有帧图像合成为一个MP4视频。
    """
    console.print(f"[cyan]正在将 {len(frame_paths)} 帧图像合成为 '{output_filename}'...[/cyan]")

    # --- 核心修改点：使用MP4的参数 ---
    with imageio.get_writer(
            output_filename,
            mode='I',
            fps=FPS,
            codec='libx264',  # 使用高效的H.264编码器
            quality=QUALITY,  # 设置视频质量
            pixelformat='yuv420p'  # 保证最佳兼容性
    ) as writer:
        for filename in tqdm(frame_paths, desc="视频合成进度"):
            image = imageio.imread(filename)
            writer.append_data(image)

    shutil.rmtree(temp_dir)
    console.print(f"[bold green]✔ MP4视频已成功保存到: {output_filename}[/bold green]")
    console.print(f"[white]临时目录 '{temp_dir.name}' 已被清理。[/white]")


# =============================================================================
# 主交互流程
# =============================================================================

def main():
    """主执行函数"""
    console.print("[bold inverse] WarpX 3D 磁场切片GIF生成器 [/bold inverse]")
    setup_chinese_font()

    selected_dirs = select_directories()
    if not selected_dirs:
        console.print("\n[yellow]未选择任何目录，程序退出。[/yellow]")
        return

    for dir_path in selected_dirs:
        console.print(f"\n[bold magenta]>>>>>>>>> 开始处理模拟: {os.path.basename(dir_path)} <<<<<<<<<[/bold magenta]")
        param_file = os.path.join(dir_path, "sim_parameters.dpkl")

        try:
            with open(param_file, "rb") as f:
                sim_obj = SimpleNamespace(**dill.load(f))
            console.print("  [green]✔ 成功加载参数文件。[/green]")


            run_name = os.path.basename(dir_path)
            output_video_name = f"video_{run_name}.mp4"

            frame_paths, temp_dir = generate_gif_frames(dir_path, sim_obj)

            if frame_paths:
                create_video_from_frames(frame_paths, temp_dir, output_video_name)
            else:
                console.print("\n[red]未能生成任何帧图像，无法为当前模拟创建GIF。[/red]")

        except Exception as e:
            console.print(f"  [red]✗ 处理模拟 {os.path.basename(dir_path)} 时发生严重错误: {e}[/red]")
            # 如果临时目录存在，也清理一下
            if Path("./temp_gif_frames").exists():
                shutil.rmtree("./temp_gif_frames")
                console.print("[yellow]已清理残留的临时文件。[/yellow]")
            continue  # 继续处理下一个目录

    console.print("\n[bold]分析完成。[/bold]")


if __name__ == "__main__":
    main()
