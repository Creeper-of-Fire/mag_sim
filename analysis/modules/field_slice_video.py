# analysis/modules/field_slice_video.py
import os
import shutil
from pathlib import Path
from typing import List, Set, Tuple

import imageio
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from tqdm import tqdm
import h5py

from .base_module import BaseVideoModule
from ..core.data_loader import _center_field_3d
from ..core.simulation import SimulationRun
from ..core.utils import console
from ..core.config import config  # 需要导入配置以获取输出目录

# --- 用户可配置参数 ---
SLICE_AXIS = 'z'
FPS = 15
QUALITY = 9
CMAP = 'plasma'


class FieldSliceVideoModule(BaseVideoModule):
    @property
    def name(self) -> str:
        return "场切片视频生成"

    @property
    def description(self) -> str:
        return f"为每个模拟生成一个磁场强度 |B| 在 {SLICE_AXIS.upper()} 平面的演化MP4视频。"

    @property
    def required_data(self) -> Set[str]:
        return {'field_files'}

    def run(self, loaded_runs: List[SimulationRun]):
        console.print("\n[bold magenta]执行: 场切片视频生成...[/bold magenta]")

        valid_runs = [r for r in loaded_runs if r.field_files]
        if not valid_runs:
            console.print("[yellow]警告: 没有找到有效的场诊断文件，跳过此分析。[/yellow]")
            return

        for i, run in enumerate(valid_runs):
            console.print(f"\n--- ({i + 1}/{len(valid_runs)}) 正在处理视频 [bold]{run.name}[/bold] ---")
            output_name = f"{run.name}_video_field_slice.mp4"
            self._generate_video_for_run(run, output_name)

    def _get_centered_b_field(self, fpath: str, target_shape: tuple, B_norm: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """从 HDF5 文件中读取、归一化并居中场分量"""
        step = int(os.path.basename(fpath).split('_')[-1].split('.')[0])
        with h5py.File(fpath, 'r') as f:
            base_path = f"/data/{step}/fields/"
            Bx_raw = f[base_path + 'B/x'][:]
            By_raw = f[base_path + 'B/y'][:]
            Bz_raw = f[base_path + 'B/z'][:]

        # 归一化
        Bx = Bx_raw / B_norm
        By = By_raw / B_norm
        Bz = Bz_raw / B_norm

        return Bx, By, Bz

    def _find_global_b_max(self, run: SimulationRun, target_shape: tuple) -> float:
        console.print("  [cyan]预扫描 HDF5 数据以确定颜色条范围...[/cyan]")
        global_max = 0.0
        # 为了速度，可以只抽样扫描
        sample_files = run.field_files[::5] if len(run.field_files) > 5 else run.field_files

        for fpath in tqdm(sample_files, desc="  预扫描", unit="file", leave=False):
            # 注意：现在需要传入 B_norm
            Bx, By, Bz = self._get_centered_b_field(fpath, target_shape, run.sim.B_norm)
            b_mag = np.sqrt(Bx ** 2 + By ** 2 + Bz ** 2)
            if b_mag.size > 0:
                global_max = max(global_max, np.max(b_mag))

        # 如果扫描后最大值为0，给一个小的默认值防止 LogNorm 出错
        if global_max == 0.0:
            global_max = 1.0
            console.print("  [yellow]警告: 全局最大场强为0，使用默认值 1.0[/yellow]")
        else:
            console.print(f"  [green]扫描完成。全局最大 |B| / B_norm = {global_max:.3e}[/green]")
        return global_max

    def _generate_video_for_run(self, run: SimulationRun, output_name: str):
        # 1. 将临时文件夹建在 config.output_dir 内部
        # 这样既整洁，又避免了路径混乱
        base_output_dir = Path(config.output_dir)
        temp_dir = base_output_dir / "temp_video_frames"

        # 确保清理旧的临时文件
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        temp_dir.mkdir(parents=True, exist_ok=True)

        target_shape = (run.sim.NX, run.sim.NY, run.sim.NZ)
        global_b_max = self._find_global_b_max(run, target_shape)
        frame_paths = []

        console.print("  [cyan]开始生成视频帧...[/cyan]")
        for fpath in tqdm(run.field_files, desc="  帧生成", unit="file", leave=False):
            step = int(os.path.basename(fpath).split('_')[-1].split('.')[0])
            time = step * run.sim.dt
            Bx, By, Bz = self._get_centered_b_field(fpath, target_shape, run.sim.B_norm)
            b_mag = np.sqrt(Bx ** 2 + By ** 2 + Bz ** 2)

            if SLICE_AXIS == 'z':
                # 取 Z 轴中间切片
                slice_idx = run.sim.NZ // 2
                data_slice = b_mag[:, :, slice_idx]
                # extent = [xmin, xmax, ymin, ymax]
                extent = [-run.sim.Lx / 2, run.sim.Lx / 2, -run.sim.Ly / 2, run.sim.Ly / 2]
                xlabel, ylabel = "x (m)", "y (m)"
                title = f"|B| @ z=0, t = {time:.2e} s"

            # ... (如果需要扩展 x, y 切片逻辑可在此处添加) ...

            fig, ax = plt.subplots(figsize=(8, 6.5))

            # 这里的 vmin 防止 log(0)
            norm = LogNorm(vmin=max(1e-4 * global_b_max, 1e-9), vmax=global_b_max)

            # 注意 transpose (.T) 以匹配 matplotlib 的 imshow (row, col) -> (y, x) 约定
            im = ax.imshow(data_slice.T, origin='lower', extent=extent, cmap=CMAP, norm=norm)

            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label('|B| / B_norm')
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.set_aspect('equal')

            # 2. 使用绝对路径直接保存，跳过 save_figure 的自动路径拼接
            frame_filename = temp_dir / f"frame_{step:06d}.png"
            fig.savefig(frame_filename, dpi=100)

            # 3. 显式关闭图形以释放内存
            plt.close(fig)

            frame_paths.append(frame_filename)

        console.print(f"  [cyan]正在将 {len(frame_paths)} 帧合成为 '{output_name}'...[/cyan]")

        # 最终视频也保存到 output_dir 中
        final_video_path = base_output_dir / output_name

        try:
            with imageio.get_writer(final_video_path, mode='I', fps=FPS, codec='libx264', quality=QUALITY, pixelformat='yuv420p') as writer:
                for filename in tqdm(frame_paths, desc="  视频合成", unit="frame", leave=False):
                    writer.append_data(imageio.imread(filename))

            console.print(f"  [bold green]✔ 视频已成功保存: {final_video_path}[/bold green]")
        except Exception as e:
            console.print(f"  [red]✗ 视频合成失败: {e}[/red]")
        finally:
            # 清理临时文件
            if temp_dir.exists():
                shutil.rmtree(temp_dir)