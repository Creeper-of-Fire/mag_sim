# modules/field_slice_video.py
import os

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
from typing import List, Set, Tuple
from pathlib import Path
import shutil
import imageio
from tqdm import tqdm

from .base_module import BaseAnalysisModule
from ..core.simulation import SimulationRun
from ..core.utils import console
# 注意：这个模块需要data_loader里的函数，但为了解耦，我们把它复制过来
from ..core.data_loader import _center_field_3d

# --- 用户可配置参数 ---
SLICE_AXIS = 'z'
FPS = 15
QUALITY = 9
CMAP = 'plasma'


class FieldSliceVideoModule(BaseAnalysisModule):
    @property
    def name(self) -> str:
        return "场切片视频生成"

    @property
    def description(self) -> str:
        return f"为每个模拟生成一个磁场强度 |B| 在 {SLICE_AXIS.upper()} 平面的演化MP4视频。"

    @property
    def required_data(self) -> Set[str]:
        # 这个模块比较特殊，它只需要参数和文件列表，自己处理I/O
        return {'field_files'}

    def run(self, loaded_runs: List[SimulationRun]):
        console.print("\n[bold magenta]执行: 场切片视频生成...[/bold magenta]")

        valid_runs = [r for r in loaded_runs if r.field_files]
        if not valid_runs:
            console.print("[yellow]警告: 没有找到有效的场诊断文件，跳过此分析。[/yellow]")
            return

        for i, run in enumerate(valid_runs):
            console.print(f"\n--- ({i + 1}/{len(valid_runs)}) 正在处理视频 [bold]{run.name}[/bold] ---")
            output_name = f"video_field_slice_{run.name}.mp4"
            self._generate_video_for_run(run, output_name)

    def _get_centered_b_field(self, fpath: str, target_shape: tuple) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        with np.load(fpath) as data:
            Bx = _center_field_3d(data.get('Bx', np.zeros(target_shape)), target_shape)
            By = _center_field_3d(data.get('By', np.zeros(target_shape)), target_shape)
            Bz = _center_field_3d(data.get('Bz', np.zeros(target_shape)), target_shape)
            return Bx, By, Bz

    def _find_global_b_max(self, field_files: List[str], target_shape: tuple) -> float:
        console.print("  [cyan]预扫描数据以确定颜色条范围...[/cyan]")
        global_max = 0.0
        for fpath in tqdm(field_files, desc="  预扫描", unit="file", leave=False):
            Bx, By, Bz = self._get_centered_b_field(fpath, target_shape)
            global_max = max(global_max, np.max(np.sqrt(Bx ** 2 + By ** 2 + Bz ** 2)))
        console.print(f"  [green]扫描完成。全局最大 |B| / B_norm = {global_max:.3e}[/green]")
        return global_max

    def _generate_video_for_run(self, run: SimulationRun, output_name: str):
        temp_dir = Path("./temp_video_frames")
        if temp_dir.exists(): shutil.rmtree(temp_dir)
        temp_dir.mkdir()

        target_shape = (run.sim.NX, run.sim.NY, run.sim.NZ)
        global_b_max = self._find_global_b_max(run.field_files, target_shape)
        frame_paths = []

        console.print("  [cyan]开始生成视频帧...[/cyan]")
        for fpath in tqdm(run.field_files, desc="  帧生成", unit="file", leave=False):
            step = int(os.path.basename(fpath).split('_')[-1].split('.')[0])
            time = step * run.sim.dt
            Bx, By, Bz = self._get_centered_b_field(fpath, target_shape)
            b_mag = np.sqrt(Bx ** 2 + By ** 2 + Bz ** 2)

            if SLICE_AXIS == 'z':
                slice_idx, data_slice = run.sim.NZ // 2, b_mag[:, :, run.sim.NZ // 2]
                extent = [-run.sim.Lx / 2, run.sim.Lx / 2, -run.sim.Ly / 2, run.sim.Ly / 2]
                xlabel, ylabel, title = "x (m)", "y (m)", f"|B| @ z=0, t = {time:.2e} s"
            # ... (此处可添加 'x', 'y' 轴的逻辑)

            fig, ax = plt.subplots(figsize=(8, 6.5))
            norm = LogNorm(vmin=max(1e-4 * global_b_max, 1e-9), vmax=global_b_max)
            im = ax.imshow(data_slice.T, origin='lower', extent=extent, cmap=CMAP, norm=norm)
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label('|B| / B_norm')
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.set_aspect('equal')

            frame_filename = temp_dir / f"frame_{step:06d}.png"
            plt.savefig(frame_filename, dpi=150, bbox_inches='tight')
            plt.close(fig)
            frame_paths.append(frame_filename)

        console.print(f"  [cyan]正在将 {len(frame_paths)} 帧合成为 '{output_name}'...[/cyan]")
        with imageio.get_writer(output_name, mode='I', fps=FPS, codec='libx264', quality=QUALITY, pixelformat='yuv420p') as writer:
            for filename in tqdm(frame_paths, desc="  视频合成", unit="frame", leave=False):
                writer.append_data(imageio.imread(filename))

        shutil.rmtree(temp_dir)
        console.print(f"  [bold green]✔ 视频已成功保存: {output_name}[/bold green]")