# analysis/tools/slimmer.py
import gc  # 引入垃圾回收
import multiprocessing
import os
import shutil
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from time import sleep
from typing import Dict, Any, Tuple, Optional, List

import h5py
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from rich.progress import (
    Progress, SpinnerColumn, BarColumn, TextColumn,
    TimeRemainingColumn, MofNCompleteColumn
)
from rich.prompt import Confirm
from scipy.constants import c, m_e, e

from ..core.data_loader import _get_h5_dataset
from ..core.utils import console

# --- 配置参数 ---
N_BINS = 2048
TARGET_PPC = 200
CORE_BIN_FRACTION = 0.3
DELTA = 1e-9

# 限制并发数，绘图和H5操作较吃内存，建议设为 CPU 核心数的一半或更少
MAX_CONCURRENT_PROCESSES = max(1, os.cpu_count() // 4)


# --- 内部算法函数 (保持数学逻辑不变) ---

def _calculate_kinetic_energy(px, py, pz) -> np.ndarray:
    p_sq = px ** 2 + py ** 2 + pz ** 2
    if p_sq.size == 0:
        return np.array([])
    m_e_c2_J = m_e * c ** 2
    J_PER_MEV = e * 1e6
    total_energy_J = np.sqrt(p_sq * c ** 2 + m_e_c2_J ** 2)
    return (total_energy_J - m_e_c2_J) / J_PER_MEV


def _create_hybrid_bins(energies: np.ndarray) -> Tuple[np.ndarray, float]:
    pos_energies = energies[energies > 0]
    if pos_energies.size < 100:
        return np.logspace(-4, 3, N_BINS + 1), 1.0

    e_min, e_max = pos_energies.min() * 0.9, pos_energies.max() * 1.1
    counts, p_bins = np.histogram(pos_energies, bins=100)
    peak_idx = np.argmax(counts)
    e_trans = (p_bins[peak_idx] + p_bins[peak_idx + 1]) / 2.0

    n_bins_core = int(N_BINS * CORE_BIN_FRACTION)
    n_bins_tail = N_BINS - n_bins_core

    bins_core = np.logspace(np.log10(e_min), np.log10(e_trans), n_bins_core, endpoint=False)
    bins_tail = np.logspace(np.log10(e_trans), np.log10(e_max), n_bins_tail)
    return np.concatenate((bins_core, bins_tail)), e_trans


def _compress_species_data(px, py, pz, weights, bins,
                           progress_queue: multiprocessing.Queue, task_id: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if weights.size == 0:
        return (np.array([]), np.array([]), np.array([]), np.array([]))

    energies = _calculate_kinetic_energy(px, py, pz)
    bin_indices = np.digitize(energies, bins)

    keep_indices = []
    new_weights = weights.copy()
    unique_bins, counts = np.unique(bin_indices, return_counts=True)

    # 汇报此物种的总步数
    total_steps = len(unique_bins)
    if total_steps > 0:
        progress_queue.put(('add_steps', task_id, total_steps))

    for i, (b_idx, count) in enumerate(zip(unique_bins, counts)):
        if b_idx == 0 or b_idx > N_BINS:
            continue

        indices_in_this_bin = np.where(bin_indices == b_idx)[0]
        if count <= TARGET_PPC:
            keep_indices.extend(indices_in_this_bin)
        else:
            factor = count / TARGET_PPC
            # 使用随机选择
            selected = np.random.choice(indices_in_this_bin, TARGET_PPC, replace=False)
            new_weights[selected] *= factor
            keep_indices.extend(selected)

        # 降低通信频率，每处理50个bin汇报一次，防止死锁
        if i % 50 == 0:
            progress_queue.put(('update', task_id, 50))

    # 补齐剩余进度
    remaining = total_steps % 50
    if remaining > 0:
        progress_queue.put(('update', task_id, remaining))

    keep_indices = np.array(keep_indices, dtype=int)
    return px[keep_indices], py[keep_indices], pz[keep_indices], new_weights[keep_indices]


def _generate_comparison_plot(original_data: Dict[str, np.ndarray], compressed_data: Dict[str, np.ndarray],
                              filename: str, output_path: Path):
    try:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), height_ratios=[3, 1], sharex=True)

        plot_bins, _ = _create_hybrid_bins(original_data['energy'])
        centers = np.sqrt(plot_bins[:-1] * plot_bins[1:])
        widths = np.diff(plot_bins)

        counts_raw, _ = np.histogram(original_data['energy'], bins=plot_bins, weights=original_data['weights'])
        dNdE_raw = counts_raw / widths
        ax1.plot(centers, dNdE_raw, 'k-', lw=3, alpha=0.6, label='Original')

        counts_comp, _ = np.histogram(compressed_data['energy'], bins=plot_bins, weights=compressed_data['weights'])
        dNdE_comp = counts_comp / widths
        ax1.plot(centers, dNdE_comp, 'r--', lw=1.5, label=f'Compressed (PPC={TARGET_PPC})')

        ax1.set_yscale('log')
        ax1.set_xscale('log')
        ax1.set_ylabel('dN/dE [/MeV]')
        ax1.set_title(f'Preview: {filename}')
        ax1.legend()
        ax1.grid(True, which='both', alpha=0.3)

        valid = (dNdE_raw > DELTA)
        rel_error = np.zeros_like(dNdE_raw)
        rel_error[valid] = (dNdE_comp[valid] - dNdE_raw[valid]) / dNdE_raw[valid]
        ax2.plot(centers, rel_error, 'b.-', lw=1)
        ax2.axhline(0, color='k', ls='--')
        ax2.set_ylabel('Rel. Err')
        ax2.set_xlabel('Energy (MeV)')
        ax2.set_ylim(-0.5, 0.5)
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig(output_path, dpi=100)
    except Exception as e:
        print(f"Plotting error for {filename}: {e}")
    finally:
        plt.close('all')  # 确保释放内存


# --- Worker Process ---

def process_single_file(h5_path: Path, dir_name: str, plot_output_dir: Path, progress_queue: multiprocessing.Queue) -> Optional[Tuple[Path, Path, int, int]]:
    unique_task_id = f"{dir_name}:{h5_path.name}"  # 使用文件名作为ID，显示更简洁

    temp_h5_path = h5_path.with_suffix(".h5.tmp")
    # 在预览图中加入 dir_name 前缀，防止同名覆盖
    plot_filename = f"{dir_name}_{h5_path.stem}_preview.png"
    plot_path = plot_output_dir / plot_filename

    # 通知主进程添加任务条
    progress_queue.put(('start_file', unique_task_id))

    # 如果临时文件存在 且 预览图也存在，说明之前处理成功了
    if temp_h5_path.exists() and plot_path.exists():
        try:
            # 虽然跳过计算，但为了最后的统计数据，我们需要快速读取粒子数
            # 这比重新计算要快得多
            recovered_raw = 0
            recovered_comp = 0

            # 快速扫描原始文件
            with h5py.File(h5_path, 'r') as f_orig:
                step = list(f_orig['data'].keys())[0]
                ptcl_grp = f_orig[f'data/{step}/particles']
                for sp in ptcl_grp.keys():
                    if 'weighting' in ptcl_grp[sp]:
                        recovered_raw += ptcl_grp[sp]['weighting'].size

            # 快速扫描临时文件
            with h5py.File(temp_h5_path, 'r') as f_temp:
                step = list(f_temp['data'].keys())[0]
                ptcl_grp = f_temp[f'data/{step}/particles']
                for sp in ptcl_grp.keys():
                    if 'weighting' in ptcl_grp[sp]:
                        recovered_comp += ptcl_grp[sp]['weighting'].size

            # 发送跳过信号
            progress_queue.put(('skipped', unique_task_id))
            # 返回结果，如同刚刚计算完一样
            return (temp_h5_path, h5_path, recovered_raw, recovered_comp)

        except Exception as e:
            # 如果读取校验失败，说明文件损坏，继续下面的重新处理流程
            # 并不需要打印错误，默默重新跑即可
            pass

    try:
        all_energies_for_bins = []
        species_names = []
        total_raw_particles = 0

        # Pass 1: Gather Energy Info
        with h5py.File(h5_path, 'r') as f_in:
            step_key = list(f_in['data'].keys())[0]
            particles_group = f_in[f'data/{step_key}/particles']

            for species_name in particles_group.keys():
                if 'photon' in species_name: continue
                if 'momentum/x' in particles_group[species_name] and 'weighting' in particles_group[species_name]:
                    species_names.append(species_name)

            for species_name in species_names:
                species_group = particles_group[species_name]
                px = _get_h5_dataset(species_group, 'momentum/x')
                py = _get_h5_dataset(species_group, 'momentum/y')
                pz = _get_h5_dataset(species_group, 'momentum/z')
                w = _get_h5_dataset(species_group, 'weighting')
                if w.size > 0:
                    all_energies_for_bins.append(_calculate_kinetic_energy(px, py, pz))
                    total_raw_particles += w.size

        # 显式垃圾回收
        gc.collect()

        if not all_energies_for_bins:
            progress_queue.put(('finish_file', unique_task_id))
            return None

        # Create global decision bins
        decision_bins, _ = _create_hybrid_bins(np.concatenate(all_energies_for_bins))
        del all_energies_for_bins
        gc.collect()

        # Pass 2: Compress
        total_comp_particles = 0
        all_energies_raw_plot, all_weights_raw_plot = [], []
        all_energies_comp_plot, all_weights_comp_plot = [], []

        with h5py.File(h5_path, 'r') as f_in, h5py.File(temp_h5_path, 'w') as f_out:
            step_key = list(f_in['data'].keys())[0]
            particles_in_group = f_in[f'data/{step_key}/particles']

            step_out_group = f_out.create_group(f'data/{step_key}')
            particles_out_group = step_out_group.create_group('particles')

            # 复制时间和其他属性
            for key, val in f_in[f'data/{step_key}'].attrs.items():
                step_out_group.attrs[key] = val

            for i_s, species_name in enumerate(species_names):
                species_in_group = particles_in_group[species_name]

                px = _get_h5_dataset(species_in_group, 'momentum/x')
                py = _get_h5_dataset(species_in_group, 'momentum/y')
                pz = _get_h5_dataset(species_in_group, 'momentum/z')
                weights = _get_h5_dataset(species_in_group, 'weighting')

                if weights.size == 0:
                    continue

                all_energies_raw_plot.append(_calculate_kinetic_energy(px, py, pz))
                all_weights_raw_plot.append(weights)

                # Compression
                c_px, c_py, c_pz, c_w = _compress_species_data(
                    px, py, pz, weights, decision_bins,
                    progress_queue, unique_task_id
                )
                total_comp_particles += len(c_w)

                species_out_group = particles_out_group.create_group(species_name)
                mom_out_group = species_out_group.create_group('momentum')
                mom_out_group.create_dataset('x', data=c_px)
                mom_out_group.create_dataset('y', data=c_py)
                mom_out_group.create_dataset('z', data=c_pz)
                species_out_group.create_dataset('weighting', data=c_w)
                # 复制 bound_box 等属性
                for key, val in species_in_group.attrs.items():
                    species_out_group.attrs[key] = val

                if c_w.size > 0:
                    all_energies_comp_plot.append(_calculate_kinetic_energy(c_px, c_py, c_pz))
                    all_weights_comp_plot.append(c_w)

                # 显式清理，防止多物种内存累积导致 crash
                del px, py, pz, weights, c_px, c_py, c_pz, c_w
                gc.collect()

        # Plotting
        final_energies_raw = np.concatenate(all_energies_raw_plot) if all_energies_raw_plot else np.array([])
        final_weights_raw = np.concatenate(all_weights_raw_plot) if all_weights_raw_plot else np.array([])
        final_energies_comp = np.concatenate(all_energies_comp_plot) if all_energies_comp_plot else np.array([])
        final_weights_comp = np.concatenate(all_weights_comp_plot) if all_weights_comp_plot else np.array([])

        _generate_comparison_plot(
            {'energy': final_energies_raw, 'weights': final_weights_raw},
            {'energy': final_energies_comp, 'weights': final_weights_comp},
            f"{dir_name}/{h5_path.name}", # 标题显示目录名
            plot_path
        )

        progress_queue.put(('finish_file', unique_task_id))
        return (temp_h5_path, h5_path, total_raw_particles, total_comp_particles)

    except Exception as e:
        err_msg = traceback.format_exc()
        progress_queue.put(('error', unique_task_id, str(e)))
        # 返回错误信息而不是直接 Crash
        return None


# --- 主控逻辑 (架构解耦的核心) ---

def run_interactive_workflow(selected_dirs: List[str]):
    """
    Slimmer 工具的唯一入口。
    处理目录扫描、并行压缩、进度显示、用户确认和文件替换。
    """
    # 1. 准备全局输出目录
    from ..core.config import config
    preview_base_dir = Path(config.output_dir) / "slimmer_previews"
    preview_base_dir.mkdir(parents=True, exist_ok=True)

    console.print(f"\n[bold]启动粒子数据瘦身工具 (并发数: {MAX_CONCURRENT_PROCESSES})[/bold]")
    console.print(f"预览图存储位置: [cyan]file://{preview_base_dir.resolve()}[/cyan]\n")

    # 2. 扫描文件
    all_files = []
    for d in selected_dirs:
        path = Path(d) / "diags" / "particle_states"
        files = sorted(list(path.glob("*.h5")))
        all_files.extend([(f, Path(d).name, preview_base_dir) for f in files])

    if not all_files:
        console.print("[red]未找到任何 .h5 文件。[/red]")
        return

    # 3. 准备并行和进度条
    manager = multiprocessing.Manager()
    progress_queue = manager.Queue()

    # 修正：列定义不要写死 TextColumn("文件总进度")，否则每行都一样
    progress = Progress(
        TextColumn("[bold blue]{task.description}", justify="right"),  # 动态描述
        SpinnerColumn(),
        BarColumn(bar_width=40),
        "[progress.percentage]{task.percentage:>3.0f}%",
        MofNCompleteColumn(),
        TimeRemainingColumn(),
        console=console
    )

    pending_ops = []  # 存储 (temp, original)
    total_raw_all = 0
    total_comp_all = 0
    skipped_count = 0

    with progress:
        # 总进度条
        main_task_id = progress.add_task("[bold green]总进度", total=len(all_files))
        active_subtasks = {}  # map filename -> task_id

        with ProcessPoolExecutor(max_workers=MAX_CONCURRENT_PROCESSES) as executor:
            # 提交所有任务
            futures = {executor.submit(process_single_file, f[0], Path(f[1]).name, f[2], progress_queue): f[0] for f in all_files}

            finished_files = 0
            while finished_files < len(all_files):
                # 处理消息队列
                while not progress_queue.empty():
                    msg = progress_queue.get()
                    m_type = msg[0]
                    t_id_str = msg[1]

                    if m_type == 'start_file':
                        # 创建子任务，初始 total=1 防止除零，稍后 update 会修正
                        task_id = progress.add_task(f"{t_id_str}", total=100, visible=True)
                        active_subtasks[t_id_str] = task_id

                    elif m_type == 'add_steps':
                        if t_id_str in active_subtasks:
                            tid = active_subtasks[t_id_str]
                            steps = msg[2]
                            # 获取当前 total，如果是初始值100（为了显示条），则重置为实际值
                            # 注意：如果是第二个物种，则是在现有 total 上增加
                            curr_total = progress.tasks[tid].total
                            # 这里的逻辑：我们简单地累加 steps。
                            # 初始 add_task(total=100) 只是占位。
                            # 第一次 add_steps: 如果 total=100，设为 steps。否则 += steps。
                            if curr_total == 100 and progress.tasks[tid].completed == 0:
                                progress.update(tid, total=steps, completed=0)
                            else:
                                progress.update(tid, total=curr_total + steps)

                    elif m_type == 'update':
                        if t_id_str in active_subtasks:
                            progress.update(active_subtasks[t_id_str], advance=msg[2])

                    elif m_type == 'skipped':
                        if t_id_str in active_subtasks:
                            tid = active_subtasks.pop(t_id_str)
                            # 将进度条设为绿色完成状态，并注明已存在
                            progress.update(tid, completed=100, total=100, description=f"[dim]{t_id_str} (Skipped)[/dim]")
                            # 稍微延迟后隐藏，或者保持显示让用户知道跳过了
                            sleep(0.05)
                            progress.update(tid, visible=False)
                            progress.advance(main_task_id)
                            finished_files += 1
                            skipped_count += 1

                    elif m_type == 'finish_file':
                        if t_id_str in active_subtasks:
                            tid = active_subtasks.pop(t_id_str)
                            progress.update(tid, visible=False)  # 完成后隐藏
                            progress.advance(main_task_id)
                            finished_files += 1

                    elif m_type == 'error':
                        if t_id_str in active_subtasks:
                            tid = active_subtasks.pop(t_id_str)
                            progress.update(tid, visible=False)
                        console.print(f"[red]Task Error ({t_id_str}): {msg[2]}[/red]")
                        progress.advance(main_task_id)  # 即使错误也算完成计数
                        finished_files += 1

                sleep(0.1)

            # 收集结果
            for fut in as_completed(futures):
                try:
                    res = fut.result()
                    if res:
                        pending_ops.append((res[0], res[1]))
                        total_raw_all += res[2]
                        total_comp_all += res[3]
                except Exception as e:
                    console.print(f"[red]Critical Future Error: {e}[/red]")

    # 4. 统计与交互确认
    if not pending_ops:
        console.print("[yellow]没有生成有效数据，操作结束。[/yellow]")
        return

    ratio = total_raw_all / total_comp_all if total_comp_all > 0 else 1
    console.print("\n" + "=" * 50)
    console.print(f"[bold white]处理完成摘要:[/bold white]")
    console.print(f"  文件数量: {len(pending_ops)}")
    console.print(f"  原始粒子: {total_raw_all:.2e}")
    console.print(f"  压缩粒子: {total_comp_all:.2e} (压缩比 {ratio:.1f}x)")
    console.print("=" * 50)

    if Confirm.ask(f"[bold red]是否使用压缩后的文件替换这 {len(pending_ops)} 个原始文件?[/bold red]"):
        console.print("正在替换文件...")
        for tmp, orig in pending_ops:
            shutil.move(tmp, orig)
        console.print("[green]替换完成。[/green]")
    else:
        console.print("正在清理临时文件...")
        for tmp, _ in pending_ops:
            if tmp.exists(): os.remove(tmp)
        console.print("[yellow]操作已取消。[/yellow]")
