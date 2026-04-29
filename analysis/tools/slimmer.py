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

# 绘图时最大采样粒子数，防止绘图撑爆内存
MAX_PLOT_PARTICLES = 500_000

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
        gc.collect()


# --- Worker Process ---

def process_single_file(h5_path: Path, dir_name: str, plot_output_dir: Path, progress_queue: multiprocessing.Queue) -> Optional[Tuple[Path, Path, int, int]]:
    unique_task_id = f"{dir_name}:{h5_path.name}"  # 使用文件名作为ID，显示更简洁

    # 显式调用垃圾回收，清除上一个任务可能残留的内存
    gc.collect()

    temp_h5_path = h5_path.with_suffix(".h5.tmp")
    # 在预览图中加入 dir_name 前缀，防止同名覆盖
    plot_filename = f"{dir_name}_{h5_path.stem}_preview.png"
    plot_path = plot_output_dir / plot_filename

    # 通知主进程添加任务条
    progress_queue.put(('start_file', unique_task_id))

    # 如果临时文件和预览图都存在，说明之前已成功处理。
    # 直接发送跳过信号并返回一个特殊标记，完全避免文件I/O。
    if temp_h5_path.exists() and plot_path.exists():
        progress_queue.put(('skipped', unique_task_id))
        # 返回一个带哨兵值(-1)的元组，主进程可以据此识别跳过的任务
        return (temp_h5_path, h5_path, -1, -1)

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

                # 立即释放大数组
                del px, py, pz, w
                gc.collect()

        # 显式垃圾回收
        gc.collect()

        if not all_energies_for_bins:
            progress_queue.put(('finish_file', unique_task_id))
            return None

        # Create global decision bins
        decision_bins, _ = _create_hybrid_bins(np.concatenate(all_energies_for_bins))
        del all_energies_for_bins # 释放能量数组
        gc.collect()

        # --- Pass 2: 压缩并写入 ---
        total_comp_particles = 0

        # 用于绘图的数据容器
        plot_data_raw = {'e': [], 'w': []}
        plot_data_comp = {'e': [], 'w': []}

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

                # --- 收集原始绘图数据 (降采样) ---
                current_energy = _calculate_kinetic_energy(px, py, pz)
                if weights.size > MAX_PLOT_PARTICLES:
                    idx = np.random.choice(weights.size, MAX_PLOT_PARTICLES, replace=False)
                    plot_data_raw['e'].append(current_energy[idx])
                    plot_data_raw['w'].append(weights[idx] * (weights.size / MAX_PLOT_PARTICLES)) # 修正权重以保持直方图高度
                else:
                    plot_data_raw['e'].append(current_energy)
                    plot_data_raw['w'].append(weights)

                # --- 压缩 ---
                c_px, c_py, c_pz, c_w = _compress_species_data(
                    px, py, pz, weights, decision_bins,
                    progress_queue, unique_task_id
                )

                # 释放原始数据
                del px, py, pz, weights, current_energy
                gc.collect()

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

                # --- 收集压缩绘图数据 (降采样) ---
                if c_w.size > 0:
                    c_energy = _calculate_kinetic_energy(c_px, c_py, c_pz)
                    if c_w.size > MAX_PLOT_PARTICLES:
                        idx = np.random.choice(c_w.size, MAX_PLOT_PARTICLES, replace=False)
                        plot_data_comp['e'].append(c_energy[idx])
                        plot_data_comp['w'].append(c_w[idx] * (c_w.size / MAX_PLOT_PARTICLES))
                    else:
                        plot_data_comp['e'].append(c_energy)
                        plot_data_comp['w'].append(c_w)

                    del c_energy

                # 释放压缩临时数据
                del c_px, c_py, c_pz, c_w
                gc.collect()

        # Plotting
        # 此时内存中只有降采样后的绘图数据，大大降低 OOM 风险
        final_e_raw = np.concatenate(plot_data_raw['e']) if plot_data_raw['e'] else np.array([])
        final_w_raw = np.concatenate(plot_data_raw['w']) if plot_data_raw['w'] else np.array([])
        final_e_comp = np.concatenate(plot_data_comp['e']) if plot_data_comp['e'] else np.array([])
        final_w_comp = np.concatenate(plot_data_comp['w']) if plot_data_comp['w'] else np.array([])

        _generate_comparison_plot(
            {'energy': final_e_raw, 'weights': final_w_raw},
            {'energy': final_e_comp, 'weights': final_w_comp},
            f"{dir_name}/{h5_path.name}", # 标题显示目录名
            plot_path
        )

        # 释放绘图数据
        del final_e_raw, final_w_raw, final_e_comp, final_w_comp, plot_data_raw, plot_data_comp
        gc.collect()

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
    preview_base_dir = Path(config.global_output_dir) / "slimmer_previews"
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

    # `files_to_replace` 存储所有最终需要执行替换操作的文件对
    files_to_replace: List[Tuple[Path, Path]] = []
    total_raw_this_run = 0
    total_comp_this_run = 0
    processed_files_count = 0
    skipped_files_count = 0
    error_files_count = 0

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
                            progress.update(tid, completed=100, total=100, description=f"[dim]{t_id_str} (Skipped)[/dim]")
                            sleep(0.05)
                            progress.update(tid, visible=False)
                        # 无论子任务条是否存在，都推进主进度
                        progress.advance(main_task_id)
                        finished_files += 1

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
                        # 只要有结果 (tmp文件存在)，就加入待替换列表
                        files_to_replace.append((res[0], res[1]))
                        
                        raw_count = res[2]
                        # 通过哨兵值-1判断是跳过还是新处理
                        if raw_count == -1:
                            skipped_files_count += 1
                        else:
                            processed_files_count += 1
                            total_raw_this_run += res[2]
                            total_comp_this_run += res[3]
                    else:
                        # fut.result() is None, 可能意味着子进程出错返回None
                        error_files_count += 1

                except Exception as e:
                    console.print(f"[red]Critical Future Error: {e}[/red]")
                    error_files_count += 1

    # 4. 统计与交互确认
    console.print("\n" + "=" * 50)
    console.print(f"[bold white]处理完成摘要:[/bold white]")
    console.print(f"  成功处理: {processed_files_count} 个文件 (本次运行)")
    console.print(f"  跳过已有: {skipped_files_count} 个文件 (来自之前运行)")
    if error_files_count > 0:
        console.print(f"  处理失败: {error_files_count} 个文件")
    console.print("=" * 50)

    # 只要有任何可替换的文件（无论是本次还是之前的），就进入确认流程
    if not files_to_replace:
        console.print("[yellow]没有任何可操作的文件，任务结束。[/yellow]")
        return

    # 如果本次运行确实处理了新文件，则显示压缩比等详细信息
    if processed_files_count > 0:
        ratio = total_raw_this_run / total_comp_this_run if total_comp_this_run > 0 else 1
        console.print(f"\n[bold]本次运行统计 ({processed_files_count} 个文件):[/bold]")
        console.print(f"  原始粒子: {total_raw_this_run:.2e}")
        console.print(f"  压缩粒子: {total_comp_this_run:.2e} (压缩比 {ratio:.1f}x)")

    # 最终确认，操作对象是所有找到的 .tmp 文件
    prompt_message = (
        f"\n[bold red]是否使用压缩版本替换总共 {len(files_to_replace)} 个文件 "
        f"({processed_files_count} 个新处理, {skipped_files_count} 个已存在)?[/bold red]"
    )
    if Confirm.ask(prompt_message):
        console.print("正在替换文件...")
        for tmp, orig in files_to_replace:
            # 确保临时文件真的存在，防止意外
            if tmp.exists():
                shutil.move(str(tmp), str(orig))
        console.print(f"[green]全部 {len(files_to_replace)} 个文件替换完成。[/green]")
    else:
        console.print("正在清理临时文件...")
        for tmp, _ in files_to_replace:
            if tmp.exists():
                os.remove(tmp)
        console.print("[yellow]操作已取消，所有临时文件已删除。[/yellow]")
