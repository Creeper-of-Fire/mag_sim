# analysis/core/cache.py
import functools
import hashlib
import inspect
import os
import pickle
from pathlib import Path
from typing import List, Any, Callable, Dict, Tuple

import lmdb

# 用于 isinstance 检查，不能放在 TYPE_CHECKING 中
from .simulation import SimulationRun
from .utils import console
from .utils import get_run_parameters


def cached_op(file_dep: str = "auto"):
    """
    智能缓存装饰器。

    缓存键值的组成:
    1. [Data] 依赖数据文件的指纹 (File Hash)
    2. [Args] 函数调用参数的指纹 (Args Hash)
    3. [Code] 定义该函数的源文件内容的指纹 (Module Hash)

    Args:
        file_dep: 依赖文件策略。
            "auto": 尝试从参数中寻找文件路径。如果找到，仅依赖该文件。
                    如果没找到，默认依赖参数文件 (run._param_file)。
            "particle": 强制依赖所有粒子文件 (run.particle_files)。
            "field": 强制依赖所有场文件 (run.field_files)。
            "all": 依赖所有文件。
    """

    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # 0. 智能定位 run 对象
            run_obj = None

            # 检查前两个位置参数（通常是 self 或者是单独的 run 参数）
            for arg in args[:2]:
                if isinstance(arg, SimulationRun):
                    run_obj = arg
                    break

            # 如果位置参数没找到，再从关键字参数里搜一把
            if not run_obj and 'run' in kwargs and isinstance(kwargs['run'], SimulationRun):
                run_obj = kwargs['run']

            if run_obj is None:
                raise ValueError(
                    f"cached_op 无法在方法 {func.__name__} 的参数中找到 SimulationRun 实例。"
                    f"请确保方法的参数包含 run 对象，或者该方法属于 SimulationRun 的子类。"
                )

            # 1. 绑定参数以方便检查
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()

            # 2. 确定依赖文件列表
            deps = []

            if file_dep == "auto":
                # 智能查找：参数里有没有看起来像文件路径的字符串？
                # 优先检查名为 'fpath', 'h5_filepath', 'file_path' 的参数
                for name, val in bound_args.arguments.items():
                    if name in ['fpath', 'h5_filepath', 'file_path', 'path'] and isinstance(val, (str, Path)) and os.path.exists(val):
                        deps = [str(val)]
                        break

                # 如果没找到特定文件，默认依赖模拟参数文件 (最保守策略)
                if not deps:
                    deps = [run_obj._param_file]

            elif file_dep == "particle":
                deps = run_obj.particle_files
            elif file_dep == "field":
                deps = run_obj.field_files
            elif file_dep == "all":
                deps = run_obj.particle_files + run_obj.field_files + [run_obj._param_file]

            # 2. 将调用上下文"物化"，交给 Cache 处理，杜绝任何 kwargs 签名冲突
            func_name = func.__name__

            return run_obj._cache.get(
                func_name=func_name,
                dependencies=deps,
                compute_func=func,
                run_obj=run_obj,  # 明确传递 Receiver 对象
                call_args=args,  # 明确传递参数元组
                call_kwargs=kwargs  # 明确传递字典
            )

        return wrapper

    return decorator


class SmartCache:
    """
    智能缓存管理器。
    负责根据 (源文件状态 + 函数参数) 生成唯一指纹，并通过 LMDB 持久化缓存。

    存储结构：一个 LMDB 文件对应一组 (函数名 + 源码哈希)，内部 key 为 (文件哈希 + 参数哈希)。
    版本号编码在缓存目录名中（如 .analysis_v3_cache），升级版本时更换目录即可。
    """

    CACHE_DIR_NAME = ".analysis_v3_cache"

    def __init__(self, run_path: Path):
        self.cache_dir = run_path / self.CACHE_DIR_NAME
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # 内存缓存：存储源代码文件的哈希，避免在一次运行中频繁读取硬盘上的 .py 文件
        self._source_file_hashes: Dict[str, str] = {}

        # LMDB 环境池：按 (函数名 + 源码哈希) 懒加载，避免重复打开
        self._envs: Dict[str, lmdb.Environment] = {}

    def _get_env(self, func_name: str, code_hash: str) -> lmdb.Environment:
        """获取或创建对应函数的 LMDB 环境。"""
        db_key = f"{func_name}_C{code_hash[:6]}"
        if db_key not in self._envs:
            db_path = str(self.cache_dir / f"{db_key}.lmdb")
            self._envs[db_key] = lmdb.open(db_path, map_size=64 * 1024 * 1024)
        return self._envs[db_key]

    def _get_files_hash(self, file_paths: List[str]) -> str:
        """计算源文件列表的状态哈希 (基于文件名和修改时间)。"""
        hasher = hashlib.md5()
        # 排序以保证列表顺序不影响哈希
        for p in sorted(file_paths):
            path = Path(p)
            if path.exists():
                # 混入文件名和修改时间
                stat = path.stat()
                # 使用 size 和 mtime 作为指纹
                fingerprint = f"{path.name}_{stat.st_size}_{stat.st_mtime}"
                hasher.update(fingerprint.encode('utf-8'))
            else:
                # 如果文件丢失，也记录下来
                hasher.update(f"{p}_MISSING".encode('utf-8'))
        return hasher.hexdigest()

    def _get_func_context_hash(self, func: Callable) -> str:
        """
        获取该函数定义的【源码内容】哈希，而不是整个文件的哈希。
        这样修改同一个文件内的绘图代码，不会导致计算函数的缓存失效。
        """
        try:
            # 1. 尝试获取函数的原始代码
            # getsource 会返回包括 @cached_op 在内的整个函数定义体
            source_code = inspect.getsource(func)

            # 2. [可选] 进一步清理代码：去除注释和首尾空格
            # 这样你改一下函数内部的注释，缓存依然有效
            lines = []
            for line in source_code.splitlines():
                stripped = line.strip()
                if stripped and not stripped.startswith('#'):
                    lines.append(line.rstrip())
            clean_source = "\n".join(lines)

            # 3. 计算源码哈希
            return hashlib.md5(clean_source.encode('utf-8')).hexdigest()

        except (TypeError, OSError) as e:
            # 如果是交互式环境或无法获取源码，退回到文件级哈希或文件名
            console.print(f"[dim]⚠ 无法获取函数 {func.__name__} 源码哈希 ({e})，尝试文件级检测。[/dim]")

            try:
                src_file = inspect.getfile(func)
                if os.path.exists(src_file):
                    if src_file in self._source_file_hashes:
                        return self._source_file_hashes[src_file]

                    with open(src_file, 'rb') as f:
                        file_hash = hashlib.md5(f.read()).hexdigest()
                    self._source_file_hashes[src_file] = file_hash
                    return file_hash
            except Exception:
                console.print(f"[yellow]⚠ 文件级哈希也失败，缓存将使用非唯一 key: {func.__name__}[/yellow]")

            return "FALLBACK_" + func.__name__

    def _get_args_hash(self, run_obj: Any, args: Tuple, kwargs: Dict) -> str:
        """
        计算参数哈希。
        这里 Cache 知道不应该序列化巨大的 run_obj 实例，只序列化物理参数(sim)和入参。
        """

        def sanitize(obj):
            # 1. 如果是 Receiver 对象本身，用占位符代替，不进行序列化
            if isinstance(obj, SimulationRun):
                return "__SIM_RUN_RECEIVER__"
            # 2. 如果是模块的 self (包含 run 方法且有名字)，也脱敏
            if hasattr(obj, 'run') and hasattr(obj, 'name'):
                return f"__MODULE_{obj.__class__.__name__}__"
            return obj

        # 清洗 args 和 kwargs，防止 pickle.dumps 爆炸
        safe_args = tuple(sanitize(a) for a in args)
        safe_kwargs = {k: sanitize(v) for k, v in kwargs.items()}
        safe_sim_params = {}
        if run_obj is not None:
            try:
                safe_sim_params = get_run_parameters(run_obj)
            except Exception as e:
                console.print(f"[yellow]⚠ 缓存键未包含模拟参数 ({e})，可能影响缓存准确性。[/yellow]")

        hash_payload = {
            'sim_params': safe_sim_params,
            'args': safe_args,
            'kwargs': safe_kwargs
        }
        try:
            return hashlib.md5(pickle.dumps(hash_payload)).hexdigest()
        except Exception as e:
            console.print(f"[yellow]⚠ 参数哈希计算失败: {e}，将使用随机哈希（不缓存）。[/yellow]")
            return "NO_CACHE_" + os.urandom(4).hex()

    def get(self,
            func_name: str,
            dependencies: List[str],
            compute_func: Callable,
            run_obj: Any,
            call_args: Tuple,
            call_kwargs: Dict) -> Any:
        """
        核心获取逻辑。显式接收 args 和 kwargs，彻底杜绝解包冲突。
        """
        # 1. 缓存指纹由 Cache 独立计算，关注点内聚
        files_hash = self._get_files_hash(dependencies)
        args_hash = self._get_args_hash(run_obj, call_args, call_kwargs)
        code_hash = self._get_func_context_hash(compute_func)

        # LMDB 文件 = 函数名 + 源码哈希，内部 key = 文件哈希 + 参数哈希
        env = self._get_env(func_name, code_hash)
        entry_key = f"F{files_hash[:6]}_A{args_hash[:6]}".encode()

        # 2. 命中缓存
        with env.begin() as txn:
            raw = txn.get(entry_key)
        if raw is not None:
            try:
                return pickle.loads(raw)
            except Exception as e:
                console.print(f"[yellow]⚠ 读取缓存 {entry_key.decode()} 失败 ({e})，将重新计算。[/yellow]")

        # 3. 未命中：由 Cache 负责触发真实计算
        console.print(f"[cyan]  -> 正在计算: {func_name} ...[/cyan]")
        result = compute_func(*call_args, **call_kwargs)

        # 4. 持久化
        if result is not None:
            try:
                with env.begin(write=True) as txn:
                    txn.put(entry_key, pickle.dumps(result))
                console.print(f"[green]     ✔ 缓存已保存: {entry_key.decode()}[/green]")
            except Exception as e:
                console.print(f"[red]✗ 保存缓存失败: {e}[/red]")

        return result
