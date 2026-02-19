# analysis/core/cache.py
import functools
import hashlib
import inspect
import os
from pathlib import Path
from typing import List, Any, Callable, Dict

import dill

from .utils import console


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
        def wrapper(run, *args, **kwargs):
            # 1. 绑定参数以方便检查
            sig = inspect.signature(func)
            bound_args = sig.bind(run, *args, **kwargs)
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
                    deps = [run._param_file]

            elif file_dep == "particle":
                deps = run.particle_files
            elif file_dep == "field":
                deps = run.field_files
            elif file_dep == "all":
                deps = run.particle_files + run.field_files + [run._param_file]

            # 3. 生成缓存 Key (函数名 + 参数)
            # 注意：我们将 run 对象排除在哈希之外，因为它包含很多动态状态
            # 但我们需要 run.sim (参数对象) 参与哈希，因为它决定了物理常量

            # 构造一个用于哈希的参数字典，排除 run，加入 sim_params 的指纹
            hash_args = dict(bound_args.arguments)
            hash_args.pop('run', None)  # 移除 run
            hash_args['__sim_params_hash__'] = run._cache._get_args_hash((run.sim,), {})  # 加入物理参数指纹

            # 4. 调用核心缓存逻辑
            return run._cache.get(
                func_name=f"{getattr(func, '__module__', 'unknown')}.{func.__name__}",
                dependencies=deps,
                compute_func=func,
                # 传递原始参数
                **bound_args.arguments
            )

        return wrapper

    return decorator


class SmartCache:
    """
    智能缓存管理器。
    负责根据 (源文件状态 + 函数参数) 生成唯一指纹，并管理磁盘缓存。
    """

    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        # 版本号，如果修改了核心数据结构，修改此值强制所有缓存失效
        self.API_VERSION = "v2.0"

        # 内存缓存：存储源代码文件的哈希，避免在一次运行中频繁读取硬盘上的 .py 文件
        self._source_file_hashes: Dict[str, str] = {}

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

    def _get_args_hash(self, args: tuple, kwargs: Dict) -> str:
        """计算函数参数的哈希。"""
        # 使用 dill 序列化参数，因为它比 pickle/json 更能处理复杂对象（如 numpy array 或 lambda）
        try:
            serialized = dill.dumps((args, kwargs))
            return hashlib.md5(serialized).hexdigest()
        except Exception as e:
            console.print(f"[yellow]⚠ 参数哈希计算失败: {e}，将使用随机哈希（不缓存）。[/yellow]")
            return "NO_CACHE_" + os.urandom(4).hex()

    def _get_func_context_hash(self, func: Callable) -> str:
        """
        获取定义该函数的**源文件**的哈希。
        """
        try:
            # 获取定义该函数的文件路径
            src_file = inspect.getfile(func)

            # 如果不是物理文件 (比如在 Jupyter 或 repl 中)，回退到函数名
            if not os.path.exists(src_file) or not os.path.isfile(src_file):
                return "INTERACTIVE_" + func.__name__

            # 检查内存缓存
            if src_file in self._source_file_hashes:
                return self._source_file_hashes[src_file]

            # 读取文件内容并哈希
            with open(src_file, 'rb') as f:
                content = f.read()
                file_hash = hashlib.md5(content).hexdigest()

            # 存入内存缓存
            self._source_file_hashes[src_file] = file_hash
            return file_hash

        except Exception as e:
            console.print(f"[dim]⚠ 无法获取源码文件哈希 ({e})，代码变更检测失效。[/dim]")
            return "UNKNOWN_SOURCE"

    def get(self,
            func_name: str,
            dependencies: List[str],
            compute_func: Callable,
            *args,
            **kwargs) -> Any:
        """
        获取数据的通用入口。

        Args:
            func_name (str): 功能标识符 (例如 'energy_evolution', 'spectrum_hist')
            dependencies (List[str]): 此计算依赖的源文件列表 (用于检查过期)
            compute_func (Callable): 如果缓存未命中，调用的计算函数
            *args, **kwargs: 传递给 compute_func 的参数
        """
        # 1. 计算指纹
        files_hash = self._get_files_hash(dependencies)
        args_hash = self._get_args_hash(args, kwargs)
        code_hash = self._get_func_context_hash(compute_func)

        # 组合成最终的 Cache Key (UUID)
        # 格式: {func_name}_{files_hash_short}_{args_hash_short}.dill
        cache_filename = (
            f"{func_name}_{self.API_VERSION}_"
            f"F{files_hash[:6]}_"  # File hash
            f"A{args_hash[:6]}_"  # Args hash
            f"C{code_hash[:6]}"  # Code hash
            f".dill"
        )
        cache_path = self.cache_dir / cache_filename

        # 2. 检查缓存是否存在
        if cache_path.exists():
            try:
                # console.print(f"[dim]  -> Hit Cache: {cache_filename}[/dim]")
                with open(cache_path, "rb") as f:
                    return dill.load(f)
            except Exception as e:
                console.print(f"[yellow]⚠ 读取缓存 {cache_filename} 失败 ({e})，将重新计算。[/yellow]")

        # 3. 缓存未命中 (Miss) -> 计算
        console.print(f"[cyan]  -> 正在计算: {func_name} ...[/cyan]")
        result = compute_func(*args, **kwargs)

        # 4. 保存缓存
        if result is not None:
            try:
                # 先写入临时文件再重命名，原子操作防止写入中断导致文件损坏
                temp_path = cache_path.with_suffix(".tmp")
                with open(temp_path, "wb") as f:
                    dill.dump(result, f)
                temp_path.replace(cache_path)
                console.print(f"[green]     ✔ 缓存已保存: {cache_filename}[/green]")
            except Exception as e:
                console.print(f"[red]✗ 保存缓存失败: {e}[/red]")

        return result
