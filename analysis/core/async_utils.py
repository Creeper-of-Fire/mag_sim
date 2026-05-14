"""
异步并行工具。

提供 asyncify（同步→异步包装）和 gather_dict（并行字典收集），
用于将 @cached_op 函数按需转为可并行调度的 async callable。

全局注册表以函数为键分配 ThreadPoolExecutor，
同一函数始终共享同一个池，不同函数的池互不干扰，天然防止死锁。
"""

import asyncio
import functools
import os
import types
from concurrent.futures import ThreadPoolExecutor
from typing import Awaitable, Callable, TypeVar

T = TypeVar('T')

_POOL_SIZE = min(32, os.cpu_count() + 4)
_pools: dict = {}


def asyncify(fn: Callable[..., T]) -> Callable[..., Awaitable[T]]:
    """
    将同步函数转为 async callable，在函数专属线程池中执行。

    全局注册表以函数为键：同一函数（含同类实例的同名方法）共享同一个
    ThreadPoolExecutor，不同函数的池互不竞争。
    """
    key = fn if not isinstance(fn, types.MethodType) else (type(fn.__self__), fn.__func__)
    pool = _pools.setdefault(key, ThreadPoolExecutor(max_workers=_POOL_SIZE))

    @functools.wraps(fn)
    async def wrapper(*args, **kwargs):
        return await asyncio.get_running_loop().run_in_executor(
            pool, functools.partial(fn, *args, **kwargs)
        )
    return wrapper


async def gather_dict(tasks: dict) -> dict:
    """并行执行 dict[str, Awaitable]，返回 dict[str, result]。"""
    keys = list(tasks.keys())
    results = await asyncio.gather(*tasks.values())
    return dict(zip(keys, results))
