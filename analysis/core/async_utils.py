"""
异步并行工具。

提供 asyncify（同步→异步包装）和 gather_dict（并行字典收集），
用于将 @cached_op 函数按需转为可并行调度的 async callable。
"""

import asyncio
import functools
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Awaitable, Callable, TypeVar

T = TypeVar('T')

_pool = ThreadPoolExecutor(max_workers=min(32, os.cpu_count() + 4))


def asyncify(fn: Callable[..., T]) -> Callable[..., Awaitable[T]]:
    """
    将同步函数转为 async callable，在全局线程池中执行。

    原函数的 @cached_op 装饰器完全不受影响——线程池执行的是已包装的缓存函数，
    缓存命中时直接返回，未命中时才真正计算。
    """
    @functools.wraps(fn)
    async def wrapper(*args, **kwargs):
        return await asyncio.get_running_loop().run_in_executor(
            _pool, functools.partial(fn, *args, **kwargs)
        )
    return wrapper


async def gather_dict(tasks: dict) -> dict:
    """并行执行 dict[str, Awaitable]，返回 dict[str, result]。"""
    keys = list(tasks.keys())
    results = await asyncio.gather(*tasks.values())
    return dict(zip(keys, results))
