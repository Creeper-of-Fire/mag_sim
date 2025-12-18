import functools
import sys
from mpi4py import MPI
import builtins

comm = MPI.COMM_WORLD

# 备份原始的 print，以防万一以后想用（比如在 crash 的时候）
_original_print = builtins.print


def mpi_print(*args, **kwargs):
    """
    黑客版的 print。
    1. 只有 rank 0 会输出。
    2. 默认强制 flush=True，防止 MPI 死锁时看不到日志。
    """
    if comm.rank == 0:
        # 如果调用者没有指定 flush，我们强制设为 True
        if 'flush' not in kwargs:
            kwargs['flush'] = True

        return _original_print(*args, **kwargs)
    return None


def enable_mpi_print():
    """
    开启全局 print 劫持。
    调用此函数后，项目里所有的 print() 都会变成“仅主节点输出且自动刷新”。
    """
    builtins.print = mpi_print

def run_on_master(func):
    """
    一个简单的函数式包装器。
    如果当前是主进程，则执行 func。
    """
    if comm.rank == 0:
        return func()
    return None

def master_only(func):
    """
    装饰器：被装饰的方法或函数只会在 rank 0 执行。
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if comm.rank == 0:
            return func(*args, **kwargs)
        return None
    return wrapper


def mpi_barrier(message: str = None, verbose: bool = False):
    """
    MPI 进程同步屏障 (Barrier) 的包装器。

    Args:
        message: 可选。如果在同步前需要打印一条日志（仅主进程），传入此字符串。
                 这对于调试程序卡在哪个同步点非常有用。
        verbose: 是否打印“同步完成”的提示。
    """
    if message and comm.rank == 0:
        print(f"[同步] 正在等待所有进程: {message} ...")
        sys.stdout.flush()  # 确保日志立即输出，防止被缓冲阻塞

    comm.Barrier()

    if message and verbose and comm.rank == 0:
        print(f"[同步] {message} 完成。")
        sys.stdout.flush()

class Bunch(dict):
    """一个允许通过属性访问键的字典。"""

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"'Bunch' object has no attribute '{key}'")

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError:
            raise AttributeError(f"'Bunch' object has no attribute '{key}'")
