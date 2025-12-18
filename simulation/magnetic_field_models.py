#!/usr/bin/env python3
import abc
import random
import time
import typing

import numpy as np
import sympy
from mpi4py import MPI as mpi

from simulation.utils import Bunch, enable_mpi_print

comm = mpi.COMM_WORLD

enable_mpi_print()

class InitialMagneticField(abc.ABC):
    """
    磁场模型的抽象基类 (ABC)。

    定义了所有初始磁场模型必须遵循的接口。
    职责:
    1. 接收必要的物理和几何参数。
    2. 构建磁场的 sympy 符号表达式。
    3. 提供可供 WarpX 使用的数值字符串和可供分析存档的 srepr 字符串。
    """

    def __init__(self, Lx: float, Ly: float, Lz: float, B_target_rms: float):
        self.Lx = Lx
        self.Ly = Ly
        self.Lz = Lz
        self.B_target_rms = B_target_rms

        # 定义 sympy 符号
        self.x, self.y, self.z = sympy.symbols('x y z')

        # 将在 _build_expressions 中被子类填充
        self.Bx_expr: typing.Optional[sympy.Expr] = None
        self.By_expr: typing.Optional[sympy.Expr] = None
        self.Bz_expr: typing.Optional[sympy.Expr] = None

        # 构建表达式
        print(f"\n--- 构建初始磁场: {self.__class__.__name__} ---")
        self._build_expressions()
        print(f"--- 磁场构建完毕 ---\n")

        # 从表达式生成字符串
        self._generate_strings()

    @abc.abstractmethod
    def _build_expressions(self):
        """
        [子类必须实现] 构建 Bx, By, Bz 的 sympy 表达式。
        """
        raise NotImplementedError

    def _generate_strings(self):
        """
        将 sympy 表达式转换为 WarpX 和分析所需的字符串格式。
        """
        if self.Bx_expr is None or self.By_expr is None or self.Bz_expr is None:
            raise RuntimeError("Sympy expressions were not built before generating strings.")

        # 1. 为 WarpX 生成数值字符串
        self.Bx_str = sympy.sstr(self.Bx_expr, full_prec=True)
        self.By_str = sympy.sstr(self.By_expr, full_prec=True)
        self.Bz_str = sympy.sstr(self.Bz_expr, full_prec=True)

        # 2. 为分析存档生成 srepr 字符串
        self.Bx_srepr = sympy.srepr(self.Bx_expr)
        self.By_srepr = sympy.srepr(self.By_expr)
        self.Bz_srepr = sympy.srepr(self.Bz_expr)

    def debug_print(self):
        print("--- 磁场设置完毕 ---\n")
        # ==================== [调试输出开始] ====================
        print("\n" + "=" * 25)
        print("   DEBUG: 检查生成的表达式   ")
        print("=" * 25)

        # 设置一个阈值，只有当高斯项较少时才打印完整表达式
        DEBUG_PRINT_THRESHOLD = 10

        # 确定要检查的项数 (对于 uniform 场，项数为1)
        num_terms = 0
        if isinstance(self, GaussianField):
            num_terms = self.num_gaussians
        elif isinstance(self, UniformField):
            num_terms = 1

        # 根据项数决定输出的详细程度
        if num_terms <= DEBUG_PRINT_THRESHOLD:
            print(f"\n[1] 表达式项数 ({num_terms}) 小于等于阈值 ({DEBUG_PRINT_THRESHOLD})，显示完整内容。")
            print("\n提供给 WarpX 的数值字符串 (以 Bx 为例):")
            print("    这种格式是纯数字和基本函数，WarpX可以直接解析。")
            print(f"    self.Bx = \"{self.Bx_str}\"")

            print("\n用于分析存档的 srepr 字符串 (以 Bx 为例):")
            print("    这种格式包含类型信息(如Float, Symbol), 可以精确地重建Sympy对象。")
            print(f"    self.Bx_srepr = \"{self.Bx_srepr}\"")
        else:
            print(f"\n[1] 表达式项数 ({num_terms}) 大于阈值 ({DEBUG_PRINT_THRESHOLD})，仅显示摘要。")
            print(f"    - WarpX 字符串 (Bx) 长度: {len(self.Bx_str)} 字符")
            print(f"    - srepr 字符串 (Bx) 长度: {len(self.Bx_srepr)} 字符")
            print(f"    - WarpX 字符串开头预览: \"{self.Bx_str[:200]}...\"")
            print(f"    - srepr 字符串开头预览: \"{self.Bx_srepr[:200]}...\"")

        # 验证步骤应该始终运行，因为它对于确保逻辑正确性至关重要
        print("\n[2] 验证 srepr 和 WarpX 字符串的一致性...")
        try:
            rebuilt_expr = sympy.parsing.sympy_parser.parse_expr(self.Bx_srepr)
            rebuilt_sstr = sympy.sstr(rebuilt_expr, full_prec=True)

            if rebuilt_sstr == self.Bx_str:
                print("    [验证成功] srepr 重建后的数值字符串与 WarpX 字符串完全一致。")
            else:
                print("    [验证警告] srepr 重建后的数值字符串与 WarpX 字符串不一致！这是一个潜在的bug。")

        except Exception as e:
            print(f"    [验证失败] 无法从 srepr 重建表达式。错误: {e}")

        print("=" * 25)
        print("       调试结束       ")
        print("=" * 25 + "\n")
        # ===================== [调试输出结束] =====================


class UniformField(InitialMagneticField):
    """均匀磁场模型。"""

    def _build_expressions(self):
        print(f"  - 创建均匀磁场 B_rms = {self.B_target_rms:.3e} T")
        # 对于均匀场，B_rms 就是每个分量的值（如果方向是(1,1,1)的话，需要调整，但这里简化）
        # 假设 B0 就是目标 B_target_rms, 且 Bx=By=Bz=B0
        self.Bx_expr = sympy.sympify(self.B_target_rms)
        self.By_expr = sympy.sympify(self.B_target_rms)
        self.Bz_expr = sympy.sympify(self.B_target_rms)


class ABCField(InitialMagneticField):
    """ABC (Arnold-Beltrami-Childress) 磁流体湍流种子场。"""

    def _build_expressions(self):
        kx = 2 * sympy.pi / self.Lx
        ky = 2 * sympy.pi / self.Ly
        kz = 2 * sympy.pi / self.Lz

        A, B, C = self.B_target_rms, self.B_target_rms, self.B_target_rms

        self.Bx_expr = A * sympy.sin(kz * self.z) + C * sympy.cos(ky * self.y)
        self.By_expr = B * sympy.sin(kx * self.x) + A * sympy.cos(kz * self.z)
        self.Bz_expr = C * sympy.sin(ky * self.y) + B * sympy.cos(kx * self.x)

        print(f"  - 创建 ABC 场 (3D湍流种子):")
        print(f"    - kx={float(kx):.2e}, ky={float(ky):.2e}, kz={float(kz):.2e}")
        print(f"    - 振幅 B0={self.B_target_rms:.2e} T")


class OrszagTangField(InitialMagneticField):
    """Orszag-Tang 涡旋 (2D MHD 湍流测试)。"""

    def _build_expressions(self):
        kx = 2 * sympy.pi / self.Lx
        ky = 2 * sympy.pi / self.Ly

        self.Bx_expr = -self.B_target_rms * sympy.sin(ky * self.y)
        self.By_expr = self.B_target_rms * sympy.sin(2 * kx * self.x)
        self.Bz_expr = sympy.sympify(0.0)

        print(f"  - 创建 Orszag-Tang 涡旋 (2D湍流种子):")
        print(f"    - 振幅 B0={self.B_target_rms:.2e} T")


class GaussianField(InitialMagneticField):
    """由多个高斯包叠加构成的随机磁场。"""

    def __init__(self, Lx: float, Ly: float, Lz: float, B_target_rms: float,
                 num_gaussians: int, gaussian_width_L_ratio: float):
        self.num_gaussians = num_gaussians
        self.gaussian_width_L_ratio = gaussian_width_L_ratio
        super().__init__(Lx, Ly, Lz, B_target_rms)

    def _monte_carlo_normalization(self, params_list, num_samples=50000):
        """
        使用蒙特卡洛积分计算叠加场的均方根 (RMS) 值。
        用于将随机场的能量归一化到全局设定值。

        Args:
            params_list: List of (x0, y0, z0, bx, by, bz)
            num_samples: 采样点数量，越多越准，50000对于1000个高斯包通常足够且秒级完成。

        Returns:
            scaling_factor: 用于乘以峰值磁场的系数
        """
        t0 = time.time()

        # 1. 在模拟域内随机采样点
        # shape: (num_samples, 3)
        pts = np.random.rand(num_samples, 3)
        pts[:, 0] = pts[:, 0] * self.Lx - self.Lx / 2.0
        pts[:, 1] = pts[:, 1] * self.Ly - self.Ly / 2.0
        pts[:, 2] = pts[:, 2] * self.Lz - self.Lz / 2.0

        # 高斯宽度
        wx = self.gaussian_width_L_ratio * self.Lx
        wy = self.gaussian_width_L_ratio * self.Ly
        wz = self.gaussian_width_L_ratio * self.Lz

        # 初始化累计磁场
        Bx_tot = np.zeros(num_samples)
        By_tot = np.zeros(num_samples)
        Bz_tot = np.zeros(num_samples)

        # 2. 向量化计算所有高斯包的贡献
        # 为了避免内存溢出 (如果 params_list 很大)，可以分批处理
        # 将 params 转换为 numpy 数组以便广播
        # params_arr shape: (N_gaussians, 6) -> x0, y0, z0, bx, by, bz
        params_arr = np.array(params_list)

        centers = params_arr[:, 0:3]  # (N_g, 3)
        b_vecs = params_arr[:, 3:6]  # (N_g, 3)

        # 分块处理高斯包，防止构建 (50000, 1000, 3) 的大矩阵撑爆内存
        chunk_size = 100  # 每次处理100个高斯包
        num_gaussians = len(params_list)

        for i in range(0, num_gaussians, chunk_size):
            end = min(i + chunk_size, num_gaussians)

            # 当前批次的中心和向量
            # shape: (batch_size, 3)
            c_batch = centers[i:end]
            b_batch = b_vecs[i:end]

            # 利用广播计算距离: (num_samples, 1, 3) - (1, batch_size, 3)
            # diff shape: (num_samples, batch_size, 3)
            diff = pts[:, np.newaxis, :] - c_batch[np.newaxis, :, :]

            # 处理周期性边界条件 (Periodic BCs)
            # 如果点和中心跨越了边界，距离应该取最短路径
            # dx = dx - L * round(dx/L)
            diff[:, :, 0] -= self.Lx * np.round(diff[:, :, 0] / self.Lx)
            diff[:, :, 1] -= self.Ly * np.round(diff[:, :, 1] / self.Ly)
            diff[:, :, 2] -= self.Lz * np.round(diff[:, :, 2] / self.Lz)

            # 计算高斯包络
            # arg shape: (num_samples, batch_size)
            arg = (diff[:, :, 0] / wx) ** 2 + (diff[:, :, 1] / wy) ** 2 + (diff[:, :, 2] / wz) ** 2
            envelope = np.exp(-arg)

            # 累加磁场
            # b_batch shape (batch_size, 3)
            # envelope shape (num_samples, batch_size)
            # result increments shape (num_samples)
            Bx_tot += np.dot(envelope, b_batch[:, 0])
            By_tot += np.dot(envelope, b_batch[:, 1])
            Bz_tot += np.dot(envelope, b_batch[:, 2])

        # 3. 计算均方值 B^2_avg
        B2_samples = Bx_tot ** 2 + By_tot ** 2 + Bz_tot ** 2
        B2_avg = np.mean(B2_samples)
        B_rms_calculated = np.sqrt(B2_avg)

        # 4. 计算缩放因子
        # 我们希望 B_rms_calculated * scale = B_target_rms
        scaling_factor = self.B_target_rms / (B_rms_calculated + 1e-30)

        t1 = time.time()
        print(f"  [蒙特卡洛] 为 {num_gaussians} 个高斯包采样了 {num_samples} 个点。")
        print(f"  [蒙特卡洛] 原始均方根磁场 B = {B_rms_calculated:.4e} T。")
        print(f"  [蒙特卡洛] 目标均方根磁场 B = {self.B_target_rms:.4e} T。")
        print(f"  [蒙特卡洛] 计算得到的缩放因子 = {scaling_factor:.6f} (耗时: {t1 - t0:.3f}秒)")

        return scaling_factor

    def _build_expressions(self):
        # 物理参数
        wx = self.gaussian_width_L_ratio * self.Lx
        wy = self.gaussian_width_L_ratio * self.Ly
        wz = self.gaussian_width_L_ratio * self.Lz

        final_params_list = None
        if comm.rank == 0:
            # 1. Rank 0 生成随机参数
            raw_params_list = []
            for _ in range(self.num_gaussians):
                x0 = random.uniform(-self.Lx / 2.0, self.Lx / 2.0)
                y0 = random.uniform(-self.Ly / 2.0, self.Ly / 2.0)
                z0 = random.uniform(-self.Lz / 2.0, self.Lz / 2.0)
                rand_vec = np.array([random.uniform(-1, 1) for _ in range(3)])
                norm_vec = rand_vec / (np.linalg.norm(rand_vec) + 1e-30)
                raw_params_list.append((x0, y0, z0, *norm_vec))

            # 2. Rank 0 计算归一化系数
            print(f"  - 正在计算 {self.num_gaussians} 个叠加高斯包的全局能量归一化系数...")
            scaling_factor = self._monte_carlo_normalization(raw_params_list)

            # 3. 应用缩放因子
            final_params_list = [(p[0], p[1], p[2], p[3] * scaling_factor, p[4] * scaling_factor, p[5] * scaling_factor) for p in raw_params_list]

        # 4. 广播最终参数
        final_params_list = comm.bcast(final_params_list, root=0)

        # 5. 构建 SymPy 表达式
        bx_terms, by_terms, bz_terms = [], [], []
        for i, params in enumerate(final_params_list):
            x0, y0, z0, b_peak_x, b_peak_y, b_peak_z = params
            gaussian_expr = sympy.exp(-(((self.x - x0) / wx) ** 2 + ((self.y - y0) / wy) ** 2 + ((self.z - z0) / wz) ** 2))
            bx_terms.append(b_peak_x * gaussian_expr)
            by_terms.append(b_peak_y * gaussian_expr)
            bz_terms.append(b_peak_z * gaussian_expr)

        self.Bx_expr = sympy.Add(*bx_terms, evaluate=False)
        self.By_expr = sympy.Add(*by_terms, evaluate=False)
        self.Bz_expr = sympy.Add(*bz_terms, evaluate=False)


def magnetic_field_factory(config: Bunch) -> InitialMagneticField:
    """
    根据配置字典创建并返回相应的磁场模型实例。
    这是一个工厂函数，它将创建逻辑与主模拟类解耦。
    """
    field_type = config.B_field_type

    # 提取通用参数
    common_args = Bunch(
        Lx=config.Lx,
        Ly=config.Ly,
        Lz=config.Lz,
        B_target_rms=config.B_target_rms,
    )

    if field_type == "uniform":
        return UniformField(**common_args)
    elif field_type == "abc":
        return ABCField(**common_args)
    elif field_type == "orszag_tang":
        return OrszagTangField(**common_args)
    elif field_type in ["single_gaussian", "multi_gaussian"]:
        # 提取高斯场特定参数
        gauss_args = Bunch(
            num_gaussians=1 if field_type == "single_gaussian" else config.num_gaussians,
            gaussian_width_L_ratio=config.gaussian_width_L_ratio,
        )
        return GaussianField(**common_args, **gauss_args)
    else:
        raise ValueError(f"未知的磁场类型: {field_type}")
