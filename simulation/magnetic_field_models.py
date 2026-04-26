#!/usr/bin/env python3
import abc
import random
import typing
from enum import Enum

import numpy as np
import sympy
from mpi4py import MPI as mpi

from simulation.utils import Bunch, enable_mpi_print

comm = mpi.COMM_WORLD

enable_mpi_print()


class Dim(Enum):
    """模拟维度枚举"""
    D2 = 2
    D3 = 3


class InitialMagneticField(abc.ABC):
    """
    磁场模型的抽象基类 (ABC)。

    定义了所有初始磁场模型必须遵循的接口。
    职责:
    1. 接收必要的物理和几何参数。
    2. 构建磁场的 sympy 符号表达式。
    3. 提供可供 WarpX 使用的数值字符串和可供分析存档的 srepr 字符串。
    """

    def __init__(self, Lx: float, Ly: float, Lz: float, B_target_rms: float, dim: Dim = Dim.D3):
        self.Lx = Lx
        self.Ly = Ly
        self.Lz = Lz
        self.B_target_rms = B_target_rms
        self.dim = dim

        # 定义 sympy 符号
        self.x, self.y, self.z = sympy.symbols('x y z')

        # 将在 _build_expressions 中被子类填充
        self.Bx_expr: typing.Optional[sympy.Expr] = None
        self.By_expr: typing.Optional[sympy.Expr] = None
        self.Bz_expr: typing.Optional[sympy.Expr] = None

        # 构建表达式
        print(f"\n--- 构建初始磁场: {self.__class__.__name__} [{self.dim.name}] ---")
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
        if self.dim == Dim.D2:
            raise NotImplementedError("ABC 场是 3D 结构，不支持 2D 模拟。")

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
        # 注意: WarpX 2D 中，通常坐标是 x, z。Y 是 out-of-plane。
        # 标准 OT 涡定义在 x-y 平面。
        # 为了适配 WarpX 2D (xz平面)，我们将原本 y 的依赖映射到 z。

        if self.dim == Dim.D2:
            # 2D 模式: x->x, y->z (模拟平面的第二个维度)
            kz = 2 * sympy.pi / self.Lz
            self.Bx_expr = -self.B_target_rms * sympy.sin(kz * self.z)
            self.Bz_expr = self.B_target_rms * sympy.sin(2 * kx * self.x)
            self.By_expr = sympy.sympify(0.0)  # Out of plane component
        else:
            # 3D 模式: 标准定义
            ky = 2 * sympy.pi / self.Ly
            self.Bx_expr = -self.B_target_rms * sympy.sin(ky * self.y)
            self.By_expr = self.B_target_rms * sympy.sin(2 * kx * self.x)
            self.Bz_expr = sympy.sympify(0.0)

        print(f"  - 创建 Orszag-Tang 涡旋:")
        print(f"    - 维度: {self.dim.name}")
        print(f"    - 振幅 B0={self.B_target_rms:.2e} T")


class GaussianField(InitialMagneticField):
    """由多个高斯包叠加构成的随机磁场。"""

    def __init__(self, Lx: float, Ly: float, Lz: float, B_target_rms: float, dim: Dim,
                 d_e: float, NX: int, NY: int, NZ: int,
                 num_gaussians: int, gaussian_width_de_ratio: float):
        self.d_e = d_e
        self.NX = NX
        self.NY = NY
        self.NZ = NZ
        self.num_gaussians = num_gaussians
        self.gaussian_width_de_ratio = gaussian_width_de_ratio

        super().__init__(Lx, Ly, Lz, B_target_rms, dim)

    def _analytical_normalization(self) -> float:
        """
        使用解析方法计算单个高斯包所需的峰值振幅 (B_peak)。
        该方法基于统计假设：N个不相关源的总能量约等于N倍的单个源的能量。
        这避免了蒙特卡洛方法中由于随机破坏性干涉导致的归一化因子过大的问题。

        Returns:
            float: 单个高斯包应有的峰值磁场振幅 B_peak。
        """
        # 物理参数
        w = self.gaussian_width_de_ratio * self.d_e

        if self.dim == Dim.D3:
            # 3D: ∫ exp(-2r^2/w^2) dV = (pi/2)^(3/2) * w^3
            # 计算单个单位峰值高斯包对 B_rms^2 的贡献
            # B_rms_single^2 = (1/V) * ∫ B_peak^2 * exp(-2*(...)) dV
            # 当 B_peak=1 时, ∫ exp(-2*[(x/wx)^2 + (y/wy)^2 + (z/wz)^2]) dV
            # = ∫ exp(-2x^2/wx^2)dx * ∫ exp(-2y^2/wy^2)dy * ∫ exp(-2z^2/wz^2)dz
            # = sqrt(π*wx^2/2) * sqrt(π*wy^2/2) * sqrt(π*wz^2/2)
            # = (π/2)^(3/2) * wx * wy * wz
            V = self.Lx * self.Ly * self.Lz
            integral_B2_unit = (np.pi / 2.0) ** 1.5 * (w ** 3)
        else:
            # 2D (XZ平面): ∫ exp(-2(x^2+z^2)/w^2) dA = (pi/2) * w^2
            # 这里的 "Volume" V 实际上是模拟区域的面积 Area
            V = self.Lx * self.Lz
            integral_B2_unit = (np.pi / 2.0) * (w ** 2)

        B_rms_single_sq_unit = integral_B2_unit / V

        # 总 B_rms^2 是 N 个高斯包贡献之和 (假设不相关)
        # B_target_rms^2 = N * B_rms_single^2 = N * B_peak^2 * B_rms_single_sq_unit
        # 从此式求解 B_peak
        if self.num_gaussians == 0:
            return 0.0

        B_peak_sq = self.B_target_rms ** 2 / (self.num_gaussians * B_rms_single_sq_unit)
        B_peak = np.sqrt(B_peak_sq)

        print(f"  [分析归一化] 为 {self.num_gaussians} 个高斯包计算峰值振幅。")
        print(f"  [分析归一化] 目标均方根磁场 B_rms = {self.B_target_rms:.4e} T。")
        print(f"  [分析归一化] 计算得到的每个高斯包的峰值振幅 B_peak = {B_peak:.6f} T。")

        return B_peak

    def _build_expressions(self):
        # 物理参数
        w = self.gaussian_width_de_ratio * self.d_e

        final_params_list = None
        if comm.rank == 0:
            # 1. Rank 0 计算确定的峰值振幅
            B_peak = self._analytical_normalization()

            # 2. Rank 0 生成随机参数 (位置和方向)
            final_params_list = []
            for _ in range(self.num_gaussians):
                x0 = random.uniform(-self.Lx / 2.0, self.Lx / 2.0)

                if self.dim == Dim.D3:
                    y0 = random.uniform(-self.Ly / 2.0, self.Ly / 2.0)
                else:
                    y0 = 0.0  # 2D 模式下 y 坐标无意义（或者是 invariant 方向）

                z0 = random.uniform(-self.Lz / 2.0, self.Lz / 2.0)
                # 生成单位方向向量
                rand_vec = np.array([random.gauss(0, 1) for _ in range(3)])  # 使用高斯分布更均匀
                norm = np.linalg.norm(rand_vec)
                if norm < 1e-30:
                    norm_vec = np.array([1.0, 0.0, 0.0])  # 避免除以零
                else:
                    norm_vec = rand_vec / norm

                # 最终的峰值磁场向量
                b_vec = B_peak * norm_vec

                final_params_list.append((x0, y0, z0, *b_vec))

        # 3. 广播最终参数
        final_params_list = comm.bcast(final_params_list, root=0)

        # 4. 构建 SymPy 表达式
        bx_terms, by_terms, bz_terms = [], [], []
        for i, params in enumerate(final_params_list):
            x0, y0, z0, b_peak_x, b_peak_y, b_peak_z = params

            if self.dim == Dim.D3:
                # 3D Gaussian: exp(-(dx^2 + dy^2 + dz^2))
                r2_norm = ((self.x - x0) / w) ** 2 + ((self.y - y0) / w) ** 2 + ((self.z - z0) / w) ** 2
            else:
                # 2D Gaussian (Cylinder): exp(-(dx^2 + dz^2))
                # 注意：这里我们假设 WarpX 2D 是 X-Z 平面
                r2_norm = ((self.x - x0) / w) ** 2 + ((self.z - z0) / w) ** 2

            gaussian_expr = sympy.exp(-r2_norm)

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
        dim=config.dim
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
            d_e=config.d_e,
            NX=config.NX, NY=config.NY, NZ=config.NZ,
            num_gaussians=1 if field_type == "single_gaussian" else config.num_gaussians,
            gaussian_width_de_ratio=config.gaussian_width_de_ratio,
        )
        return GaussianField(**common_args, **gauss_args)
    else:
        raise ValueError(f"未知的磁场类型: {field_type}")
