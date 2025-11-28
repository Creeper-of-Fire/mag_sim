#!/usr/bin/env python3
import itertools
import random
import shutil
import typing
from pathlib import Path
import sympy
from sympy.parsing.sympy_parser import parse_expr

import dill
import numpy as np
from mpi4py import MPI as mpi
from pywarpx import callbacks, fields, libwarpx, picmi
from scipy.constants import e, m_e, c

if typing.TYPE_CHECKING:
    from pywarpx.Bucket import Bucket
    from config import SimulationParameters  # 用于类型提示

constants = picmi.constants
comm = mpi.COMM_WORLD


class PlasmaReconnection(object):
    def __init__(self, params: 'SimulationParameters', verbose: bool):
        self.p = params
        self.verbose = verbose
        self.output_dir = Path(self.p.output_dir)
        self.diags_output_dir = self.output_dir / "diags"

        self._calculate_derived_parameters()

        # 从 config 中读取磁场参数
        self.B_field_type = self.p.B_field_type
        self.num_gaussians = self.p.num_gaussians
        self.gaussian_width_L_ratio = self.p.gaussian_width_L_ratio

        self._setup_magnetic_field()

        comm.Barrier()

        if comm.rank == 0:
            print(f"主进程 (rank 0) 正在准备输出目录: {self.output_dir}")
            # is_dir() 是比 exists() 更精确的检查
            if self.output_dir.is_dir():
                print(f"警告: 输出目录 {self.output_dir} 已存在，将删除。")
                shutil.rmtree(self.output_dir)

            # parents=True 可以在父目录不存在时创建，mkdir 默认 exist_ok=False
            self.output_dir.mkdir(parents=True)

            with open(self.output_dir / "sim_parameters.dpkl", "wb") as f:
                # 只保存纯数据字典，而不是整个对象
                dill.dump(self.__dict__, f)
            print("主进程 (rank 0) 目录准备完毕。")

        comm.Barrier()

        self._print_summary()

        self.setup_run()

    def _calculate_derived_parameters(self):
        # 将所有参数从参数对象复制到实例属性
        self.B0 = self.p.B0
        self.n_plasma = self.p.n_plasma
        self.T_plasma = self.p.T_plasma_eV

        self.n_photon = self.p.n_photon_to_plasma_ratio * self.n_plasma

        self.beam_fraction = self.p.beam_fraction

        # 从动能计算归一化动量 u_drift
        self.beam_energy_eV = self.p.beam_energy_eV
        # 1. 将动能从 eV 转换为焦耳
        beam_ke_joules = self.beam_energy_eV * constants.q_e
        # 2. 计算洛伦兹因子 gamma = 1 + KE / (m*c^2)
        gamma_beam = 1.0 + beam_ke_joules / (constants.m_e * constants.c ** 2)
        # 3. 计算归一化动量 u = sqrt(gamma^2 - 1)
        self.beam_u_drift = np.sqrt(gamma_beam ** 2 - 1.0)

        self.LX = self.p.LX
        self.LY = self.p.LY
        self.LZ = self.p.LZ
        self.LT = self.p.LT
        self.DT = self.p.DT
        self.NX = self.p.NX
        self.NY = self.p.NY
        self.NZ = self.p.NZ
        self.NPPC = self.p.NPPC

        # 计算新的派生等离子体参量
        self.get_plasma_quantities()

        # 无论初始磁场如何，都使用基于等离子体热力学性质的普适尺度进行归一化

        # 1. 磁场归一化尺度: 基于磁能与总热能均分的磁场
        #    B_norm^2 / (2*mu0) = 2 * n_plasma * T_plasma_J
        total_thermal_energy_density = 2 * self.n_plasma * self.T_plasma_J
        self.B_norm = np.sqrt(2 * constants.mu0 * total_thermal_energy_density)

        # 2. 电流归一化尺度: 基于相对论热电流
        self.J_norm = self.n_plasma * constants.q_e * constants.c

        # 现在使用计算出的物理尺度来定义模拟域的绝对尺寸
        self.Lx = self.LX * self.d_e
        self.Ly = self.LY * self.d_e
        self.Lz = self.LZ * self.d_e
        self.dt = self.DT / self.w_pe  # dt based on w_pe

        self.total_steps = int(self.LT / self.DT)
        self.diag_steps = self.total_steps // 2000  # 每200步输出一次诊断
        self.diag_steps = max(1, self.diag_steps)

        self.Bx = f"{self.B0}"
        self.By = f"{self.B0}"
        self.Bz = f"{self.B0}"

        self.dx = self.Lx / self.NX
        self.dy = self.Ly / self.NY
        self.dz = self.Lz / self.NZ

    def _print_summary(self):
        # 打印更新后的等离子体参数
        if comm.rank == 0:
            print("--- Normalization Scales ---")
            print(f"\tB_norm (Equipartition Field) = {self.B_norm:.3e} T")
            print(f"\tJ_norm (Thermal Current)   = {self.J_norm:.3e} A/m^2")
            print("--- Independent Input Parameters ---")
            print(
                f"\tB0 = {self.B0:.2f} T\n"
                f"\tn0 = {self.n_plasma:.2e} m^-3 (per species)\n"
                f"\tT = {self.T_plasma * 1e-6:.2f} MeV\n"
                f"\tBeam Fraction = {self.beam_fraction * 100:.1f}%\n"
                f"\tBeam Kinetic Energy = {self.beam_energy_eV * 1e-6:.2f} MeV\n"
                f"\tBeam u_drift = {self.beam_u_drift:.2f}\n"
            )
            print("--- Derived Plasma Parameters ---")
            print(
                f"\td_e = {self.d_e:.3e} m (Electron Skin Depth)\n"
                f"\t1/w_pe = {1.0 / self.w_pe:.3e} s (Plasma Period)\n"
                f"\tMagnetization sigma = {self.sigma:.3f}\n"
                f"\tRelativistic Theta = {self.theta:.3f}\n"
            )
            print("--- Numerical Parameters ---")
            print(
                f"\tDomain = {self.Lx:.3e}m x {self.Ly:.3e}m x {self.Lz:.3e}m "
                f"({self.LX:.0f} d_e x {self.LY:.0f} d_e x {self.LZ:.0f} d_e)\n"
                f"\tGrid = {self.NX} x {self.NY} x {self.NZ}\n"
                f"\tdx = {self.Lx / self.NX:.3e} m, dy = {self.Ly / self.NY:.3e} m, dz = {self.Lz / self.NZ:.3e} m\n"
                f"\tdt = {self.dt:.3e} s ({self.DT:.2e} / w_pe)\n"
                f"\tTotal steps = {self.total_steps:d} (runtime = {self.total_steps * self.dt:.3e} s)\n"
            )

    def get_plasma_quantities(self):
        """
        为电子-正电子对等离子体计算派生参数。
        """
        # Plasma frequency (rad/s) for electrons (positrons have the same)
        self.w_pe = np.sqrt(self.n_plasma * constants.q_e ** 2 / (constants.m_e * constants.ep0))

        # Electron skin depth (m) - the new natural length scale
        self.d_e = constants.c / self.w_pe

        # Plasma temperature in Joules
        self.T_plasma_J = self.T_plasma * constants.q_e

        # Relativistic thermal parameter Theta = kT / (m_e * c^2)
        self.theta = self.T_plasma_J / (constants.m_e * constants.c ** 2)

        # Magnetization parameter (sigma)
        # Ratio of magnetic energy density to total plasma enthalpy density
        # For a pair plasma, enthalpy density is 2 * n * (kT + m_e*c^2)
        # Assuming gamma=4/3 for a relativistic gas, enthalpy is ~ 2 * n * 4kT
        # Here we use a more general form with rest mass energy.
        total_energy_density = 2 * self.n_plasma * (self.T_plasma_J + constants.m_e * constants.c ** 2)
        magnetic_energy_density = self.B0 ** 2 / (2.0 * constants.mu0)
        self.sigma = magnetic_energy_density / total_energy_density

    def check_fields(self):
        """回调函数，用于在每个诊断步保存场数据。"""
        step = self.simulation.extension.warpx.getistep(lev=0) - 1

        if not (step == 1 or step % self.diag_steps == 0):
            return

        # 获取场数据
        Ex = fields.ExFPWrapper()[...]
        Ey = fields.EyFPWrapper()[...]
        Ez = fields.EzFPWrapper()[...]

        # 使用归一化尺度
        Jey = fields.JyFPWrapper()[...] / self.J_norm
        Jy = fields.JyFPWrapper()[...] / self.J_norm
        Bx = fields.BxFPWrapper()[...] / self.B_norm
        By = fields.ByFPWrapper()[...] / self.B_norm
        Bz = fields.BzFPWrapper()[...] / self.B_norm

        if libwarpx.amr.ParallelDescriptor.MyProc() != 0:
            return

        # 将场数据保存到文件
        with open(self.diags_output_dir / f"fields/fields_{step:06d}.npz", "wb") as f:
            np.savez(f, Ex=Ex, Ey=Ey, Ez=Ez, Jey=Jey, Jy=Jy, Bx=Bx, By=By, Bz=Bz,
                     J_norm=self.J_norm, B_norm=self.B_norm)

    def create_collision_pairs(self, species_list: list, ndt: int, coulomb_log: float = None) -> list:
        """
        根据给定的物种列表，自动创建所有唯一的二进制碰撞对。
        这包括物种间的碰撞和物种内的自碰撞。

        Args:
            species_list: picmi.Species 对象的列表。
            ndt: 计算碰撞的频率（以时间步为单位）。
            coulomb_log: 库仑对数。如果为None，WarpX将自动计算。

        Returns:
            一个包含所有 picmi.CoulombCollisions 对象的列表。
        """
        collision_objects = []
        # itertools.combinations_with_replacement 会生成所有唯一的组合，包括 (A,A), (A,B), (B,B)
        for s1, s2 in itertools.combinations_with_replacement(species_list, 2):
            collision_name = f"coll_{s1.name}__{s2.name}"
            collision_pair = picmi.CoulombCollisions(
                name=collision_name,
                species=[s1, s2],
                CoulombLog=coulomb_log,
                ndt=ndt
            )
            collision_objects.append(collision_pair)
            if comm.rank == 0:
                print(f"  - Created collision pair: {collision_name}")

        return collision_objects

    def _setup_magnetic_field(self):
        """
        根据配置生成初始磁场表达式。
        此方法现在会并行创建两个版本：
        1. self.Bx, self.By, self.Bz: 给 WarpX 使用的、代入数值的字符串。
        2. self.Bx_sym_srepr, ...: 用于分析和存档的、可重建的 sympy 对象字符串表示。
        """
        # 注意: sympy 的引入会增加模拟环境的依赖
        if comm.rank == 0:
            print(f"\n--- 设置初始磁场 (使用 sympy 构建): 类型 = {self.B_field_type} ---")

        # 定义 sympy 符号
        x, y, z = sympy.symbols('x y z')

        if self.B_field_type == 'uniform':
            # 创建符号表达式 (尽管这里只是常数)
            Bx_expr = sympy.sympify(self.B0)
            By_expr = sympy.sympify(self.B0)
            Bz_expr = sympy.sympify(self.B0)

            if comm.rank == 0:
                print(f"  - 创建均匀磁场 B = ({self.B0}, {self.B0}, {self.B0}) T")

        elif self.B_field_type in ['single_gaussian', 'multi_gaussian']:
            num_gaussians = 1 if self.B_field_type == 'single_gaussian' else self.num_gaussians

            # 初始化空的符号表达式
            Bx_expr, By_expr, Bz_expr = sympy.sympify(0), sympy.sympify(0), sympy.sympify(0)

            # 在循环之前，创建空的 Python 列表
            bx_terms_list = []
            by_terms_list = []
            bz_terms_list = []

            # 计算高斯宽度 (物理单位)
            wx = self.gaussian_width_L_ratio * self.Lx
            wy = self.gaussian_width_L_ratio * self.Ly
            wz = self.gaussian_width_L_ratio * self.Lz

            # 在 rank 0 上生成所有随机参数，然后广播
            # 这确保所有 MPI 进程拥有完全相同的磁场定义
            if comm.rank == 0:
                params_list = []
                for _ in range(num_gaussians):
                    x0 = random.uniform(-self.Lx / 2.0, self.Lx / 2.0)
                    y0 = random.uniform(-self.Ly / 2.0, self.Ly / 2.0)
                    z0 = random.uniform(-self.Lz / 2.0, self.Lz / 2.0)
                    rand_vec = np.array([random.uniform(-1, 1) for _ in range(3)])
                    norm_vec = rand_vec / np.linalg.norm(rand_vec)
                    b_peak_x, b_peak_y, b_peak_z = self.B0 * norm_vec
                    params_list.append((x0, y0, z0, b_peak_x, b_peak_y, b_peak_z))
            else:
                params_list = None

            # 广播参数列表
            params_list = comm.bcast(params_list, root=0)

            for i, params in enumerate(params_list):
                x0, y0, z0, b_peak_x, b_peak_y, b_peak_z = params

                # 使用 sympy 创建符号化的高斯函数
                # 注意：参数 (x0, wx 等) 是 Python 的浮点数，但变量 (x,y,z) 是 sympy 符号
                gaussian_expr = sympy.exp(
                    -(((x - x0) / wx) ** 2 + ((y - y0) / wy) ** 2 + ((z - z0) / wz) ** 2)
                )

                # 将这个小表达式添加到列表中，而不是直接加到总表达式上
                bx_terms_list.append(b_peak_x * gaussian_expr)
                by_terms_list.append(b_peak_y * gaussian_expr)
                bz_terms_list.append(b_peak_z * gaussian_expr)

                if comm.rank == 0:
                    print(f"  - 添加第 {i + 1}/{num_gaussians} 个高斯场:")
                    print(f"    - 中心: ({x0:.2e}, {y0:.2e}, {z0:.2e}) m")
                    print(f"    - 峰值 B 矢量: ({b_peak_x:.2e}, {b_peak_y:.2e}, {b_peak_z:.2e}) T")

            # 循环结束后，一次性构建总表达式
            Bx_expr = sympy.Add(*bx_terms_list, evaluate=False)
            By_expr = sympy.Add(*by_terms_list, evaluate=False)
            Bz_expr = sympy.Add(*bz_terms_list, evaluate=False)

            # `evaluate=False` 参数建议在项数非常多时使用，
            # 它可以阻止 sympy 尝试进行一些可能很慢的自动简化，
            # 在我们的场景下，这通常是安全的，而且能进一步提速。
        else:
            raise ValueError(f"未知的磁场类型: {self.B_field_type}")

        # 1. 为 WarpX 生成数值字符串
        # sympy.sstr() 默认使用科学计数法，非常适合 WarpX 的解析器
        self.Bx = sympy.sstr(Bx_expr, full_prec=True)
        self.By = sympy.sstr(By_expr, full_prec=True)
        self.Bz = sympy.sstr(Bz_expr, full_prec=True)

        # 2. 为分析存档生成 srepr 字符串
        # srepr 是一个可以被 eval() 或 parse_expr() 重建为 sympy 对象的表示
        self.Bx_sym_srepr = sympy.srepr(Bx_expr)
        self.By_sym_srepr = sympy.srepr(By_expr)
        self.Bz_sym_srepr = sympy.srepr(Bz_expr)

        if comm.rank == 0:
            print("--- 磁场设置完毕 ---\n")
            # ==================== [调试输出开始] ====================
            print("\n" + "=" * 25)
            print("   DEBUG: 检查生成的表达式   ")
            print("=" * 25)

            # 设置一个阈值，只有当高斯项较少时才打印完整表达式
            DEBUG_PRINT_THRESHOLD = 10

            # 确定要检查的项数 (对于 uniform 场，项数为1)
            num_terms = 0
            if 'gaussian' in self.B_field_type:
                num_terms = num_gaussians
            elif self.B_field_type == 'uniform':
                num_terms = 1

            # 根据项数决定输出的详细程度
            if num_terms <= DEBUG_PRINT_THRESHOLD:
                print(f"\n[1] 表达式项数 ({num_terms}) 小于等于阈值 ({DEBUG_PRINT_THRESHOLD})，显示完整内容。")
                print("\n提供给 WarpX 的数值字符串 (以 Bx 为例):")
                print("    这种格式是纯数字和基本函数，WarpX可以直接解析。")
                print(f"    self.Bx = \"{self.Bx}\"")

                print("\n用于分析存档的 srepr 字符串 (以 Bx 为例):")
                print("    这种格式包含类型信息(如Float, Symbol), 可以精确地重建Sympy对象。")
                print(f"    self.Bx_sym_srepr = \"{self.Bx_sym_srepr}\"")
            else:
                print(f"\n[1] 表达式项数 ({num_terms}) 大于阈值 ({DEBUG_PRINT_THRESHOLD})，仅显示摘要。")
                print(f"    - WarpX 字符串 (Bx) 长度: {len(self.Bx)} 字符")
                print(f"    - srepr 字符串 (Bx) 长度: {len(self.Bx_sym_srepr)} 字符")
                print(f"    - WarpX 字符串开头预览: \"{self.Bx[:200]}...\"")
                print(f"    - srepr 字符串开头预览: \"{self.Bx_sym_srepr[:200]}...\"")

            # 验证步骤应该始终运行，因为它对于确保逻辑正确性至关重要
            print("\n[2] 验证 srepr 和 WarpX 字符串的一致性...")
            try:
                rebuilt_expr = sympy.parsing.sympy_parser.parse_expr(self.Bx_sym_srepr)
                rebuilt_sstr = sympy.sstr(rebuilt_expr, full_prec=True)

                if rebuilt_sstr == self.Bx:
                    print("    [验证成功] srepr 重建后的数值字符串与 WarpX 字符串完全一致。")
                else:
                    print("    [验证警告] srepr 重建后的数值字符串与 WarpX 字符串不一致！这是一个潜在的bug。")

            except Exception as e:
                print(f"    [验证失败] 无法从 srepr 重建表达式。错误: {e}")

            print("=" * 25)
            print("       调试结束       ")
            print("=" * 25 + "\n")
            # ===================== [调试输出结束] =====================

    def setup_run(self):
        """Setup simulation components."""
        #######################################################################
        # 设置粒子种类 (Species)
        #######################################################################

        # Calculate densities for thermal and beam components
        n_thermal = self.n_plasma * (1.0 - self.beam_fraction)
        n_beam = self.n_plasma * self.beam_fraction

        # Define four species: thermal and beam populations for both e- and e+
        self.electrons_thermal = picmi.Species(
            name="electrons_thermal",
            charge="-q_e",
            mass="m_e",
            initial_distribution=picmi.UniformDistribution(density=n_thermal),
            method='LLRK4'
        )
        self.electrons_beam = picmi.Species(
            name="electrons_beam",
            charge="-q_e",
            mass="m_e",
            initial_distribution=picmi.UniformDistribution(density=n_beam),
            method='LLRK4'
        )

        self.positrons_thermal = picmi.Species(
            name="positrons_thermal",
            charge="q_e",
            mass="m_e",
            initial_distribution=picmi.UniformDistribution(density=n_thermal),
            method='LLRK4'
        )
        self.positrons_beam = picmi.Species(
            name="positrons_beam",
            charge="q_e",
            mass="m_e",
            initial_distribution=picmi.UniformDistribution(density=n_beam),
            method='LLRK4'
        )

        self.photons = picmi.Species(
            name="photons",
            charge=0,
            mass=0
        )
        # TODO 在这里，我们没有为光子创建它的初始能量，包括电子也只是设置了初始速度……因为picmi没有实现这个功能，我再查查看怎么做。

        #######################################################################
        # 添加库仑碰撞
        #######################################################################

        print("\n--- Generating Binary Coulomb Collision Pairs ---")
        all_charged_species = [
            self.electrons_thermal, self.electrons_beam,
            self.positrons_thermal, self.positrons_beam
        ]

        all_collisions = self.create_collision_pairs(all_charged_species, ndt=25)

        simulation = picmi.Simulation(
            warpx_serialize_initial_conditions=True,
            verbose=0,
            warpx_collisions=all_collisions,  # 传递包含多个碰撞对象的列表
            warpx_reduced_diags_path=str(self.diags_output_dir),
            warpx_used_inputs_file=str(self.output_dir / "warpx_used_inputs")
        )
        self.simulation = simulation
        print(f"--- Total of {len(all_collisions)} collision pairs added to simulation. ---\n")

        #######################################################################
        # 设置几何、边界条件和时间步
        #######################################################################

        self.grid = picmi.Cartesian3DGrid(
            number_of_cells=[self.NX, self.NY, self.NZ],
            lower_bound=[-self.Lx / 2.0, -self.Ly / 2.0, -self.Lz / 2.0],
            upper_bound=[self.Lx / 2.0, self.Ly / 2.0, self.Lz / 2.0],
            lower_boundary_conditions=["periodic", "periodic", "periodic"],
            upper_boundary_conditions=["periodic", "periodic", "periodic"],
            lower_boundary_conditions_particles=["periodic", "periodic", "periodic"],
            upper_boundary_conditions_particles=["periodic", "periodic", "periodic"],
            warpx_max_grid_size=min(self.NX, self.NY, self.NZ),  # 调整最大网格大小
        )
        simulation.time_step_size = self.dt
        simulation.max_steps = self.total_steps
        simulation.current_deposition_algo = "direct"
        simulation.particle_shape = 1
        simulation.use_filter = False
        simulation.verbose = self.verbose

        #######################################################################
        # 设置场求解器和外场
        #######################################################################

        self.solver = picmi.ElectromagneticSolver(
            grid=self.grid,
            method='Yee',
            cfl=0.999
        )
        simulation.solver = self.solver

        B_ext = picmi.AnalyticInitialField(
            Bx_expression=self.Bx, By_expression=self.By, Bz_expression=self.Bz
        )
        simulation.add_applied_field(B_ext)

        #######################################################################
        # 添加粒子和布局
        #######################################################################

        # Calculate NPPC for each component, ensuring they are integers
        nppc_thermal = int(self.NPPC * (1.0 - self.beam_fraction))
        nppc_beam = int(self.NPPC * self.beam_fraction)

        # Create layouts for each of the four species
        layout_e_thermal = picmi.PseudoRandomLayout(
            grid=self.grid, n_macroparticles_per_cell=nppc_thermal
        )
        layout_e_beam = picmi.PseudoRandomLayout(
            grid=self.grid, n_macroparticles_per_cell=nppc_beam
        )
        layout_p_thermal = picmi.PseudoRandomLayout(
            grid=self.grid, n_macroparticles_per_cell=nppc_thermal
        )
        layout_p_beam = picmi.PseudoRandomLayout(
            grid=self.grid, n_macroparticles_per_cell=nppc_beam
        )
        layout_photons = picmi.PseudoRandomLayout(
            grid=self.grid, n_macroparticles_per_cell=self.NPPC
        )

        # Add all species to the simulation
        simulation.add_species(self.electrons_thermal, layout=layout_e_thermal)
        simulation.add_species(self.electrons_beam, layout=layout_e_beam)
        simulation.add_species(self.positrons_thermal, layout=layout_p_thermal)
        simulation.add_species(self.positrons_beam, layout=layout_p_beam)
        simulation.add_species(self.photons, layout=None)
        # TODO 目前，光子和其他粒子没有相互作用，如果有相互作用，它通常会通过电子对产生、康普顿散射之类的效应来提供“热化”或“阻力”，阻碍磁重联。

        #######################################################################
        # 添加诊断
        #######################################################################

        callbacks.installafterEsolve(self.check_fields)

        # Update diagnostics to include all new species
        all_particle_species_for_diags = [
            self.electrons_thermal, self.electrons_beam,
            self.positrons_thermal, self.positrons_beam
        ]
        # 探测平面诊断
        plane = picmi.ReducedDiagnostic(
            diag_type="FieldProbe",
            name="plane",
            period=self.diag_steps,
            path=str(self.diags_output_dir)+"/",
            extension="dat",
            probe_geometry="Plane",
            resolution=60,
            x_probe=0.0,
            y_probe=0.0,
            z_probe=0.0,
            detector_radius=self.d_e,
            target_up_x=0,
            target_up_y=0,
            target_up_z=1.0,
        )
        simulation.add_diagnostic(plane)

        # 粒子状态快照 (OpenPMD格式)
        particle_state_diag = picmi.ParticleDiagnostic(
            name="particle_states",
            period=self.diag_steps,
            species=all_particle_species_for_diags,
            data_list=["ux", "uy", "uz", "x", "y", "z", "weighting"],
            warpx_format='openpmd',
            warpx_openpmd_backend='h5',
            write_dir=str(self.diags_output_dir),
        )
        simulation.add_diagnostic(particle_state_diag)

        # 场状态快照 (OpenPMD格式)
        field_state_diag = picmi.FieldDiagnostic(
            name="field_states",
            grid=self.grid,
            period=self.diag_steps,
            data_list=["Bx", "By", "Bz", "Ex", "Ey", "Ez", "Jx", "Jy", "Jz"],
            warpx_format='openpmd',
            warpx_openpmd_backend='h5',
            write_dir=str(self.diags_output_dir),
        )
        simulation.add_diagnostic(field_state_diag)

        #######################################################################
        # 初始化模拟
        #######################################################################

        if comm.rank == 0:
            if Path.exists(Path(self.diags_output_dir)):
                shutil.rmtree(self.diags_output_dir)
            Path(self.diags_output_dir / "fields").mkdir(parents=True, exist_ok=True)

        simulation.initialize_inputs()

        self.between_initialize()

        simulation.initialize_warpx()

    def between_initialize(self):
        """
        在初始化模拟输入之后，初始化WarpX模拟之前，通过bucket操作准备移交给WarpX的数据。
        """
        self._set_initial_momenta()

    def _set_initial_momenta(self):
        # 设置初始动量分布
        theta_plasma = (self.T_plasma * e) / (m_e * c ** 2)
        u_drift = self.beam_u_drift

        print(f"\n--- Setting Initial Momentum Distributions ---")
        print(f"Thermal populations: Maxwell-Jüttner with Theta = {theta_plasma:.3f}")
        print(f"Beam populations: Counter-streaming with u_z = +/-{u_drift:.2f}")
        print(f"--------------------------------------------\n")

        # Thermal populations
        electrons_thermal_bucket: Bucket = self.electrons_thermal.species
        electrons_thermal_bucket.add_new_attr("momentum_distribution_type", "maxwell_juttner")
        electrons_thermal_bucket.add_new_attr("theta", theta_plasma)

        positrons_thermal_bucket: Bucket = self.positrons_thermal.species
        positrons_thermal_bucket.add_new_attr("momentum_distribution_type", "maxwell_juttner")
        positrons_thermal_bucket.add_new_attr("theta", theta_plasma)

        # Beam populations
        electrons_beam_bucket: Bucket = self.electrons_beam.species
        electrons_beam_bucket.add_new_attr("momentum_distribution_type", "constant")
        electrons_beam_bucket.add_new_attr("ux", 0.0)
        electrons_beam_bucket.add_new_attr("uy", 0.0)
        electrons_beam_bucket.add_new_attr("uz", u_drift)

        positrons_beam_bucket: Bucket = self.positrons_beam.species
        positrons_beam_bucket.add_new_attr("momentum_distribution_type", "constant")
        positrons_beam_bucket.add_new_attr("ux", 0.0)
        positrons_beam_bucket.add_new_attr("uy", 0.0)
        positrons_beam_bucket.add_new_attr("uz", -u_drift)

    def run_simulation(self):
        """执行模拟的时间步进循环。"""
        self.simulation.step()
