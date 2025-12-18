#!/usr/bin/env python3
import inspect
import typing

import numpy as np
from mpi4py import MPI as mpi
from pywarpx import picmi
from scipy.constants import e, m_e, c

import itertools
from simulation.io_manager import IOManager
from simulation.magnetic_field_models import InitialMagneticField, magnetic_field_factory
from simulation.utils import Bunch, mpi_barrier, enable_mpi_print

if typing.TYPE_CHECKING:
    from pywarpx.Bucket import Bucket
    from simulation.config import SimulationParameters  # 用于类型提示

constants = picmi.constants
comm = mpi.COMM_WORLD

enable_mpi_print()


class SpeciesWrapper:
    """
    轻量级粒子包装器（使用 SimpleNamespace）。

    职责：
    1. 存储 picmi.Species 的初始化参数 (picmi_kwargs)。
    2. 存储布局配置。
    3. 使用 Bunch 存储需要通过 Bucket 注入的参数 (bucket_params)。
    4. 管理生命周期：先创建 picmi 对象，待 simulation 初始化后注入 bucket 参数。
    """

    def __init__(self, name: str,
                 layout_config: typing.Any = None,
                 bucket_params: Bunch = None,
                 **picmi_kwargs):
        self.name = name
        self.picmi_kwargs = picmi_kwargs
        self.picmi_kwargs['name'] = name

        self.layout_config = layout_config

        # 如果未提供，则初始化为空的 Bunch
        self.bucket_params = bucket_params if bucket_params is not None else Bunch()

        self.instance: typing.Optional['picmi.Species'] = None

    def initialize_picmi_object(self) -> 'picmi.Species':
        """实例化 picmi.Species 对象"""
        self.instance = picmi.Species(**self.picmi_kwargs)
        return self.instance

    def get_layout(self, grid: 'picmi.Cartesian3DGrid') -> typing.Any:
        """
        根据 layout_config 动态生成或返回布局对象。
        """
        if self.layout_config is None:
            return None

        # 情况1: layout_config 是一个工厂函数/lambda
        if callable(self.layout_config):
            # 智能检查 lambda 是否需要 grid 参数
            sig = inspect.signature(self.layout_config)
            if 'grid' in sig.parameters:
                return self.layout_config(grid=grid)
            else:
                # 如果工厂不需要 grid，直接调用
                return self.layout_config()

        # 情况2: 假设 layout_config 是一个预先创建好的对象。
        # 我们不检查它的类型，直接返回。如果它不是一个有效的布局对象，
        # simulation.add_species(...) 会在稍后抛出错误，这是正确的行为。
        return self.layout_config

    def apply_bucket_attributes(self):
        """
        将 bucket 参数注入到底层。
        必须在 simulation.initialize_inputs() 之后调用。
        """
        # 检查 bucket_params 是否有任何属性
        if not self.bucket_params:
            return

        if self.instance is None:
            raise RuntimeError(f"粒子种类 {self.name} 尚未初始化。")

        bucket: Bucket = self.instance.species

        bucket_dict = self.bucket_params
        print(f"  -> 正在应用 bucket 属性至 '{self.name}': {list(bucket_dict.keys())}")

        for key, value in bucket_dict.items():
            bucket.add_new_attr(key, value)


class PlasmaReconnection(object):
    def __init__(self, params: 'SimulationParameters', verbose: bool):
        self.p = params
        self.verbose = verbose
        self.io = IOManager(self.p.output_dir)

        self._calculate_derived_parameters()

        self._init_magnetic_field()

        mpi_barrier("磁场构建完毕，准备文件IO")

        # 准备目录并保存参数
        self.io.prepare_directories(overwrite=True)
        # 准备要保存的参数字典
        self._archive_parameters()

        mpi_barrier("文件系统准备完毕")

        self._print_summary()

        self.setup_run()

    @staticmethod
    def get_plasma_quantities(n_plasma, T_plasma, p_target_sigma=None, target_B0=None):
        """
        为电子-正电子对等离子体计算派生参数。
        并确定目标磁场强度。
        """
        # 等离子体频率和趋肤深度 (rad/s)
        w_pe = np.sqrt(n_plasma * constants.q_e ** 2 / (constants.m_e * constants.ep0))
        d_e = constants.c / w_pe

        # 温度 (Joules) 和 相对论参数 Theta
        T_plasma_J = T_plasma * constants.q_e
        theta = T_plasma_J / (constants.m_e * constants.c ** 2)

        # --- 能量密度计算 (Energy Density Design) ---

        # 1. 粒子热焓密度 (Particle Enthalpy Density)
        # 对于相对论对等离子体，能量密度 U ~ 2 * n * (m_e*c^2 + \hat{\gamma} * T / (gamma - 1))
        # 这里做一个简化估算：包含静止质量能和热能
        # Enthalpy Density w = U + P. 在 sigma 定义中通常使用 w.
        # 对于相对论气体 (gamma=4/3), w = 4 * P = 4 * n * kT.
        # 对于非相对论 (gamma=5/3), w = n * m * c^2 + 5/2 n * kT.
        # 我们这里采用包含静止质量的通用形式:
        particle_energy_density = 2 * n_plasma * (constants.m_e * constants.c ** 2 + 3.0 * T_plasma_J)

        # 2. 确定目标磁化率 (Target Sigma)
        # 如果 config 中定义了 target_sigma，则优先使用；否则根据 B0 计算当前的 sigma
        # 假设我们在 SimulationParameters (self.p) 中添加了 target_sigma
        # 如果没有，我们默认使用 B0 对应的 sigma 作为目标
        if p_target_sigma and p_target_sigma > 0:
            target_sigma = p_target_sigma
            # 倒推需要的 B_rms
            target_magnetic_energy = target_sigma * particle_energy_density
            B_target_rms = np.sqrt(2 * constants.mu0 * target_magnetic_energy)
            print(f"--- 能量设计: 目标 Sigma = {target_sigma} ---")
            print(f"    -> 所需 B_rms = {B_target_rms:.4f} T")
        else:
            # Fallback: 用户指定了 B0，我们将其视为目标 B_rms
            B_target_rms = target_B0
            target_magnetic_energy = B_target_rms ** 2 / (2 * constants.mu0)
            target_sigma = target_magnetic_energy / particle_energy_density
            print(f"--- 能量设计: 固定 B0 输入 ---")
            print(f"    -> 目标 B_rms = {B_target_rms:.4f} T")
            print(f"    -> 结果 Sigma = {target_sigma:.4f}")

        sigma = target_sigma

        return w_pe, d_e, T_plasma_J, theta, particle_energy_density, B_target_rms, target_magnetic_energy, target_sigma, sigma

    def _calculate_derived_parameters(self):
        # --- 基础输入参数 ---
        # 将所有参数从参数对象复制到实例属性
        self.B0 = self.p.B0
        self.n_plasma = self.p.n_plasma
        self.T_plasma = self.p.T_plasma_eV
        self.n_photon = self.p.n_photon_to_plasma_ratio * self.n_plasma

        # --- 束流参数 ---
        self.beam_fraction = self.p.beam_fraction
        # 从动能计算归一化动量 u_drift
        self.beam_energy_eV = self.p.beam_energy_eV
        # 1. 将动能从 eV 转换为焦耳
        beam_ke_joules = self.beam_energy_eV * constants.q_e
        # 2. 计算洛伦兹因子 gamma = 1 + KE / (m*c^2)
        gamma_beam = 1.0 + beam_ke_joules / (constants.m_e * constants.c ** 2)
        # 3. 计算归一化动量 u = sqrt(gamma^2 - 1)
        self.beam_u_drift = np.sqrt(gamma_beam ** 2 - 1.0)

        # 计算新的派生等离子体参量
        w_pe, d_e, T_plasma_J, theta, particle_energy_density, B_target_rms, target_magnetic_energy, target_sigma, sigma = self.get_plasma_quantities(
            self.n_plasma, self.T_plasma, self.p.target_sigma, self.B0
        )
        self.w_pe = w_pe
        self.d_e = d_e
        self.T_plasma_J = T_plasma_J
        self.theta = theta
        self.particle_energy_density = particle_energy_density
        self.B_target_rms = B_target_rms
        self.target_magnetic_energy = target_magnetic_energy
        self.target_sigma = target_sigma
        self.sigma = sigma

        # --- 归一化尺度 (用于诊断输出) ---
        # 无论初始磁场如何，都使用基于等离子体热力学性质的普适尺度进行归一化

        # 1. 磁场归一化尺度: 基于磁能与总热能均分的磁场
        #    B_norm^2 / (2*mu0) = 2 * n_plasma * T_plasma_J
        total_thermal_energy_density = 2 * self.n_plasma * self.T_plasma_J
        self.B_norm = np.sqrt(2 * constants.mu0 * total_thermal_energy_density)

        # 2. 电流归一化尺度: 基于相对论热电流
        self.J_norm = self.n_plasma * constants.q_e * constants.c

        # --- 空间与时间网格 ---
        self.LX, self.LY, self.LZ = self.p.LX, self.p.LY, self.p.LZ
        self.LT, self.DT = self.p.LT, self.p.DT
        self.NX, self.NY, self.NZ = self.p.NX, self.p.NY, self.p.NZ
        self.NPPC = self.p.NPPC

        # 现在使用计算出的物理尺度来定义模拟域的绝对尺寸
        self.Lx = self.LX * self.d_e
        self.Ly = self.LY * self.d_e
        self.Lz = self.LZ * self.d_e
        self.dt = self.DT / self.w_pe  # dt based on w_pe
        self.dx = self.Lx / self.NX
        self.dy = self.Ly / self.NY
        self.dz = self.Lz / self.NZ

        # 诊断频率
        self.total_steps = int(self.LT / self.DT)
        self.diag_steps = self.total_steps // 50  # 每200步输出一次诊断
        self.diag_steps = max(1, self.diag_steps)

    def _print_summary(self):
        # 打印更新后的等离子体参数
        print("--- 归一化尺度 ---")
        print(f"\tB_norm (能量均分磁场) = {self.B_norm:.3e} T")
        print(f"\tJ_norm (热电流)   = {self.J_norm:.3e} A/m^2")
        print("--- 独立输入参数 ---")
        print(
            f"\tB0 = {self.B0:.2f} T\n"
            f"\tn0 = {self.n_plasma:.2e} m^-3 (每种粒子)\n"
            f"\tT = {self.T_plasma * 1e-6:.2f} MeV\n"
            f"\t电子束占比 = {self.beam_fraction * 100:.1f}%\n"
            f"\t电子束动能 = {self.beam_energy_eV * 1e-6:.2f} MeV\n"
            f"\t电子束漂移动量 = {self.beam_u_drift:.2f}\n"
        )
        print("--- 派生等离子体参数 ---")
        print(
            f"\td_e = {self.d_e:.3e} m (电子趋肤深度)\n"
            f"\t1/w_pe = {1.0 / self.w_pe:.3e} s (等离子体周期)\n"
            f"\t磁化强度 sigma = {self.sigma:.3f}\n"
            f"\t相对论Theta参数 = {self.theta:.3f}\n"
        )
        print("--- 数值参数 ---")
        print(
            f"\t计算域 = {self.Lx:.3e}m x {self.Ly:.3e}m x {self.Lz:.3e}m "
            f"({self.LX:.0f} d_e x {self.LY:.0f} d_e x {self.LZ:.0f} d_e)\n"
            f"\t网格 = {self.NX} x {self.NY} x {self.NZ}\n"
            f"\tdx = {self.Lx / self.NX:.3e} m, dy = {self.Ly / self.NY:.3e} m, dz = {self.Lz / self.NZ:.3e} m\n"
            f"\t时间步长 = {self.dt:.3e} s ({self.DT:.2e} / w_pe)\n"
            f"\t总步数 = {self.total_steps:d} (运行时间 = {self.total_steps * self.dt:.3e} s)\n"
        )

    def _archive_parameters(self):
        """准备并保存用于复现的参数字典。"""
        # 构建一个纯数据字典，去除没必要序列化的对象
        params_to_save = self.__dict__.copy()
        params_to_save.pop('magnetic_field', None)  # 移除整个对象
        params_to_save.pop('p', None)  # 移除原始配置对象，因为其信息已被提取
        params_to_save.pop('io', None)  # 移除 IOManager

        # 添加可复现的磁场符号表示
        params_to_save['Bx_srepr'] = self.magnetic_field.Bx_srepr
        params_to_save['By_srepr'] = self.magnetic_field.By_srepr
        params_to_save['Bz_srepr'] = self.magnetic_field.Bz_srepr

        # 委托给 IOManager 保存
        self.io.save_simulation_parameters(params_to_save)

    def _init_magnetic_field(self):
        # 从 config 中读取磁场参数
        b_field_config = Bunch(
            B_field_typ=self.p.B_field_type,
            Lx=self.Lx,
            Ly=self.Ly,
            Lz=self.Lz,
            B_target_rms=self.B_target_rms,
            num_gaussians=self.p.num_gaussians,
            gaussian_width_L_ratio=self.p.gaussian_width_L_ratio,
            B_field_type=self.p.B_field_type,
        )
        # 使用工厂创建磁场对象
        self.magnetic_field: InitialMagneticField = magnetic_field_factory(b_field_config)
        self.magnetic_field.debug_print()

    def _setup_species_wrappers(self):
        """
        集中定义所有的粒子物种、它们的初始 picmi 参数、布局参数以及
        需要稍后注入 Bucket 的底层参数。
        """
        # 1. 计算密度和宏粒子数
        n_thermal = self.n_plasma * (1.0 - self.beam_fraction)
        n_beam = self.n_plasma * self.beam_fraction

        nppc_thermal = int(self.NPPC * (1.0 - self.beam_fraction))
        nppc_beam = int(self.NPPC * self.beam_fraction)

        # 2. 准备物理参数 (bucket args)
        theta_plasma = (self.T_plasma * e) / (m_e * c ** 2)
        u_drift = self.beam_u_drift

        # 定义通用的热分布 Bucket 参数
        bucket_thermal = Bunch(
            momentum_distribution_type="maxwell_juttner",
            theta=theta_plasma
        )

        # 定义 Beam 的 Bucket 参数
        bucket_beam_e = Bunch(
            momentum_distribution_type="constant",
            ux=0.0, uy=0.0, uz=u_drift
        )
        bucket_beam_p = Bunch(
            momentum_distribution_type="constant",
            ux=0.0, uy=0.0, uz=-u_drift
        )

        # --- 使用 lambda 定义布局工厂 ---
        # 这是一个依赖 grid 的布局
        thermal_layout_factory = lambda grid: picmi.PseudoRandomLayout(
            grid=grid, n_macroparticles_per_cell=nppc_thermal
        )
        beam_layout_factory = lambda grid: picmi.PseudoRandomLayout(
            grid=grid, n_macroparticles_per_cell=nppc_beam
        )

        species_wrappers: typing.List[SpeciesWrapper] = []

        # --- 电子 (热) ---
        species_wrappers.append(SpeciesWrapper(
            name="electrons_thermal",
            charge="-q_e", mass="m_e",
            initial_distribution=picmi.UniformDistribution(density=n_thermal),
            method='LLRK4',
            layout_config=thermal_layout_factory,
            bucket_params=bucket_thermal
        ))

        # --- 电子 (束流) ---
        species_wrappers.append(SpeciesWrapper(
            name="electrons_beam",
            charge="-q_e", mass="m_e",
            initial_distribution=picmi.UniformDistribution(density=n_beam),
            method='LLRK4',
            layout_config=thermal_layout_factory,
            bucket_params=bucket_beam_e
        ))

        # --- 正电子 (热) ---
        species_wrappers.append(SpeciesWrapper(
            name="positrons_thermal",
            charge="q_e", mass="m_e",
            initial_distribution=picmi.UniformDistribution(density=n_thermal),
            method='LLRK4',
            layout_config=beam_layout_factory,
            bucket_params=bucket_thermal
        ))

        # --- 正电子 (束流) ---
        species_wrappers.append(SpeciesWrapper(
            name="positrons_beam",
            charge="q_e", mass="m_e",
            initial_distribution=picmi.UniformDistribution(density=n_beam),
            method='LLRK4',
            layout_config=beam_layout_factory,
            bucket_params=bucket_beam_p
        ))

        # --- 光子 (无 Bucket 参数) ---
        # TODO 在这里，我们没有为光子创建它的初始能量，包括电子也只是设置了初始速度……因为picmi没有实现这个功能，我再查查看怎么做。
        species_wrappers.append(SpeciesWrapper(
            name="photons",
            charge=0, mass=0,
            # layout_config=lambda grid: picmi.PseudoRandomLayout(
            #             grid=grid, n_macroparticles_per_cell=self.NPPC
            #         ),
            # TODO 目前，光子和其他粒子没有相互作用，如果有相互作用，它通常会通过电子对产生、康普顿散射之类的效应来提供“热化”或“阻力”，阻碍磁重联。同时，我们也并没有加入初始光子到模拟中。
            # bucket_params 留空
        ))

        return species_wrappers

    @staticmethod
    def create_collision_pairs(species_list: list, ndt: int, coulomb_log: float = None) -> list:
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
            print(f"  - Created collision pair: {collision_name}")

        return collision_objects

    def setup_run(self):
        """
        执行 Picmi 初始化的主要流水线。
        """
        #######################################################################
        # 初始化 Species 对象
        #######################################################################

        # 准备物种列表容器
        # 提前定义好所有的物种参数配置
        species_wrappers = self._setup_species_wrappers()
        self.species_wrappers = species_wrappers

        # 实例化所有 picmi 对象
        for wrapper in species_wrappers:
            wrapper.initialize_picmi_object()

        #######################################################################
        # 添加库仑碰撞
        #######################################################################

        print("\n--- 正在生成二元库仑碰撞对 ---")
        all_charged_species = [
            w.instance for w in species_wrappers if w.name != "photons"
        ]

        all_collisions = self.create_collision_pairs(all_charged_species, ndt=25)

        simulation = picmi.Simulation(
            warpx_serialize_initial_conditions=True,
            verbose=0,
            warpx_collisions=all_collisions,  # 传递包含多个碰撞对象的列表
            warpx_reduced_diags_path=str(self.io.diags_dir),
            warpx_used_inputs_file=str(self.io.output_dir / "warpx_used_inputs")
        )
        self.simulation = simulation
        print(f"--- 总共 {len(all_collisions)} 个碰撞对已添加到模拟中。 ---\n")

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
            Bx_expression=self.magnetic_field.Bx_str,
            By_expression=self.magnetic_field.By_str,
            Bz_expression=self.magnetic_field.Bz_str
        )
        simulation.add_applied_field(B_ext)

        #######################################################################
        # 添加粒子和布局
        #######################################################################

        # 使用 Wrapper 中的布局参数将粒子添加到模拟中
        for wrapper in species_wrappers:
            layout = wrapper.get_layout(self.grid)
            simulation.add_species(wrapper.instance, layout=layout)

        #######################################################################
        # 添加诊断
        #######################################################################

        # 添加所有粒子
        all_particle_species_for_diags = all_charged_species

        # 探测平面诊断
        plane = picmi.ReducedDiagnostic(
            diag_type="FieldProbe",
            name="plane",
            period=self.diag_steps,
            path=str(self.io.diags_dir) + "/",
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
            write_dir=str(self.io.diags_dir),
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
            write_dir=str(self.io.diags_dir),
        )
        simulation.add_diagnostic(field_state_diag)

        #######################################################################
        # 初始化模拟
        #######################################################################

        simulation.initialize_inputs()

        self.between_initialize()

        simulation.initialize_warpx()

    def between_initialize(self):
        """
        在初始化模拟输入之后，初始化WarpX模拟之前，通过bucket操作准备移交给WarpX的数据。
        """
        print(f"\n--- 正在应用低层级 Bucket 属性 ---")
        for wrapper in self.species_wrappers:
            # 只有定义了 bucket_params 的 wrapper 才会执行实际操作
            wrapper.apply_bucket_attributes()
        print(f"--------------------------------------------\n")

    def run_simulation(self):
        """执行模拟的时间步进循环。"""
        self.simulation.step()
