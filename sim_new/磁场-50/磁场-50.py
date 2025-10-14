#!/usr/bin/env python3
#
# --- Test script for a fully kinetic pair-plasma simulation in WarpX.
# --- This script simulates magnetic reconnection in a relativistic,
# --- electron-positron force-free current sheet. The setup is analogous
# --- to the original ion-electron problem but adapted for a pair plasma,
# --- a common scenario in high-energy astrophysics.
# ---
# --- Key changes from the original hybrid script:
# --- 1. Replaced kinetic ions and fluid electrons with kinetic electrons,
# ---    kinetic positrons, and photons.
# --- 2. Switched from a Hybrid PIC solver to a full Electromagnetic PIC solver.
# --- 3. Adjusted physical parameters for a relativistic (MeV-scale) plasma.
# --- 4. Renormalized the problem using electron skin depth (d_e) and plasma
# ---    frequency (w_pe) as the natural scales.

import argparse
import shutil
import sys
import typing
from pathlib import Path
import itertools

import dill
import numpy as np
from mpi4py import MPI as mpi

from pywarpx import callbacks, fields, libwarpx, picmi
from scipy.constants import e, m_e, c

if typing.TYPE_CHECKING:
    from pywarpx.Bucket import Bucket

constants = picmi.constants

comm = mpi.COMM_WORLD


# simulation = picmi.Simulation(warpx_serialize_initial_conditions=True, verbose=0)


# =============================================================================
# 这个模拟关注于80KeV，这通常是锂疑难的发生温度左右。
#
# =============================================================================

# =============================================================================
# START OF USER-CONFIGURABLE PARAMETERS
# =============================================================================
class SimulationParameters:
    """
    这是一个专门用于存放所有用户可配置参数的类。
    通过修改这里的数值，您可以独立地控制模拟的各个方面。
    """
    # --- 1. 基础物理参数 (Independent Physical Parameters) ---
    # 这些参数现在描述一个相对论性的电子-正电子对等离子体

    # 设置一个与等效热能磁场 B_norm (根据先前模拟约 2.2e4 T) 可比拟的非零初始磁场
    B0 = 1.0e4  # 初始磁场强度 (T)

    n_plasma = 7.3e27  # 等离子体数密度 (m^-3) (这是指电子或正电子的数密度)
    T_plasma_eV = 8.4e4  # 等离子体温度 (eV), e.g., 1 MeV. 对电子和正电子相同。
    # 在80KeV这个温度，正负电子已经在大量湮灭了，
    n_photon_to_plasma_ratio = 3.93  # 光子与电子(或正电子)的数密度之比 (Ng/Ne ~ 6.3/9.6 ~ 0.66 in BBN)

    # --- 2. 无量纲模拟参数 (Dimensionless Simulation Setup) ---
    LX = 20.0  # 模拟域 x 方向长度 (单位: 电子趋肤深度 d_e)
    LY = 20.0  # 模拟域 y 方向长度 (单位: 电子趋肤深度 d_e)
    LZ = 20.0  # 模拟域 z 方向长度 (单位: 电子趋肤深度 d_e)
    LT = 400.0  # 模拟总时长 (单位: 等离子体周期 1/w_pe)
    DT = 0.05  # 时间步长 (单位: 等离子体周期 1/w_pe) (需满足CFL条件)

    # --- 3. 数值和扰动参数 (Numerical and Perturbation Parameters) ---
    NX = 32  # x 方向网格数
    NY = 32  # y 方向网格数
    NZ = 32  # z 方向网格数
    NPPC = 10  # 每个单元的宏粒子数 (每个物种)

    # 磁场和导向场设置
    Bg_ratio = 0  # 导向场与B0的比值
    dB_ratio = 0  # 初始扰动磁场与B0的比值

    # (REMOVED) 电阻率和子步数不再需要，因为我们使用标准的电磁求解器
    # eta_normalized = 6e-3
    # substeps = 20

    # --- 4. 非热扰动参数 (Non-thermal Perturbation Parameters) --- #
    beam_fraction = 0.5  # 非热束流粒子占总数的比例 (e.g., 20%)
    beam_energy_eV = 8.4e4  # 束流粒子的动能 (eV)。例如 1.0e6 表示 1 MeV。

# =============================================================================
# END OF USER-CONFIGURABLE PARAMETERS
# =============================================================================


class PairPlasmaReconnection(object):  # Class name updated for clarity
    def __init__(self, params: SimulationParameters, test: bool, verbose: bool):
        self.p = params
        self.test = test
        self.verbose = verbose or self.test

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
        gamma_beam = 1.0 + beam_ke_joules / (constants.m_e * constants.c**2)
        # 3. 计算归一化动量 u = sqrt(gamma^2 - 1)
        self.beam_u_drift = np.sqrt(gamma_beam**2 - 1.0)

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

        # run very low resolution as a CI test
        if self.test:
            self.total_steps = 20
            self.diag_steps = self.total_steps // 5
        else:
            self.total_steps = int(self.LT / self.DT)
            self.diag_steps = self.total_steps // 25
            # 您也可以在这里覆盖参数以进行快速测试，例如：
            # self.NX = 32
            # self.NZ = 32
            # self.total_steps = 200
            # self.diag_steps = 10

        self.Bx = "0.0"
        self.By = "0.0"
        self.Bz = f"{self.B0}"

        self.dx = self.Lx / self.NX
        self.dy = self.Ly / self.NY
        self.dz = self.Lz / self.NZ

        # dump all the current attributes to a dill pickle file
        if comm.rank == 0:
            with open("sim_parameters.dpkl", "wb") as f:
                dill.dump(self, f)

        # print out updated plasma parameters
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

        self.setup_run()

    def get_plasma_quantities(self):
        """
        Calculate derived plasma parameters for an electron-positron plasma.
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

    def setup_run(self):
        """Setup simulation components."""
        #######################################################################
        # Particle types setup                                                #
        #######################################################################

        #######################################################################
        # Particle types setup                                                #
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
        )
        self.electrons_beam = picmi.Species(
            name="electrons_beam",
            charge="-q_e",
            mass="m_e",
            initial_distribution=picmi.UniformDistribution(density=n_beam),
        )

        self.positrons_thermal = picmi.Species(
            name="positrons_thermal",
            charge="q_e",
            mass="m_e",
            initial_distribution=picmi.UniformDistribution(density=n_thermal),
        )
        self.positrons_beam = picmi.Species(
            name="positrons_beam",
            charge="q_e",
            mass="m_e",
            initial_distribution=picmi.UniformDistribution(density=n_beam),
        )

        self.photons = picmi.Species(
            name="photons",
            charge=0,
            mass=0
        )
        # TODO 在这里，我们没有为光子创建它的初始能量，包括电子也只是设置了初始速度……因为picmi没有实现这个功能，我再查查看怎么做。

        #######################################################################
        # Add Collisions for Thermalization                                   #
        #######################################################################

        print("\n--- Generating Binary Coulomb Collision Pairs ---")
        all_charged_species = [
            self.electrons_thermal, self.electrons_beam,
            self.positrons_thermal, self.positrons_beam
        ]

        all_collisions = self.create_collision_pairs(all_charged_species, ndt=25)

        global simulation
        simulation = picmi.Simulation(
            warpx_serialize_initial_conditions=True,
            verbose=0,
            warpx_collisions=all_collisions  # 传递包含多个碰撞对象的列表
        )
        print(f"--- Total of {len(all_collisions)} collision pairs added to simulation. ---\n")

        #######################################################################
        # Set geometry and boundary conditions                                #
        ################################################################diag_steps #######

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
        # Field solver and external field                                     #
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
        # Add diagnostics                                                     #
        #######################################################################

        callbacks.installafterEsolve(self.check_fields)

        # Update diagnostics to include all new species
        all_particle_species_for_diags = [
            self.electrons_thermal, self.electrons_beam,
            self.positrons_thermal, self.positrons_beam
        ]

        if self.test:
            particle_diag = picmi.ParticleDiagnostic(
                name="diag1",
                period=self.total_steps,
                species=all_particle_species_for_diags,
                data_list=["ux", "uy", "uz", "x", "z", "weighting"],
            )
            simulation.add_diagnostic(particle_diag)
            field_diag = picmi.FieldDiagnostic(
                name="diag1",
                grid=self.grid,
                period=self.total_steps,
                data_list=["Bx", "By", "Bz", "Ex", "Ey", "Ez", "Jx", "Jy", "Jz"],
            )
            simulation.add_diagnostic(field_diag)

        plane = picmi.ReducedDiagnostic(
            diag_type="FieldProbe",
            name="plane",
            period=self.diag_steps,
            path="diags/",
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

        particle_state_diag = picmi.ParticleDiagnostic(
            name="particle_states",
            period=self.diag_steps,
            species=all_particle_species_for_diags,
            data_list=["ux", "uy", "uz", "x", "y",  "z", "weighting"],
            warpx_format='openpmd',
            warpx_openpmd_backend='h5',
        )
        simulation.add_diagnostic(particle_state_diag)

        field_state_diag = picmi.FieldDiagnostic(
            name="field_states",
            grid=self.grid,
            period=self.diag_steps,
            data_list=["Bx", "By", "Bz", "Ex", "Ey", "Ez", "Jx", "Jy", "Jz"],
            warpx_format='openpmd',
            warpx_openpmd_backend='h5',
        )
        simulation.add_diagnostic(field_state_diag)

        #######################################################################
        # Initialize                                                          #
        #######################################################################

        if comm.rank == 0:
            if Path.exists(Path("diags")):
                shutil.rmtree("diags")
            Path("diags/fields").mkdir(parents=True, exist_ok=True)

        simulation.initialize_inputs()

        # <--- MODIFIED SECTION START ---
        # Now, set the momentum distribution for each of the four species
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

        simulation.initialize_warpx()

    def check_fields(self):
        step = simulation.extension.warpx.getistep(lev=0) - 1

        if not (step == 1 or step % self.diag_steps == 0):
            return

        # --- 最终解决方案 (考虑Yee交错网格) ---
        # 1. 获取必定存在的电场分量。它们的形状因交错网格而不同。
        #    假设 Ex.shape is (NX, NZ+1) and Ez.shape is (NX+1, NZ)
        #    (或者反过来，这取决于WarpX的索引顺序，但差分方法是相同的)
        Ex = fields.ExFPWrapper()[...]
        Ey = fields.EyFPWrapper()[...]
        Ez = fields.EzFPWrapper()[...]

        # 2. 获取网格尺寸以进行数值微分。
        dx = self.Lx / self.NX
        dy = self.Ly / self.NY
        dz = self.Lz / self.NZ

        # 3D散度计算 - 这里需要根据实际的场分量形状进行调整
        # 注意：WarpX在3D中的场分量形状可能不同，需要根据实际情况调整

        # 简化处理：只保存场数据，不计算散度
        # 在3D中计算散度更复杂，这里暂时跳过

        # Update field wrappers for the new setup
        # Jiy (ion current) is replaced by Jey (electron current)
        # rho = _MultiFABWrapper(mf_name="rho")[:, :]
        # 使用在 __init__ 中定义的、对 B0=0 情况稳健的归一化尺度

        # # 3. 手动计算中心差分，将导数结果统一到网格单元中心。
        # #    这将产生两个形状为 (NX, NZ) 的数组。
        #
        # # 检查并处理可能的维度顺序
        # # WarpX/AMReX 通常是 (x, z) 顺序
        # if Ex.shape[0] == self.NX and Ex.shape[1] == self.NZ + 1:
        #     # Ex.shape is (NX, NZ+1) -> (32, 33)
        #     # Ez.shape is (NX+1, NZ) -> (33, 32)
        #     dEx_dx_centered = (Ex[1:, :-1] - Ex[:-1, :-1]) / dx  # 这部分可能有误，我们直接算散度
        #     dEz_dz_centered = (Ez[:-1, 1:] - Ez[:-1, :-1]) / dz
        #
        #     # 一个更稳健的中心差分方法：
        #     dEx_dx = (Ex[1:, :-1] - Ex[:-1, :-1]) / dx  # 应该是Ez的x导数
        #     # 让我们重新思考散度定义
        #     # div(E)_i,k = (Ex_i+1/2,k - Ex_i-1/2,k)/dx + (Ez_i,k+1/2 - Ez_i,k-1/2)/dz
        #
        #     # 对于形状为 (NX+1, NZ) 的 Ez 数组：
        #     dEz_dx = (Ez[1:, :] - Ez[:-1, :]) / dx  # 结果形状 (NX, NZ)
        #
        #     # 对于形状为 (NX, NZ+1) 的 Ex 数组：
        #     # (注意：根据错误信息，Ex和Ez的形状可能是反过来的，但逻辑不变)
        #     if Ex.shape[0] == self.NX:  # Ex shape (NX, NZ+1)
        #         dEx_dz = (Ex[:, 1:] - Ex[:, :-1]) / dz  # 结果形状 (NX, NZ)
        #         div_E = dEz_dx + dEx_dz  # 应该是 dEx/dx + dEz/dz
        #     else:  # Ex shape (NX+1, NZ)
        #         dEx_dx = (Ex[1:, :] - Ex[:-1, :]) / dx  # 结果形状 (NX, NZ)
        #         # Ez shape (NX, NZ+1)
        #         dEz_dz = (Ez[:, 1:] - Ez[:, :-1]) / dz  # 结果形状 (NX, NZ)
        #         div_E = dEx_dx + dEz_dz
        #
        # else:
        #     # 如果顺序是反的 (Python/Numpy 默认 row-major, (z,x))
        #     # Ez.shape is (NZ, NX+1)
        #     # Ex.shape is (NZ+1, NX)
        #     dEz_dx = (Ez[:, 1:] - Ez[:, :-1]) / dx  # 结果形状 (NZ, NX)
        #     dEx_dz = (Ex[1:, :] - Ex[:-1, :]) / dz  # 结果形状 (NZ, NX)
        #     div_E = dEx_dz + dEz_dx  # 应该是 dEx/dx + dEz/dz
        #
        # # --- 让我们简化并使用最可能正确的形式 ---
        # # 假设 WarpX 导出到 Python 的数组是 (x, z) 索引
        # # Ez.shape = (NX+1, NZ) -> (33, 32)
        # # Ex.shape = (NX, NZ+1) -> (32, 33)
        # # (这与错误信息中的形状顺序相反，但我们来测试一下)
        #
        # # 假设 E_x[i, k+1/2] 和 E_z[i+1/2, k]
        # # div(E) 在 (i,k) 处 = (E_z[i+1/2, k] - E_z[i-1/2, k])/dx + (E_x[i, k+1/2] - E_x[i, k-1/2])/dz
        #
        # dEz_dx = (Ez[1:, :] - Ez[:-1, :]) / dx  # 结果形状 (NX, NZ) = (32, 32)
        # dEx_dz = (Ex[:, 1:] - Ex[:, :-1]) / dz  # 结果形状 (NX, NZ) = (32, 32)
        #
        # div_E = dEz_dx + dEx_dz
        #
        # # 4. 根据高斯定律 ρ = ε₀ * ∇·E 计算电荷密度。
        # rho = constants.ep0 * div_E
        #
        # # Update field wrappers for the new setup
        # # Jiy (ion current) is replaced by Jey (electron current)
        # # rho = _MultiFABWrapper(mf_name="rho")[:, :]
        # # 使用在 __init__ 中定义的、对 B0=0 情况稳健的归一化尺度


        Jey = fields.JyFPWrapper()[...] / self.J_norm
        Jy = fields.JyFPWrapper()[...] / self.J_norm
        Bx = fields.BxFPWrapper()[...] / self.B_norm
        By = fields.ByFPWrapper()[...] / self.B_norm
        Bz = fields.BzFPWrapper()[...] / self.B_norm

        if libwarpx.amr.ParallelDescriptor.MyProc() != 0:
            return

        # save the fields to file
        with open(f"diags/fields/fields_{step:06d}.npz", "wb") as f:
            np.savez(f, Ex=Ex, Ey=Ey, Ez=Ez, Jey=Jey, Jy=Jy, Bx=Bx, By=By, Bz=Bz,
                     J_norm=self.J_norm, B_norm=self.B_norm)


##########################
# parse input parameters
##########################

parser = argparse.ArgumentParser()
parser.add_argument(
    "-t",
    "--test",
    help="toggle whether this script is run as a short CI test",
    action="store_true",
)
parser.add_argument(
    "-v",
    "--verbose",
    help="Verbose output",
    action="store_true",
)
args, left = parser.parse_known_args()
sys.argv = sys.argv[:1] + left

run = PairPlasmaReconnection(params=SimulationParameters(), test=args.test, verbose=args.verbose)
simulation.step()
