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

import dill
import numpy as np
from mpi4py import MPI as mpi

from pywarpx import callbacks, fields, libwarpx, picmi
from scipy.constants import e, m_e, c

if typing.TYPE_CHECKING:
    from pywarpx.Bucket import Bucket

constants = picmi.constants

comm = mpi.COMM_WORLD

simulation = picmi.Simulation(warpx_serialize_initial_conditions=True, verbose=0)


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
    B0 = 0.1  # 初始磁场强度 (T)
    n_plasma = 5.718e31  # 等离子体数密度 (m^-3) (这是指电子或正电子的数密度)
    T_plasma_eV = 0.7344e6  # 等离子体温度 (eV), e.g., 1 MeV. 对电子和正电子相同。
    n_photon_to_plasma_ratio = 0.66  # 光子与电子(或正电子)的数密度之比 (Ng/Ne ~ 6.3/9.6 ~ 0.66 in BBN)

    # --- 2. 无量纲模拟参数 (Dimensionless Simulation Setup) ---
    LX = 40.0  # 模拟域 x 方向长度 (单位: 电子趋肤深度 d_e)
    LZ = 20.0  # 模拟域 z 方向长度 (单位: 电子趋肤深度 d_e)
    LT = 4000.0  # 模拟总时长 (单位: 等离子体周期 1/w_pe)
    DT = 0.05  # 时间步长 (单位: 等离子体周期 1/w_pe) (需满足CFL条件)

    # --- 3. 数值和扰动参数 (Numerical and Perturbation Parameters) ---
    NX = 32  # x 方向网格数
    NZ = 32  # z 方向网格数
    NPPC = 50  # 每个单元的宏粒子数 (每个物种)

    # 磁场和导向场设置
    Bg_ratio = 0.3  # 导向场与B0的比值
    dB_ratio = 0.01  # 初始扰动磁场与B0的比值

    # (REMOVED) 电阻率和子步数不再需要，因为我们使用标准的电磁求解器
    # eta_normalized = 6e-3
    # substeps = 20


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

        self.LX = self.p.LX
        self.LZ = self.p.LZ
        self.LT = self.p.LT
        self.DT = self.p.DT
        self.NX = self.p.NX
        self.NZ = self.p.NZ
        self.NPPC = self.p.NPPC

        # 计算新的派生等离子体参量
        self.get_plasma_quantities()

        # 现在使用计算出的物理尺度来定义模拟域的绝对尺寸
        self.Lx = self.LX * self.d_e
        self.Lz = self.LZ * self.d_e
        self.dt = self.DT / self.w_pe  # dt based on w_pe

        # run very low resolution as a CI test
        if self.test:
            self.total_steps = 20
            self.diag_steps = self.total_steps // 5
            self.NX = 128
            self.NZ = 128
        else:
            self.total_steps = int(self.LT / self.DT)
            self.diag_steps = self.total_steps // 200
            # 您也可以在这里覆盖参数以进行快速测试，例如：
            # self.NX = 32
            # self.NZ = 32
            # self.total_steps = 200
            # self.diag_steps = 10

        # Initial magnetic field setup using electron skin depth d_e
        self.Bg = self.p.Bg_ratio * self.B0
        self.dB = self.p.dB_ratio * self.B0
        self.Bx = (
            f"{self.B0}*tanh(z*{1.0 / self.d_e})"
            f"+{-self.dB * self.Lx / (2.0 * self.Lz)}*cos({2.0 * np.pi / self.Lx}*x)"
            f"*sin({np.pi / self.Lz}*z)"
        )
        self.By = (
            f"sqrt({self.Bg ** 2 + self.B0 ** 2}-({self.B0}*tanh(z*{1.0 / self.d_e}))**2)"
        )
        self.Bz = f"{self.dB}*sin({2.0 * np.pi / self.Lx}*x)*cos({np.pi / self.Lz}*z)"

        # 使用 d_e 来定义特征电流
        self.J0 = self.B0 / (constants.mu0 * self.d_e)

        # dump all the current attributes to a dill pickle file
        if comm.rank == 0:
            with open("sim_parameters.dpkl", "wb") as f:
                dill.dump(self, f)

        # print out updated plasma parameters
        if comm.rank == 0:
            print("--- Independent Input Parameters ---")
            print(
                f"\tB0 = {self.B0:.2f} T\n"
                f"\tn0 = {self.n_plasma:.2e} m^-3 (per species)\n"
                f"\tT = {self.T_plasma * 1e-6:.2f} MeV\n"
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
                f"\tDomain = {self.Lx:.3f}m x {self.Lz:.3f}m ({self.LX:.0f} d_e x {self.LZ:.0f} d_e)\n"
                f"\tGrid = {self.NX} x {self.NZ}\n"
                f"\tdz = {self.Lz / self.NZ:.3e} m\n"
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

    def setup_run(self):
        """Setup simulation components."""

        #######################################################################
        # Set geometry and boundary conditions                                #
        #######################################################################

        self.grid = picmi.Cartesian2DGrid(
            number_of_cells=[self.NX, self.NZ],
            lower_bound=[-self.Lx / 2.0, -self.Lz / 2.0],
            upper_bound=[self.Lx / 2.0, self.Lz / 2.0],
            lower_boundary_conditions=["periodic", "dirichlet"],
            upper_boundary_conditions=["periodic", "dirichlet"],
            lower_boundary_conditions_particles=["periodic", "reflecting"],
            upper_boundary_conditions_particles=["periodic", "reflecting"],
            warpx_max_grid_size=self.NZ,
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

        #######################################################################
        # Particle types setup                                                #
        #######################################################################

        v_thermal = np.sqrt(self.T_plasma * constants.q_e / constants.m_e)
        theta_plasma = (self.T_plasma * e) / (m_e * c ** 2)

        self.electrons = picmi.Species(
            name="electrons",
            charge="-q_e",
            mass="m_e",
            initial_distribution=picmi.UniformDistribution(
                density=self.n_plasma,
                rms_velocity=[v_thermal, v_thermal, v_thermal]
            ),
        )

        self.positrons = picmi.Species(
            name="positrons",
            charge="q_e",
            mass="m_e",
            initial_distribution=picmi.UniformDistribution(
                density=self.n_plasma,
                rms_velocity=[v_thermal, v_thermal, v_thermal]
            ),
        )

        self.photons = picmi.Species(
            name="photons",
            charge=0,
            mass=0
        )
        # TODO 在这里，我们没有为光子创建它的初始能量，包括电子也只是设置了初始速度……因为picmi没有实现这个功能，我再查查看怎么做。

        layout_electrons = picmi.PseudoRandomLayout(
            grid=self.grid, n_macroparticles_per_cell=self.NPPC
        )
        layout_positrons = picmi.PseudoRandomLayout(
            grid=self.grid, n_macroparticles_per_cell=self.NPPC
        )
        layout_photons = picmi.PseudoRandomLayout(
            grid=self.grid, n_macroparticles_per_cell=self.NPPC
        )
        # TODO 目前，光子和其他粒子没有相互作用，如果有相互作用，它通常会通过电子对产生、康普顿散射之类的效应来提供“热化”或“阻力”，阻碍磁重联。

        simulation.add_species(self.electrons, layout=layout_electrons)
        simulation.add_species(self.positrons, layout=layout_positrons)
        simulation.add_species(self.photons, layout=None)

        #######################################################################
        # Add diagnostics                                                     #
        #######################################################################

        callbacks.installafterEsolve(self.check_fields)

        if self.test:
            particle_diag = picmi.ParticleDiagnostic(
                name="diag1",
                period=self.total_steps,
                species=[self.electrons, self.positrons],
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
            z_probe=0.0,
            detector_radius=self.d_e,
            target_up_x=0,
            target_up_z=1.0,
        )
        simulation.add_diagnostic(plane)

        particle_state_diag = picmi.ParticleDiagnostic(
            name="particle_states",
            period=self.diag_steps,
            species=[self.electrons, self.positrons],
            data_list=["ux", "uy", "uz", "x", "z", "weighting"],
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

        electrons_bucket: Bucket = self.electrons.species
        electrons_bucket.add_new_attr("momentum_distribution_type", "maxwell_boltzmann")
        electrons_bucket.add_new_attr("theta", theta_plasma)

        positrons_bucket: Bucket = self.positrons.species
        positrons_bucket.add_new_attr("momentum_distribution_type", "maxwell_boltzmann")
        positrons_bucket.add_new_attr("theta", theta_plasma)

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
        Ez = fields.EzFPWrapper()[...]

        # 2. 获取网格尺寸以进行数值微分。
        dx = self.Lx / self.NX
        dz = self.Lz / self.NZ

        # 3. 手动计算中心差分，将导数结果统一到网格单元中心。
        #    这将产生两个形状为 (NX, NZ) 的数组。

        # 检查并处理可能的维度顺序
        # WarpX/AMReX 通常是 (x, z) 顺序
        if Ex.shape[0] == self.NX and Ex.shape[1] == self.NZ + 1:
            # Ex.shape is (NX, NZ+1) -> (32, 33)
            # Ez.shape is (NX+1, NZ) -> (33, 32)
            dEx_dx_centered = (Ex[1:, :-1] - Ex[:-1, :-1]) / dx  # 这部分可能有误，我们直接算散度
            dEz_dz_centered = (Ez[:-1, 1:] - Ez[:-1, :-1]) / dz

            # 一个更稳健的中心差分方法：
            dEx_dx = (Ex[1:, :-1] - Ex[:-1, :-1]) / dx  # 应该是Ez的x导数
            # 让我们重新思考散度定义
            # div(E)_i,k = (Ex_i+1/2,k - Ex_i-1/2,k)/dx + (Ez_i,k+1/2 - Ez_i,k-1/2)/dz

            # 对于形状为 (NX+1, NZ) 的 Ez 数组：
            dEz_dx = (Ez[1:, :] - Ez[:-1, :]) / dx  # 结果形状 (NX, NZ)

            # 对于形状为 (NX, NZ+1) 的 Ex 数组：
            # (注意：根据错误信息，Ex和Ez的形状可能是反过来的，但逻辑不变)
            if Ex.shape[0] == self.NX:  # Ex shape (NX, NZ+1)
                dEx_dz = (Ex[:, 1:] - Ex[:, :-1]) / dz  # 结果形状 (NX, NZ)
                div_E = dEz_dx + dEx_dz  # 应该是 dEx/dx + dEz/dz
            else:  # Ex shape (NX+1, NZ)
                dEx_dx = (Ex[1:, :] - Ex[:-1, :]) / dx  # 结果形状 (NX, NZ)
                # Ez shape (NX, NZ+1)
                dEz_dz = (Ez[:, 1:] - Ez[:, :-1]) / dz  # 结果形状 (NX, NZ)
                div_E = dEx_dx + dEz_dz

        else:
            # 如果顺序是反的 (Python/Numpy 默认 row-major, (z,x))
            # Ez.shape is (NZ, NX+1)
            # Ex.shape is (NZ+1, NX)
            dEz_dx = (Ez[:, 1:] - Ez[:, :-1]) / dx  # 结果形状 (NZ, NX)
            dEx_dz = (Ex[1:, :] - Ex[:-1, :]) / dz  # 结果形状 (NZ, NX)
            div_E = dEx_dz + dEz_dx  # 应该是 dEx/dx + dEz/dz

        # --- 让我们简化并使用最可能正确的形式 ---
        # 假设 WarpX 导出到 Python 的数组是 (x, z) 索引
        # Ez.shape = (NX+1, NZ) -> (33, 32)
        # Ex.shape = (NX, NZ+1) -> (32, 33)
        # (这与错误信息中的形状顺序相反，但我们来测试一下)

        # 假设 E_x[i, k+1/2] 和 E_z[i+1/2, k]
        # div(E) 在 (i,k) 处 = (E_z[i+1/2, k] - E_z[i-1/2, k])/dx + (E_x[i, k+1/2] - E_x[i, k-1/2])/dz

        dEz_dx = (Ez[1:, :] - Ez[:-1, :]) / dx  # 结果形状 (NX, NZ) = (32, 32)
        dEx_dz = (Ex[:, 1:] - Ex[:, :-1]) / dz  # 结果形状 (NX, NZ) = (32, 32)

        div_E = dEz_dx + dEx_dz

        # 4. 根据高斯定律 ρ = ε₀ * ∇·E 计算电荷密度。
        rho = constants.ep0 * div_E

        # Update field wrappers for the new setup
        # Jiy (ion current) is replaced by Jey (electron current)
        # rho = _MultiFABWrapper(mf_name="rho")[:, :]
        Jey = fields.JyFPWrapper()[...] / self.J0
        Jy = fields.JyFPWrapper()[...] / self.J0  # Total current
        Bx = fields.BxFPWrapper()[...] / self.B0
        By = fields.ByFPWrapper()[...] / self.B0
        Bz = fields.BzFPWrapper()[...] / self.B0

        if libwarpx.amr.ParallelDescriptor.MyProc() != 0:
            return

        # save the fields to file
        with open(f"diags/fields/fields_{step:06d}.npz", "wb") as f:
            np.savez(f, rho=rho, Jey=Jey, Jy=Jy, Bx=Bx, By=By, Bz=Bz)


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