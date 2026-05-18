#!/usr/bin/env python3
"""BBN 混合 PIC 模拟引擎。

使用 WarpX HybridPICSolver：
- 电子作为无惯量流体（广义 Ohm 定律计算 E 场）
- 离子（质子）作为 PIC 粒子推进
"""

import typing

import numpy as np
from mpi4py import MPI as mpi
from pywarpx import picmi
from scipy.constants import e as q_e, c, m_e, m_p

from simulation.bbn.config import (
    SimulationParameters,
    load_cosmocons_row,
    _cgs_to_si,
    compute_plasma_params,
)
from simulation.ep_pair.engine import SpeciesWrapper
from simulation.ep_pair.io_manager import IOManager
from simulation.ep_pair.utils import Bunch, mpi_barrier, enable_mpi_print

if typing.TYPE_CHECKING:
    pass

constants = picmi.constants
comm = mpi.COMM_WORLD

enable_mpi_print()


class BBNHybridSimulation:
    def __init__(self, params: SimulationParameters, output_dir: str, verbose: bool):
        self.p = params
        self.output_dir = output_dir
        self.verbose = verbose
        self.io = IOManager(self.output_dir)

        print(f"\n=== 初始化 BBN 混合 PIC 模拟 ===")

        self._load_cosmocons()
        self._calculate_derived_parameters()

        mpi_barrier("参数计算完毕")

        self.io.prepare_directories(overwrite=True)
        self._archive_parameters()

        mpi_barrier("文件系统就绪")

        self._print_summary()
        self.setup_run()

    def _load_cosmocons(self):
        """从 CosmoCons.dat 加载物理数据。"""
        print("--- 加载 CosmoCons 数据 ---")
        row = load_cosmocons_row(self.p.cosmocons_row)
        si = _cgs_to_si(row)

        self.cosmocons_raw = row
        self.cosmocons_si = si

        # 核心物理量
        self.n_e = si["n_e"]                # 电子数密度 m^-3
        self.T_e_eV = si["T_eV"]            # 电子温度 eV
        self.T_e_MeV = si["T_MeV"]
        self.rho_b = si["rho_b"]            # 重子质量密度 kg/m^3
        self.H_hubble = si["H"]             # Hubble 参数 s^-1

        # 离子数密度：从重子质量密度计算
        self.n_ion = self.rho_b * self.p.ion_mass_fraction / self.p.ion_mass

        print(f"  行 {self.p.cosmocons_row}: T = {self.T_e_MeV:.5f} MeV = {self.T_e_eV:.0f} eV")
        print(f"  n_e = {self.n_e:.4e} m^-3")
        print(f"  n_ion = {self.n_ion:.4e} m^-3  (n_ion/n_e = {self.n_ion/self.n_e:.4e})")
        print(f"  rho_b = {self.rho_b:.4e} kg/m^3")

    def _calculate_derived_parameters(self):
        """从数据计算所有派生等离子体参数。"""
        print("--- 计算派生参数 ---")

        plasma = compute_plasma_params(
            self.n_e, self.n_ion, self.T_e_eV, self.p.ion_mass
        )
        self.w_pe = plasma["w_pe"]
        self.w_pi = plasma["w_pi"]
        self.d_e = plasma["d_e"]
        self.d_i = plasma["d_i"]
        self.theta_i = plasma["theta_i"]
        self.u_th_ion = plasma["u_th_ion"]
        self.v_th_ion = plasma["v_th_ion"]

        # 空间/时间网格（以 d_i 和 ω_pi 为基准）
        self.Lx = self.p.LX * self.d_i
        self.Ly = self.p.LY * self.d_i
        self.Lz = self.p.LZ * self.d_i
        self.dt = self.p.DT / self.w_pi

        self.NX, self.NY, self.NZ = self.p.NX, self.p.NY, self.p.NZ
        self.NPPC = self.p.NPPC
        self.dx = self.Lx / self.NX

        self.total_steps = int(self.p.LT / self.p.DT)
        self.field_diag_steps = max(1, self.total_steps // self.p.field_total_step)
        self.particle_diag_steps = max(1, self.total_steps // self.p.particle_total_step)

        print(f"  d_i = {self.d_i:.4e} m")
        print(f"  ω_pi = {self.w_pi:.4e} rad/s")
        print(f"  u_th(ion) = {self.u_th_ion:.6f} c")
        print(f"  计算域 = {self.Lx:.4e} m × {self.Lz:.4e} m")
        print(f"  dt = {self.dt:.4e} s, 总步数 = {self.total_steps}")

    def _print_summary(self):
        print("\n=== BBN 混合 PIC 参数汇总 ===")
        print(f"  CosmoCons 行 {self.p.cosmocons_row}: T = {self.T_e_eV/1e3:.2f} keV")
        print(f"  n_e = {self.n_e:.3e} m^-3")
        print(f"  n_ion = {self.n_ion:.3e} m^-3")
        print(f"  d_i = {self.d_i:.3e} m")
        print(f"  ω_pi = {self.w_pi:.3e} s^-1")
        print(f"  v_th(ion) = {self.u_th_ion * c:.3e} m/s ({self.u_th_ion:.4f} c)")
        print(f"  H = {self.H_hubble:.3e} s^-1 (ω_pi/H = {self.w_pi/self.H_hubble:.2e})")
        print(f"  网格: {self.NX}×{self.NZ}, dx = {self.dx:.3e} m ({self.dx/self.d_i:.3f} d_i)")
        print(f"  步数: {self.total_steps}, dt = {self.dt:.3e} s")

    def _archive_parameters(self):
        params_to_save = {k: v for k, v in self.__dict__.items()
                          if not k.startswith('_') and k != 'io' and k != 'p'}
        params_to_save['cosmocons_row'] = self.p.cosmocons_row
        for attr in dir(self.p):
            if not attr.startswith('_'):
                params_to_save[f'config.{attr}'] = getattr(self.p, attr)
        self.io.save_simulation_parameters(params_to_save)

    def _setup_species(self) -> list[SpeciesWrapper]:
        """设置离子物种。"""
        Z = self.p.ion_charge_number
        mass = self.p.ion_mass
        u_th = self.u_th_ion

        bucket_ion = Bunch(
            species_type="proton",
            mass=mass,
            charge=f"{Z}*q_e" if Z != 1 else "q_e",
            momentum_distribution_type="gaussian",
            ux_m=0.0, uy_m=0.0, uz_m=0.0,
            ux_th=u_th, uy_th=u_th, uz_th=u_th,
        )

        layout_factory = lambda grid: picmi.PseudoRandomLayout(
            grid=grid, n_macroparticles_per_cell=self.NPPC
        )

        return [
            SpeciesWrapper(
                name="protons",
                initial_distribution=picmi.UniformDistribution(
                    density=self.n_ion,
                ),
                method='Boris',
                layout_config=layout_factory,
                bucket_params=bucket_ion,
            )
        ]

    def setup_run(self):
        """初始化 WarpX 模拟。"""
        species_wrappers = self._setup_species()
        self.species_wrappers = species_wrappers

        for w in species_wrappers:
            w.initialize_picmi_object()

        # ---- 网格 ----
        grid_kwargs = dict(
            number_of_cells=[self.NX, self.NZ],
            lower_bound=[-self.Lx / 2, -self.Lz / 2],
            upper_bound=[self.Lx / 2, self.Lz / 2],
            lower_boundary_conditions=["periodic", "periodic"],
            upper_boundary_conditions=["periodic", "periodic"],
            lower_boundary_conditions_particles=["periodic", "periodic"],
            upper_boundary_conditions_particles=["periodic", "periodic"],
            warpx_max_grid_size=min(self.NX, self.NZ),
        )

        if self.p.dim == 3:
            self.grid = picmi.Cartesian3DGrid(**grid_kwargs)
        else:
            self.grid = picmi.Cartesian2DGrid(**grid_kwargs)

        # ---- HybridPICSolver ----
        print("--- 初始化 HybridPICSolver ---")
        print(f"  Te = {self.T_e_eV} eV")
        print(f"  n0 = {self.n_e} m^-3")
        print(f"  gamma = {self.p.electron_gamma_eos}")
        print(f"  resistivity = {self.p.plasma_resistivity}")

        solver = picmi.HybridPICSolver(
            grid=self.grid,
            gamma=self.p.electron_gamma_eos,
            Te=self.T_e_eV,
            n0=self.n_e,
            n_floor=0.1 * self.n_ion,
            plasma_resistivity=self.p.plasma_resistivity,
            substeps=self.p.hybrid_substeps,
        )

        # ---- Simulation ----
        sim_kwargs = Bunch(
            warpx_serialize_initial_conditions=True,
            verbose=0,
            warpx_random_seed=42,
            warpx_particle_pusher_algo="boris",
            warpx_reduced_diags_path=str(self.io.diags_dir),
            warpx_used_inputs_file=str(self.io.output_dir / "warpx_used_inputs"),
        )

        simulation = picmi.Simulation(**sim_kwargs)
        simulation.solver = solver
        simulation.time_step_size = self.dt
        simulation.max_steps = self.total_steps
        simulation.current_deposition_algo = "direct"
        simulation.particle_shape = 1
        simulation.use_filter = False
        simulation.verbose = self.verbose

        self.simulation = simulation

        # ---- 添加离子物种 ----
        for w in species_wrappers:
            layout = w.get_layout(self.grid)
            simulation.add_species(w.instance, layout=layout)

        # ---- 诊断 ----
        all_species = [w.instance for w in species_wrappers]
        particle_data = ["ux", "uy", "uz", "x", "z", "weighting"]

        particle_diag = picmi.ParticleDiagnostic(
            name="particle_states",
            period=self.field_diag_steps,
            species=all_species,
            data_list=particle_data,
            warpx_format='openpmd',
            warpx_openpmd_backend='h5',
            write_dir=str(self.io.diags_dir),
        )
        simulation.add_diagnostic(particle_diag)

        field_diag = picmi.FieldDiagnostic(
            name="field_states",
            grid=self.grid,
            period=self.particle_diag_steps,
            data_list=["Bx", "By", "Bz", "Ex", "Ey", "Ez", "Jx", "Jy", "Jz"],
            warpx_format='openpmd',
            warpx_openpmd_backend='h5',
            write_dir=str(self.io.diags_dir),
        )
        simulation.add_diagnostic(field_diag)

        # ---- 初始化 ----
        print("--- 初始化 WarpX ---")
        simulation.initialize_inputs()

        for w in species_wrappers:
            w.apply_bucket_attributes()

        simulation.initialize_warpx()

    def run_simulation(self):
        """执行模拟。"""
        print(f"\n=== 开始 BBN 混合 PIC 模拟 ({self.total_steps} 步) ===")
        self.simulation.step()
        print("=== 模拟完成 ===")
