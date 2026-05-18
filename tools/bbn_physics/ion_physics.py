"""Compute BBN-era ion plasma parameters for PIC simulation design.

Derives ion-relevant quantities from CosmoCons.dat data:
- Ion plasma frequencies, Debye lengths, Larmor radii
- Particle weight ratios for PIC
- Time step and grid constraints
- Nuclear reaction rate estimates
"""

import numpy as np
from pathlib import Path
from parse_cosmocons import (
    load_cosmocons, compute_bbn_plasma_params, t9_from_MeV,
    c_light, k_B, m_p, m_e, e_charge, h_planck, MeV_to_K, MeV_to_erg,
    debye_length_electrons, plasma_frequency, thermal_velocity,
    coupling_parameter, coulomb_logarithm,
)


# Nucleon data: (name, Z, A, mass in g)
NUCLEONS = {
    "p":    {"Z": 1, "A": 1, "mass": m_p},
    "n":    {"Z": 0, "A": 1, "mass": m_p * 1.001378},
    "D":    {"Z": 1, "A": 2, "mass": m_p * 2.013553},
    "He3":  {"Z": 2, "A": 3, "mass": m_p * 3.014932},
    "He4":  {"Z": 2, "A": 4, "mass": m_p * 4.001506},
    "Li6":  {"Z": 3, "A": 6, "mass": m_p * 6.013477},
    "Li7":  {"Z": 3, "A": 7, "mass": m_p * 7.014357},
    "Be7":  {"Z": 4, "A": 7, "mass": m_p * 7.014732},
    "Be9":  {"Z": 4, "A": 9, "mass": m_p * 9.009991},
}

# BBN mass fractions at freeze-out (approximate, for T ~ 0.03-0.08 MeV)
# X_p ~ 0.75, Y_He4 ~ 0.25, D/H ~ 2.5e-5
DEFAULT_ABUNDANCES = {
    "p":   0.75,
    "D":   2.5e-5,
    "He3": 1.0e-5,
    "He4": 0.25,
    "Li7": 1.0e-10,
}


def ion_number_density(rhob_g_cm3: float, species: str,
                       abundance: float | None = None) -> float:
    """Ion number density [cm^-3] for a given species.

    Parameters
    ----------
    rhob_g_cm3 : baryon mass density [g/cm^3]
    species : nucleon key (e.g. 'p', 'He4')
    abundance : mass fraction (default: from DEFAULT_ABUNDANCES)
    """
    if abundance is None:
        abundance = DEFAULT_ABUNDANCES.get(species, 1e-10)
    nuc = NUCLEONS[species]
    n_baryon = rhob_g_cm3 / m_p  # total baryon number density
    n_species = abundance * n_baryon / nuc["A"]
    return n_species


def ion_plasma_frequency(n_i: float, Z: int, mass: float) -> float:
    """Ion plasma frequency ω_pi [rad/s]."""
    return np.sqrt(4 * np.pi * n_i * (Z * e_charge)**2 / mass)


def ion_debye_length(T_MeV: float, n_e: float, n_i: float,
                     Z: int, T_i_MeV: float | None = None) -> float:
    """Total Debye screening length including ion contribution.

    1/λ_D² = 1/λ_De² + 1/λ_Di²
    """
    if T_i_MeV is None:
        T_i_MeV = T_MeV
    T_e_K = T_MeV * MeV_to_K
    T_i_K = T_i_MeV * MeV_to_K
    kBT_e = k_B * T_e_K
    kBT_i = k_B * T_i_K

    inv_lambda2 = 0.0
    if n_e > 0 and kBT_e > 0:
        inv_lambda2 += 4 * np.pi * n_e * e_charge**2 / kBT_e
    if n_i > 0 and kBT_i > 0:
        inv_lambda2 += 4 * np.pi * n_i * (Z * e_charge)**2 / kBT_i

    if inv_lambda2 > 0:
        return 1.0 / np.sqrt(inv_lambda2)
    return np.inf


def ion_gyroradius(T_MeV: float, B_gauss: float, mass: float,
                   Z: int) -> float:
    """Ion Larmor radius r_Li [cm] in magnetic field B.

    r_Li = m_i v_th / (|Z| e B)
    """
    if B_gauss <= 0:
        return np.inf
    v_th = thermal_velocity(T_MeV, mass)
    return mass * v_th / (abs(Z) * e_charge * B_gauss)


def ion_cyclotron_frequency(B_gauss: float, mass: float, Z: int) -> float:
    """Ion cyclotron frequency ω_ci [rad/s]."""
    if B_gauss <= 0:
        return 0.0
    return abs(Z) * e_charge * B_gauss / mass


def pic_time_step_constraints(omega_pe: float, omega_pi: float,
                               v_th_e: float, dx: float) -> dict:
    """Compute time step constraints for PIC simulation.

    Returns dict of max Δt values based on different criteria.
    """
    constraints = {}
    if omega_pe > 0:
        constraints["electron_plasma"] = 2.0 / omega_pe
    if omega_pi > 0:
        constraints["ion_plasma"] = 2.0 / omega_pi
    if dx > 0 and v_th_e > 0:
        constraints["CFL_electron"] = dx / v_th_e
    constraints["recommended"] = min(constraints.values()) if constraints else 0
    return constraints


def estimate_BBN_magnetic_field(T_MeV: float, n_e: float) -> float:
    """Rough estimate of maximum primordial magnetic field from Biermann battery.

    Returns B in Gauss. This is an upper bound estimate.
    B ~ (k_B T / (e L)) × (n_b / n_e) × (v / c)
    With L ~ c/H (Hubble scale).
    """
    if n_e <= 0:
        return 0.0
    T_K = T_MeV * MeV_to_K
    kBT = k_B * T_K
    # Very rough: B ~ 10^-20 G at T ~ 1 MeV, scaling as T^2
    B_ref = 1e-20  # Gauss at T_ref = 1 MeV
    return B_ref * (T_MeV)**2


def print_ion_summary(data: dict) -> None:
    """Print comprehensive ion parameters at each temperature."""
    print("=" * 120)
    print("BBN Ion Plasma Parameters Summary")
    print("=" * 120)

    header = (f"{'T(MeV)':>10} {'t(s)':>10} {'n_p(cm⁻³)':>14} {'n_He4':>14} "
              f"{'ω_pi(p)':>12} {'ω_pi(α)':>12} {'λ_D(tot)':>12} "
              f"{'v_th(p)/c':>10} {'Γ_ii':>8}")
    print(header)
    print("-" * 120)

    for i in range(len(data["Temp"])):
        T = data["Temp"][i]
        if T < 0.01 or data["Ne"][i] <= 0:
            continue

        rhob = data["rhob"][i]
        ne = data["Ne"][i]

        # Ion densities
        n_p = ion_number_density(rhob, "p", 0.75)
        n_He4 = ion_number_density(rhob, "He4", 0.25)

        # Use He4 fraction appropriate for temperature
        if T > 0.1:
            n_He4 = ion_number_density(rhob, "He4", 1e-10)  # Not yet formed
            n_p = ion_number_density(rhob, "p", 0.9999)

        # Ion plasma frequencies
        omega_pi_p = ion_plasma_frequency(n_p, 1, m_p)
        omega_pi_He4 = ion_plasma_frequency(n_He4, 2, NUCLEONS["He4"]["mass"])

        # Total Debye length (electron + proton)
        lam_D_total = ion_debye_length(T, ne, n_p, 1)

        # Thermal velocity
        v_th_p = thermal_velocity(T, m_p)

        # Ion-ion coupling
        if n_p > 0:
            Gamma_ii = e_charge**2 / ((3/(4*np.pi*n_p))**(1./3.) * k_B * T * MeV_to_K)
        else:
            Gamma_ii = 0

        print(f"{T:>10.3e} {data['time'][i]:>10.3e} {n_p:>14.3e} {n_He4:>14.3e} "
              f"{omega_pi_p:>12.3e} {omega_pi_He4:>12.3e} {lam_D_total:>12.3e} "
              f"{v_th_p/c_light:>10.4f} {Gamma_ii:>8.4f}")


def print_pic_design(data: dict, T_target: float = 0.1) -> None:
    """Print PIC simulation design parameters for a target temperature."""
    # Find closest temperature point
    idx = np.argmin(np.abs(data["Temp"] - T_target))
    T = data["Temp"][idx]
    print(f"\n{'='*60}")
    print(f"PIC Simulation Design at T = {T:.4f} MeV")
    print(f"{'='*60}")

    ne = data["Ne"][idx]
    rhob = data["rhob"][idx]
    H = data["H"][idx]

    n_p = ion_number_density(rhob, "p", 0.75)
    n_He4 = ion_number_density(rhob, "He4", 0.25)

    lam_D = debye_length_electrons(T, ne)
    omega_pe = plasma_frequency(ne, m_e)
    omega_pi_p = ion_plasma_frequency(n_p, 1, m_p)
    omega_pi_He4 = ion_plasma_frequency(n_He4, 2, NUCLEONS["He4"]["mass"])
    v_th_e = thermal_velocity(T, m_e)
    v_th_p = thermal_velocity(T, m_p)

    print(f"\n  Electron parameters:")
    print(f"    n_e = {ne:.3e} cm⁻³")
    print(f"    ω_pe = {omega_pe:.3e} s⁻¹  (f_pe = {omega_pe/2/np.pi:.3e} Hz)")
    print(f"    v_th(e)/c = {v_th_e/c_light:.4f}")
    print(f"    λ_D = {lam_D:.3e} cm")

    print(f"\n  Proton parameters:")
    print(f"    n_p = {n_p:.3e} cm⁻³")
    print(f"    ω_pi(p) = {omega_pi_p:.3e} s⁻¹  (f_pi = {omega_pi_p/2/np.pi:.3e} Hz)")
    print(f"    v_th(p)/c = {v_th_p/c_light:.6f}")
    print(f"    ω_pe/ω_pi = {omega_pe/omega_pi_p:.2e}")

    print(f"\n  Helium-4 parameters:")
    print(f"    n_He4 = {n_He4:.3e} cm⁻³")
    print(f"    ω_pi(α) = {omega_pi_He4:.3e} s⁻¹")

    print(f"\n  Cosmological context:")
    print(f"    H = {H:.3e} s⁻¹")
    print(f"    c/H = {c_light/H:.3e} cm")
    print(f"    ω_pe/H = {omega_pe/H:.2e}")
    print(f"    time = {data['time'][idx]:.3e} s")

    # PIC constraints
    dx = lam_D / 3  # ~3 cells per Debye length
    dt_e = 0.5 / omega_pe  # electron time step
    dt_i = 0.5 / omega_pi_p  # ion time step

    print(f"\n  PIC grid & time step (physical units):")
    print(f"    Δx ≤ {dx:.3e} cm  (λ_D/3)")
    print(f"    Δt(e) ≤ {dt_e:.3e} s  (0.5/ω_pe)")
    print(f"    Δt(i) ≤ {dt_i:.3e} s  (0.5/ω_pi)")
    print(f"    Δt(e)/Δt(i) = {dt_e/dt_i:.2e}  → sub-cycling ratio")

    # Number of particles
    n_cells_debye = 3
    box_debye = 100  # 100 Debye lengths
    n_cells = n_cells_debye * box_debye
    ppc = 50  # particles per cell
    n_macro = n_cells * ppc
    print(f"\n  PIC particle estimate (100 λ_D box, 50 PPC):")
    print(f"    Grid: {n_cells} cells  (1D)")
    print(f"    Macro particles: {n_macro}")


if __name__ == "__main__":
    import sys

    default_path = Path(__file__).parent / "CosmoCons.dat"
    filepath = Path(sys.argv[1]) if len(sys.argv) > 1 else default_path

    data = load_cosmocons(filepath)

    print_ion_summary(data)
    print_pic_design(data, T_target=0.1)
    print_pic_design(data, T_target=0.03)
