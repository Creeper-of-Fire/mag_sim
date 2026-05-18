"""Parse CosmoCons.dat and compute derived physical quantities for BBN plasma simulation."""

import numpy as np
from pathlib import Path

# Physical constants (CGS)
c_light = 2.998e10       # cm/s
G_grav = 6.674e-8        # cm^3/(g s^2)
h_planck = 6.626e-27     # erg s
hbar = 1.055e-27         # erg s
k_B = 1.381e-16          # erg/K
m_p = 1.673e-24          # g
m_e = 9.109e-28          # g
e_charge = 4.803e-10     # esu (CGS)
sigma_T = 6.652e-25      # cm^2 (Thomson cross section)
a_rad = 7.566e-15        # erg/(cm^3 K^4)
MeV_to_K = 1.1605e10     # K per MeV
MeV_to_erg = 1.602e-6    # erg per MeV

COLUMNS = [
    "Temp",      # MeV
    "time",      # s
    "rhog",      # g/cm^3
    "rhoe",      # g/cm^3
    "rhone",     # g/cm^3
    "rhob",      # g/cm^3
    "phie",      # dimensionless
    "rhotot",    # g/cm^3
    "H",         # 1/s
    "Ne",        # 1/cm^3
    "Ng",        # 1/cm^3
    "nLamdg",    # ?
    "LamdD",     # ?
    "c_over_H",  # cm
]


def load_cosmocons(filepath: str | Path) -> dict[str, np.ndarray]:
    """Load CosmoCons.dat into a dict of arrays."""
    filepath = Path(filepath)
    lines = filepath.read_text().strip().split("\n")
    # Skip header lines (column names and dashes)
    data_lines = [l for l in lines if not l.startswith(" ") or l.strip().startswith("Temp")]
    data_lines = [l for l in lines if l.strip() and not l.strip().startswith("Temp")
                  and "---" not in l]

    rows = []
    for line in data_lines:
        vals = line.split()
        if len(vals) == len(COLUMNS):
            rows.append([float(v) if v != "INF" else np.inf for v in vals])

    arr = np.array(rows)
    return {name: arr[:, i] for i, name in enumerate(COLUMNS)}


def t9_from_MeV(T_MeV: float | np.ndarray) -> float | np.ndarray:
    """Convert temperature from MeV to 10^9 K."""
    return T_MeV * MeV_to_K / 1e9


def photon_number_density(t9: float | np.ndarray) -> float | np.ndarray:
    """Photon number density n_gamma [cm^-3] from t9 [10^9 K]."""
    return 20.284 * t9**3


def debye_length_electrons(T_MeV: float, n_e: float) -> float:
    """Debye screening length [cm] for electrons in a plasma.

    λ_D = sqrt(k_B T / (4π e² n_e))
    """
    T_K = T_MeV * MeV_to_K
    kBT = k_B * T_K  # erg
    return np.sqrt(kBT / (4 * np.pi * e_charge**2 * n_e))


def plasma_frequency(n_e: float, mass: float = m_e) -> float:
    """Plasma frequency ω_p [rad/s].

    ω_p = sqrt(4π n_e e² / m)
    """
    return np.sqrt(4 * np.pi * n_e * e_charge**2 / mass)


def thermal_velocity(T_MeV: float, mass_g: float) -> float:
    """Thermal velocity v_th = sqrt(k_B T / m) [cm/s]."""
    T_K = T_MeV * MeV_to_K
    return np.sqrt(k_B * T_K / mass_g)


def coulomb_logarithm(T_MeV: float, n_e: float, Z1: int = 1, Z2: int = 1) -> float:
    """Coulomb logarithm ln(Λ) for plasma.

    ln(Λ) = ln(b_max / b_min)
    b_max = λ_D (Debye length)
    b_min = Z1*Z2*e²/(k_B T) (classical distance of closest approach)
    """
    T_K = T_MeV * MeV_to_K
    kBT = k_B * T_K
    b_min = abs(Z1 * Z2) * e_charge**2 / kBT
    lam_d = debye_length_electrons(T_MeV, n_e)
    if b_min <= 0 or lam_d <= 0:
        return 0.0
    ratio = lam_d / b_min
    if ratio <= 1:
        return 0.0
    return np.log(ratio)


def ion_sound_speed(T_e_MeV: float, T_i_MeV: float,
                    n_e: float, n_i: float,
                    gamma_e: float = 1.0, gamma_i: float = 3.0) -> float:
    """Ion acoustic speed c_s [cm/s].

    c_s = sqrt((γ_e k_B T_e + γ_i k_B T_i) / m_i)
    For BBN: T_e ≈ T_i (single temperature fluid), γ_i = 3 (1D adiabatic).
    """
    kBT_e = T_e_MeV * MeV_to_erg
    kBT_i = T_i_MeV * MeV_to_erg
    return np.sqrt((gamma_e * kBT_e + gamma_i * kBT_i) / m_p)


def degen_parameter(T_MeV: float, n: float, mass: float) -> float:
    """Quantum degeneracy parameter χ = n λ_dB³.

    λ_dB = h / sqrt(2π m k_B T)  (thermal de Broglie wavelength)
    If χ >> 1: quantum degenerate; χ << 1: classical.
    """
    T_K = T_MeV * MeV_to_K
    kBT = k_B * T_K
    lam_db = h_planck / np.sqrt(2 * np.pi * mass * kBT)
    return n * lam_db**3


def coupling_parameter(T_MeV: float, n_e: float) -> float:
    """Plasma coupling parameter Γ = e² / (a k_B T).

    a = (3/(4πn))^{1/3}  (Wigner-Seitz radius)
    Γ << 1: weakly coupled; Γ >> 1: strongly coupled.
    """
    T_K = T_MeV * MeV_to_K
    a_ws = (3.0 / (4 * np.pi * n_e))**(1.0 / 3.0)
    return e_charge**2 / (a_ws * k_B * T_K)


def compute_bbn_plasma_params(data: dict) -> dict:
    """Compute all derived BBN plasma parameters from CosmoCons data."""
    out = {}
    T = data["Temp"]
    Ne = data["Ne"]
    Ng = data["Ng"]
    t9 = t9_from_MeV(T)

    # Temperature in K
    out["T_K"] = T * MeV_to_K

    # Photon number density (verify against data)
    out["Ng_calc"] = photon_number_density(t9)

    # Debye length
    out["lambda_D"] = np.array([
        debye_length_electrons(t, ne) if (ne > 0 and np.isfinite(ne)) else np.nan
        for t, ne in zip(T, Ne)
    ])

    # Plasma frequency
    out["omega_pe"] = np.array([
        plasma_frequency(ne, m_e) if (ne > 0 and np.isfinite(ne)) else np.nan
        for ne in Ne
    ])
    out["f_pe"] = out["omega_pe"] / (2 * np.pi)  # Hz

    # Ion plasma frequency
    n_baryon = data["rhob"] / m_p
    out["omega_pi"] = np.array([
        plasma_frequency(nb, m_p) if (nb > 0 and np.isfinite(nb)) else np.nan
        for nb in n_baryon
    ])

    # Thermal velocities
    out["v_th_e"] = np.array([
        thermal_velocity(t, m_e) if t > 0 else np.nan for t in T
    ])
    out["v_th_p"] = np.array([
        thermal_velocity(t, m_p) if t > 0 else np.nan for t in T
    ])

    # Coulomb logarithm
    out["ln_Lambda"] = np.array([
        coulomb_logarithm(t, ne) if (ne > 0 and np.isfinite(ne) and t > 0) else np.nan
        for t, ne in zip(T, Ne)
    ])

    # Coupling parameter
    out["Gamma"] = np.array([
        coupling_parameter(t, ne) if (ne > 0 and np.isfinite(ne) and t > 0) else np.nan
        for t, ne in zip(T, Ne)
    ])

    # Degeneracy parameter (electrons)
    out["chi_e"] = np.array([
        degen_parameter(t, ne, m_e) if (ne > 0 and np.isfinite(ne) and t > 0) else np.nan
        for t, ne in zip(T, Ne)
    ])

    # Ion acoustic speed (assuming T_e = T_i = T)
    out["c_s"] = np.array([
        ion_sound_speed(t, t, ne, ne) if (t > 0 and np.isfinite(ne) and ne > 0) else np.nan
        for t, ne in zip(T, Ne)
    ])

    # Baryon number density
    out["n_baryon"] = n_baryon

    return out


def print_summary(data: dict, params: dict) -> None:
    """Print a summary table of key plasma parameters at selected temperatures."""
    print(f"{'T(MeV)':>10} {'time(s)':>12} {'n_e(cm⁻³)':>14} {'λ_D(cm)':>12} "
          f"{'ω_pe(s⁻¹)':>12} {'Γ':>10} {'ln(Λ)':>8} {'c/H(cm)':>14}")
    print("-" * 102)

    for i in range(len(data["Temp"])):
        T = data["Temp"][i]
        ne = data["Ne"][i]
        lam = params["lambda_D"][i]
        wp = params["omega_pe"][i]
        gamma = params["Gamma"][i]
        lnL = params["ln_Lambda"][i]
        cH = data["c_over_H"][i]

        ne_str = f"{ne:.3e}" if np.isfinite(ne) and ne > 0 else "0"
        lam_str = f"{lam:.3e}" if np.isfinite(lam) else "---"
        wp_str = f"{wp:.3e}" if np.isfinite(wp) else "---"
        g_str = f"{gamma:.4f}" if np.isfinite(gamma) else "---"
        ln_str = f"{lnL:.2f}" if np.isfinite(lnL) else "---"
        cH_str = f"{cH:.3e}" if np.isfinite(cH) else "---"

        print(f"{T:>10.3e} {data['time'][i]:>12.3e} {ne_str:>14} {lam_str:>12} "
              f"{wp_str:>12} {g_str:>10} {ln_str:>8} {cH_str:>14}")


if __name__ == "__main__":
    import sys

    default_path = Path(__file__).parent / "CosmoCons.dat"
    filepath = sys.argv[1] if len(sys.argv) > 1 else default_path

    print(f"Loading {filepath} ...")
    data = load_cosmocons(filepath)
    params = compute_bbn_plasma_params(data)

    print(f"\nLoaded {len(data['Temp'])} data points")
    print(f"Temperature range: {data['Temp'].min():.4e} — {data['Temp'].max():.4e} MeV")
    print(f"Time range: {data['time'].min():.4e} — {data['time'].max():.4e} s\n")

    print_summary(data, params)
