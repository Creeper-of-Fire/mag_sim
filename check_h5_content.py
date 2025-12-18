# compare_diagnostics.py
import os
import h5py
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

# =============================================================================
# --- USER CONFIGURATION ---
# =============================================================================
# --- 1. Set the path to the simulation results directory ---
SIM_DIR = "sim_result/无初始场"

# --- 2. Set the exact timestep (integer) you want to compare ---
#      Use a non-zero step where thermal fluctuations should exist.
STEP_TO_COMPARE = 3920
# =============================================================================

console = Console()


def _center_field_3d(field: np.ndarray, target_shape: tuple) -> np.ndarray:
    """Helper function to center staggered field data."""
    if field.shape == target_shape: return field
    nx, ny, nz = target_shape
    if field.shape == (nx + 1, ny, nz): return 0.5 * (field[:-1, :, :] + field[1:, :, :])
    if field.shape == (nx, ny + 1, nz): return 0.5 * (field[:, :-1, :] + field[:, 1:, :])
    if field.shape == (nx, ny, nz + 1): return 0.5 * (field[:, :, :-1] + field[:, :, 1:])
    # Add other staggered shapes if necessary
    console.print(f"[yellow]Warning: Unhandled shape {field.shape} for centering, returning as-is.[/yellow]")
    return field


def load_npz_data(filepath: str) -> dict:
    """Loads and returns all relevant arrays from the .npz file."""
    if not os.path.exists(filepath):
        return None
    with np.load(filepath) as data:
        return {key: data[key] for key in data.files}


def load_h5_data(filepath: str, step: int, target_shape: tuple) -> dict:
    """Loads and returns all relevant field arrays from the .h5 file."""
    if not os.path.exists(filepath):
        return None

    data = {}
    with h5py.File(filepath, 'r') as f:
        field_group = f[f"/data/{step}/fields/"]

        for field_char in ['B', 'E', 'j']:
            if field_char in field_group:
                for component in ['x', 'y', 'z']:
                    path = f"{field_char}/{component}"
                    full_key = f"{field_char}{component}"  # e.g., 'Bx'
                    if path in field_group:
                        data[full_key] = field_group[path][:]
                    else:
                        data[full_key] = np.zeros(target_shape)  # Assume zero if not present
    return data


def main():
    console.print(Panel(f"[bold]Comparing Diagnostics for Step [cyan]{STEP_TO_COMPARE}[/cyan] in '{SIM_DIR}'[/bold]",
                        title="Diagnostic Probe", border_style="green"))

    # --- 1. Construct file paths ---
    npz_path = os.path.join(SIM_DIR, "diags/fields", f"fields_{STEP_TO_COMPARE:06d}.npz")
    h5_path = os.path.join(SIM_DIR, "diags/field_states", f"openpmd_{STEP_TO_COMPARE:06d}.h5")

    # --- 2. Load data from both sources ---
    npz_data = load_npz_data(npz_path)
    if npz_data is None:
        console.print(f"[red]Error: NPZ file not found at {npz_path}[/red]")
        return

    # We need a target shape for the H5 loader in case a field is missing.
    # We can guess it from the Bx shape in the NPZ file (which is staggered).
    try:
        s = npz_data['Bx'].shape
        target_shape_centered = (s[0] - 1, s[1], s[2])
    except:
        console.print("[red]Could not determine target shape from NPZ file.[/red]")
        return

    h5_data = load_h5_data(h5_path, STEP_TO_COMPARE, target_shape_centered)
    if h5_data is None:
        console.print(f"[red]Error: HDF5 file not found at {h5_path}[/red]")
        return

    b_norm = npz_data.get('B_norm', 1.0)  # Get normalization constant

    # --- 3. Perform and display comparison ---
    table = Table(title="Field Component Comparison")
    table.add_column("Component", justify="center", style="cyan")
    table.add_column("Source", justify="center", style="magenta")
    table.add_column("Data Shape", justify="right", style="white")
    table.add_column("Min Value", justify="right")
    table.add_column("Max Value", justify="right")
    table.add_column("Mean Value", justify="right")
    table.add_column("RMS", justify="right", style="yellow")

    components_to_check = ['Bx', 'By', 'Bz', 'Ex', 'Ey', 'Ez']

    for comp in components_to_check:
        table.add_section()

        # --- NPZ Data Analysis ---
        data_npz = npz_data.get(comp)
        if data_npz is not None:
            rms_npz = np.sqrt(np.mean(data_npz ** 2))
            table.add_row(comp, ".npz (normalized)", str(data_npz.shape),
                          f"{np.min(data_npz):.3e}", f"{np.max(data_npz):.3e}",
                          f"{np.mean(data_npz):.3e}", f"{rms_npz:.3e}")

        # --- H5 Data Analysis ---
        data_h5_si = h5_data.get(comp)
        if data_h5_si is not None:
            # Normalize H5 data for fair comparison
            data_h5_norm = data_h5_si / b_norm if 'B' in comp else data_h5_si  # Assuming E is already SI and doesn't need scaling against B_norm
            rms_h5 = np.sqrt(np.mean(data_h5_norm ** 2))
            table.add_row(comp, ".h5 (normalized)", str(data_h5_norm.shape),
                          f"{np.min(data_h5_norm):.3e}", f"{np.max(data_h5_norm):.3e}",
                          f"{np.mean(data_h5_norm):.3e}", f"{rms_h5:.3e}")

        # --- Direct Difference Calculation ---
        if data_npz is not None and data_h5_norm is not None:
            # We must center the staggered NPZ data to compare with centered H5 data
            data_npz_centered = _center_field_3d(data_npz, data_h5_norm.shape)

            if data_npz_centered.shape == data_h5_norm.shape:
                diff = data_npz_centered - data_h5_norm
                mean_abs_err = np.mean(np.abs(diff))
                max_abs_err = np.max(np.abs(diff))

                summary = Text.assemble(
                    (f"Difference Analysis for {comp}:\n", "bold"),
                    ("  Mean Absolute Error: ", "white"), (f"{mean_abs_err:.3e}\n", "yellow"),
                    ("  Max Absolute Error:  ", "white"), (f"{max_abs_err:.3e}", "red")
                )
                console.print(Panel(summary, border_style="blue", expand=False))
            else:
                console.print(f"[red]Could not compare {comp} directly due to shape mismatch after centering.[/red]")

    console.print(table)

    console.print(Panel(
        "[bold]Summary:[/bold]\n"
        "1. The '.npz' source is from the custom `check_fields` callback, grabbing data directly from memory.\n"
        "2. The '.h5' source is from the standard `picmi.FieldDiagnostic` OpenPMD output.\n"
        "3. [yellow]Key Observation:[/yellow] Note the difference in [bold]Data Shape[/bold]. The '.npz' data is likely on a [underline]staggered[/underline] Yee grid, while the '.h5' data has been interpolated to the [underline]cell center[/underline] before saving.\n"
        "4. The 'Difference Analysis' compares the centered '.npz' data against the '.h5' data. Any remaining difference is due to the interpolation scheme used by OpenPMD vs. our simple averaging.",
        title="[bold]Probe Analysis Conclusion[/bold]", border_style="red"
    ))


if __name__ == "__main__":
    main()