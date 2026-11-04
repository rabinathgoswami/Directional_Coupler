"""
Normalised power transferred to waveguide 2 vs propagation distance z, for TM and TE.

  Left panel : several GAPS, one fixed phase mismatch d_beta
  Right panel: several PHASE MISMATCHES d_beta, one fixed gap

Coupled-mode theory:   P2(z) = kappa^2/(kappa^2+delta^2) * sin^2( sqrt(kappa^2+delta^2) * z )
                       delta = d_beta/2 ,  d_beta = beta1 - beta2
kappa(gap) is computed with the full-vector solver (identical waveguides),
kappa = pi*(n_even - n_odd)/wavelength.

Needs dc_supermodes_fullvec.py in the same folder.   Units: micrometres.
Run:  python power_vs_z.py     ->  power_vs_z_TM.png , power_vs_z_TE.png
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from dc_supermodes_fullvec import supermodes

# =====================================================================================
#  >>>>>>>>>>>>>>>>>>>>>>>>>>>>   EDIT THESE VALUES   <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# =====================================================================================
WAVELENGTH = 1.55          # um
WIDTH      = 0.45          # um  core width used to compute kappa(gap)
HEIGHT     = 0.22          # um  core height
N_CORE, N_SUB, N_CLAD = 3.476, 1.444, 1.444
GRID_STEP  = 0.02          # um  (smaller = more accurate but much slower)

SETTINGS = {
    "TM": dict(
        gaps        = [0.15, 0.20, 0.25, 0.30, 0.40],   # um   -> left panel curves
        dbeta_fixed = 0.30,                              # 1/um -> left panel (fixed mismatch)
        dbetas      = [0.0, 0.1, 0.3, 0.5, 0.8],         # 1/um -> right panel curves
        gap_fixed   = 0.20,                              # um   -> right panel (fixed gap)
        zmax        = 40.0,                              # um   plot range
    ),
    "TE": dict(
        gaps        = [0.10, 0.15, 0.20, 0.25, 0.30],
        dbeta_fixed = 0.10,
        dbetas      = [0.0, 0.05, 0.10, 0.20, 0.40],
        gap_fixed   = 0.20,
        zmax        = 100.0,
    ),
}
N_POINTS = 800             # points along z
# =====================================================================================

_kappa_cache = {}
def kappa_of_gap(gap, pol):
    """Coupling coefficient (1/um) from the full-vector solver; results are cached."""
    key = (round(gap, 6), pol)
    if key not in _kappa_cache:
        r = supermodes(w=WIDTH, h=HEIGHT, gap=gap, wl=WAVELENGTH, n_core=N_CORE, n_sub=N_SUB,
                       n_clad=N_CLAD, pol=pol, dx=GRID_STEP, dy=GRID_STEP)
        _kappa_cache[key] = np.pi * (r["even"][0] - r["odd"][0]) / WAVELENGTH
    return _kappa_cache[key]

def P2(z, kappa, dbeta):
    """Normalised power in waveguide 2 (light launched into waveguide 1)."""
    delta = dbeta / 2.0
    S = np.sqrt(kappa**2 + delta**2)
    return (kappa**2 / S**2) * np.sin(S * z) ** 2

def plot_power_vs_z(pol, cfg, fname):
    z = np.linspace(0, cfg["zmax"], N_POINTS)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)

    # ---- left: different gaps, fixed d_beta
    cm = plt.get_cmap("plasma")
    gaps = cfg["gaps"]
    for i, g in enumerate(gaps):
        k = kappa_of_gap(g, pol)
        a1.plot(z, P2(z, k, cfg["dbeta_fixed"]), color=cm(i / max(len(gaps) - 1, 1) * 0.85), lw=2,
                label=rf"gap = {g*1e3:.0f} nm  ($\kappa$ = {k:.3f} $\mu$m$^{{-1}}$)")
    a1.set_title(f"{pol}: different gaps, fixed "
                 rf"$\Delta\beta$ = {cfg['dbeta_fixed']:g} $\mu$m$^{{-1}}$")

    # ---- right: different d_beta, fixed gap
    k = kappa_of_gap(cfg["gap_fixed"], pol)
    cm = plt.get_cmap("viridis")
    dbs = cfg["dbetas"]
    for i, db in enumerate(dbs):
        a2.plot(z, P2(z, k, db), color=cm(i / max(len(dbs) - 1, 1) * 0.9), lw=2,
                label=rf"$\Delta\beta$ = {db:g} $\mu$m$^{{-1}}$  (max {100*k**2/(k**2+(db/2)**2):.0f}%)")
    a2.set_title(f"{pol}: different "
                 rf"$\Delta\beta$, fixed gap = {cfg['gap_fixed']*1e3:.0f} nm "
                 rf"($\kappa$ = {k:.3f} $\mu$m$^{{-1}}$)")

    for a in (a1, a2):
        a.set_xlabel(r"Propagation distance  z  ($\mu$m)")
        a.set_xlim(0, cfg["zmax"]); a.set_ylim(0, 1.4); a.set_yticks(np.arange(0, 1.01, 0.2)); a.grid(alpha=.3)
        a.legend(fontsize=8, loc="upper right")
    a1.set_ylabel("Normalised power in waveguide 2")
    fig.suptitle(f"{pol} mode: power transfer along the coupler  "
                 f"(w = {WIDTH*1e3:.0f} nm, h = {HEIGHT*1e3:.0f} nm, "
                 rf"$\lambda$ = {WAVELENGTH*1e3:.0f} nm)", fontsize=12)
    fig.tight_layout(); fig.savefig(fname, dpi=140); plt.close(fig)
    print("saved", fname)

if __name__ == "__main__":
    for pol in ("TM", "TE"):
        plot_power_vs_z(pol, SETTINGS[pol], f"power_vs_z_{pol}.png")
