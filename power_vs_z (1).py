"""
Power transfer analysis of a directional coupler (TM and TE), 4 plots per polarisation:

  (1) P2(z) for several GAPS at fixed d_beta            (+ dashed line at target power)
  (2) P2(z) for several d_beta at fixed gap             (+ dashed line at target power)
  (3) distance needed to reach the target power vs GAP, one curve per d_beta
  (4) distance needed to reach the target power vs d_beta, one curve per GAP

Coupled-mode theory:  P2(z) = kappa^2/(kappa^2+delta^2) * sin^2( sqrt(kappa^2+delta^2) * z ),
                      delta = d_beta/2.
Distance to reach target power P (first time):
                      z_P = asin( sqrt(P/F) ) / sqrt(kappa^2+delta^2),  F = kappa^2/(kappa^2+delta^2)
                      (does not exist if F < P: that curve simply stops)
kappa(gap) comes from the full-vector solver (identical waveguides),
kappa = pi*(n_even - n_odd)/wavelength.

Needs dc_supermodes_fullvec.py in the same folder.   Units: micrometres.
Run:  python power_vs_z.py     ->  power_vs_z_TM.png , power_vs_z_TE.png
"""
import numpy as np
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

TARGET_POWER = 0.30        # fraction of power to transfer (dashed line + plots 3 and 4)

# kappa is solved at these gaps, then interpolated (exponentially) for the sweeps in plots 3, 4
KAPPA_GAPS = np.arange(0.10, 0.501, 0.05)          # um   (each point = 1 solver run)

SETTINGS = {
    "TM": dict(
        # plots 1 & 2 -------------------------------------------------
        gaps        = [0.15, 0.20, 0.25, 0.30, 0.40],   # um   plot 1 curves
        dbeta_fixed = 0.30,                              # 1/um plot 1 fixed mismatch
        dbetas      = [0.0, 0.1, 0.3, 0.5, 0.8],         # 1/um plot 2 curves
        gap_fixed   = 0.20,                              # um   plot 2 fixed gap
        zmax        = 40.0,                              # um   z range of plots 1, 2
        # plot 3: z_P vs gap, curves = d_beta ------------------------
        gap_range   = (0.10, 0.50),                      # um   x range
        dbetas_p3   = [0.0, 0.2, 0.4, 0.6, 0.8],         # 1/um curves
        # plot 4: z_P vs d_beta, curves = gap ------------------------
        dbeta_range = (0.0, 1.5),                        # 1/um x range
        gaps_p4     = [0.15, 0.20, 0.25, 0.30, 0.40],    # um   curves
    ),
    "TE": dict(
        gaps        = [0.10, 0.15, 0.20, 0.25, 0.30],
        dbeta_fixed = 0.10,
        dbetas      = [0.0, 0.05, 0.10, 0.20, 0.40],
        gap_fixed   = 0.20,
        zmax        = 100.0,
        gap_range   = (0.10, 0.40),
        dbetas_p3   = [0.0, 0.03, 0.06, 0.1, 0.15],
        dbeta_range = (0.0, 0.5),
        gaps_p4     = [0.10, 0.15, 0.20, 0.25, 0.30],
    ),
}
N_POINTS = 800             # points along z
# =====================================================================================

# ---------------------------------------------------------------- physics helpers
_cache = {}
def kappa_solver(gap, pol):
    """kappa (1/um) from the full-vector solver; cached."""
    key = (round(float(gap), 6), pol)
    if key not in _cache:
        r = supermodes(w=WIDTH, h=HEIGHT, gap=gap, wl=WAVELENGTH, n_core=N_CORE, n_sub=N_SUB,
                       n_clad=N_CLAD, pol=pol, dx=GRID_STEP, dy=GRID_STEP)
        _cache[key] = np.pi * (r["even"][0] - r["odd"][0]) / WAVELENGTH
    return _cache[key]

def kappa_interp(gap, pol):
    """kappa at arbitrary gap: log-interpolation of solver results on KAPPA_GAPS."""
    ks = np.array([kappa_solver(g, pol) for g in KAPPA_GAPS])
    return np.exp(np.interp(gap, KAPPA_GAPS, np.log(ks)))

def P2(z, kappa, dbeta):
    S = np.sqrt(kappa**2 + (dbeta / 2) ** 2)
    return (kappa**2 / S**2) * np.sin(S * z) ** 2

def z_target(kappa, dbeta, P=TARGET_POWER):
    """First distance at which P2 = P (NaN where the max transfer F < P)."""
    kappa, dbeta = np.broadcast_arrays(np.asarray(kappa, float), np.asarray(dbeta, float))
    S = np.sqrt(kappa**2 + (dbeta / 2) ** 2)
    F = kappa**2 / S**2
    out = np.full(S.shape, np.nan)
    ok = F >= P
    out[ok] = np.arcsin(np.sqrt(P / F[ok])) / S[ok]
    return out

# ---------------------------------------------------------------- plotting
def _colors(name, n, top=0.85):
    cm = plt.get_cmap(name)
    return [cm(i / max(n - 1, 1) * top) for i in range(n)]

def plot_pol(pol, cfg, fname):
    P = TARGET_POWER
    z = np.linspace(0, cfg["zmax"], N_POINTS)
    fig, ax = plt.subplots(2, 2, figsize=(14, 9.5))
    (a1, a2), (a3, a4) = ax
    tl = f"{int(round(P*100))}% target"

    # ---- (1) different gaps, fixed d_beta
    for c, g in zip(_colors("plasma", len(cfg["gaps"])), cfg["gaps"]):
        k = kappa_solver(g, pol)
        a1.plot(z, P2(z, k, cfg["dbeta_fixed"]), color=c, lw=2,
                label=rf"gap = {g*1e3:.0f} nm  ($\kappa$ = {k:.3f} $\mu$m$^{{-1}}$)")
    a1.set_title(f"{pol}: different gaps, fixed "
                 rf"$\Delta\beta$ = {cfg['dbeta_fixed']:g} $\mu$m$^{{-1}}$")

    # ---- (2) different d_beta, fixed gap
    k = kappa_solver(cfg["gap_fixed"], pol)
    for c, db in zip(_colors("viridis", len(cfg["dbetas"]), 0.9), cfg["dbetas"]):
        a2.plot(z, P2(z, k, db), color=c, lw=2,
                label=rf"$\Delta\beta$ = {db:g} $\mu$m$^{{-1}}$  (max {100*k**2/(k**2+(db/2)**2):.0f}%)")
    a2.set_title(f"{pol}: different "
                 rf"$\Delta\beta$, fixed gap = {cfg['gap_fixed']*1e3:.0f} nm "
                 rf"($\kappa$ = {k:.3f} $\mu$m$^{{-1}}$)")
    for a in (a1, a2):
        a.axhline(P, color="k", ls="--", lw=1.3, label=tl)
        a.set_xlabel(r"Propagation distance  z  ($\mu$m)")
        a.set_ylabel("Normalised power in waveguide 2")
        a.set_xlim(0, cfg["zmax"]); a.set_ylim(0, 1.4)
        a.set_yticks(np.arange(0, 1.01, 0.2)); a.grid(alpha=.3)
        a.legend(fontsize=8, loc="upper right")

    # ---- (3) z_P vs gap, curves = d_beta
    gg = np.linspace(*cfg["gap_range"], 200)
    kk = kappa_interp(gg, pol)
    for c, db in zip(_colors("viridis", len(cfg["dbetas_p3"]), 0.9), cfg["dbetas_p3"]):
        a3.plot(gg * 1e3, z_target(kk, db), color=c, lw=2,
                label=rf"$\Delta\beta$ = {db:g} $\mu$m$^{{-1}}$")
    a3.set_xlabel("Gap between waveguides  (nm)")
    a3.set_ylabel(rf"Distance for {int(round(P*100))}% power transfer  ($\mu$m)")
    a3.set_title(f"{pol}: distance for {int(round(P*100))}% transfer vs gap\n"
                 r"(curve ends where max transfer $<$ target)")

    # ---- (4) z_P vs d_beta, curves = gap
    dd = np.linspace(*cfg["dbeta_range"], 400)
    for c, g in zip(_colors("plasma", len(cfg["gaps_p4"])), cfg["gaps_p4"]):
        k = kappa_solver(g, pol)
        a4.plot(dd, z_target(k, dd), color=c, lw=2, label=f"gap = {g*1e3:.0f} nm")
    a4.set_xlabel(r"Phase mismatch  $\Delta\beta$  ($\mu$m$^{-1}$)")
    a4.set_ylabel(rf"Distance for {int(round(P*100))}% power transfer  ($\mu$m)")
    a4.set_title(f"{pol}: distance for {int(round(P*100))}% transfer vs "
                 r"$\Delta\beta$" "\n(curve ends where max transfer $<$ target)")
    for a in (a3, a4):
        a.grid(alpha=.3); a.legend(fontsize=8, loc="best"); a.set_xlim(left=a.get_xlim()[0]); a.set_ylim(bottom=0)

    fig.suptitle(f"{pol} mode: power transfer analysis  "
                 f"(w = {WIDTH*1e3:.0f} nm, h = {HEIGHT*1e3:.0f} nm, "
                 rf"$\lambda$ = {WAVELENGTH*1e3:.0f} nm)", fontsize=13)
    fig.tight_layout(); fig.savefig(fname, dpi=140); plt.close(fig)
    print("saved", fname)

if __name__ == "__main__":
    for pol in ("TM", "TE"):
        plot_pol(pol, SETTINGS[pol], f"power_vs_z_{pol}.png")
