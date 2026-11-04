"""
Two sweep/plot tools built on dc_supermodes_fullvec.py (keep both files in one folder).

1) plot_beta_vs_width      : propagation constant (and n_eff) of an isolated strip vs width
2) plot_fmax_vs_gap        : maximum power transfer vs gap, colour-coded by phase mismatch d_beta
Units: micrometres.
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from dc_supermodes_fullvec import make_eps_rects, solve, _pick_modes, supermodes

# ---------------------------------------------------------------- helpers
def isolated_neff(w, h=0.22, wl=1.55, pol="TE", n_core=3.476, n_sub=1.444, n_clad=1.444,
                  dx=0.02, dy=0.02, margin=1.2, below=1.2, above=1.2, nmodes=6):
    """n_eff of the fundamental mode of one strip (width w, height h) on SiO2."""
    Nx = int(round((w + 2 * margin) / dx))
    Ny = int(round((below + h + above) / dy))
    x0 = -Nx / 2 * dx
    eps = make_eps_rects([(-w / 2, w / 2, 0.0, h)], n_core, n_sub, n_clad)
    ne, F, _, _ = solve(eps, x0, -below, Nx, Ny, dx, dy, wl, n_core * 0.8, nmodes=nmodes)
    return ne[_pick_modes(ne, F, pol, 1)[0]]

# ---------------------------------------------------------------- 1) beta vs width
def plot_beta_vs_width(widths=np.arange(0.36, 0.621, 0.02), h=0.22, wl=1.55,
                       pols=("TE", "TM"), w_ref=None, fname="beta_vs_width.png", **kw):
    """Left axis: beta = 2*pi*n_eff/wl (1/um).  Right axis: n_eff.
    w_ref: optional reference width -> horizontal/vertical guide lines (e.g. WG1's width),
           to read off which width gives zero phase mismatch."""
    k0 = 2 * np.pi / wl
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    colors = {"TE": "tab:blue", "TM": "tab:red"}
    results = {}
    for pol in pols:
        n = np.array([isolated_neff(w, h, wl, pol, **kw) for w in widths])
        results[pol] = n
        ax.plot(widths * 1e3, k0 * n, "o-", color=colors.get(pol), label=f"{pol}$_0$ mode")
        if w_ref is not None:
            nref = np.interp(w_ref, widths, n)
            ax.plot([w_ref * 1e3], [k0 * nref], "s", ms=9, mfc="none", color=colors.get(pol),
                    label=f"{pol}: reference w = {w_ref*1e3:.0f} nm")
            ax.axhline(k0 * nref, color=colors.get(pol), ls=":", lw=1)
    if w_ref is not None:
        ax.axvline(w_ref * 1e3, color="gray", ls=":", lw=1)
    ax.set_xlabel("Waveguide width  w  (nm)")
    ax.set_ylabel(r"Propagation constant  $\beta$  ($\mu$m$^{-1}$)")
    ax.set_title(f"Propagation constant vs waveguide width\n"
                 f"(SOI strip, h = {h*1e3:.0f} nm, $\\lambda$ = {wl*1e3:.0f} nm, SiO$_2$ clad)")
    ax.grid(alpha=.3)
    sec = ax.secondary_yaxis("right", functions=(lambda b: b / k0, lambda n: n * k0))
    sec.set_ylabel(r"Effective index  $n_{eff}$")
    ax.legend(loc="best")
    fig.tight_layout(); fig.savefig(fname, dpi=140); plt.close(fig)
    return widths, results

# ---------------------------------------------------------------- 2) Fmax vs gap
def kappa_vs_gap(gaps, w=0.45, h=0.22, wl=1.55, pol="TE", dx=0.02, **kw):
    """kappa(gap) from identical waveguides:  kappa = pi*(n_even - n_odd)/wl  (1/um)."""
    kap = []
    for g in gaps:
        r = supermodes(w=w, h=h, gap=g, wl=wl, pol=pol, dx=dx, dy=dx, **kw)
        kap.append(np.pi * (r["even"][0] - r["odd"][0]) / wl)
    return np.array(kap)

def plot_fmax_vs_gap(gaps=np.arange(0.10, 0.501, 0.05), dbetas=(0.0, 0.05, 0.1, 0.2, 0.3, 0.5),
                     w=0.45, h=0.22, wl=1.55, pol="TE", fname="fmax_vs_gap.png", **kw):
    """Maximum power transfer  F = kappa^2 / (kappa^2 + (d_beta/2)^2)  vs edge-to-edge gap,
    one curve per phase mismatch d_beta (colour-coded).  kappa(gap) comes from the
    full-vector solver (identical waveguides); d_beta is imposed."""
    kappa = kappa_vs_gap(gaps, w, h, wl, pol, **kw)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.8))

    a1.semilogy(gaps * 1e3, kappa, "ko-")
    a1.set_xlabel("Gap  (nm)"); a1.set_ylabel(r"Coupling coefficient  $\kappa$  ($\mu$m$^{-1}$)")
    a1.set_title(f"{pol}: coupling coefficient vs gap\n(identical waveguides, w = {w*1e3:.0f} nm)")
    a1.grid(alpha=.3, which="both")

    cmap = plt.get_cmap("viridis")
    norm = matplotlib.colors.Normalize(vmin=min(dbetas), vmax=max(dbetas))
    gfine = np.linspace(gaps.min(), gaps.max(), 300)
    kfine = np.exp(np.interp(gfine, gaps, np.log(kappa)))        # log-interp (kappa ~ exp(-gap))
    for db in dbetas:
        F = kfine**2 / (kfine**2 + (db / 2) ** 2)
        a2.plot(gfine * 1e3, 100 * F, color=cmap(norm(db)), lw=2,
                label=rf"$\Delta\beta$ = {db:g} $\mu$m$^{{-1}}$")
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, ax=a2); cb.set_label(r"Phase mismatch  $\Delta\beta = \beta_1-\beta_2$  ($\mu$m$^{-1}$)")
    a2.set_xlabel("Gap  (nm)"); a2.set_ylabel("Maximum power transfer  (%)")
    a2.set_ylim(0, 102)
    a2.set_title(f"{pol}: maximum power transfer vs gap\n"
                 r"$F=\kappa^2/(\kappa^2+(\Delta\beta/2)^2)$")
    a2.grid(alpha=.3); a2.legend(fontsize=8, loc="center right")
    fig.tight_layout(); fig.savefig(fname, dpi=140); plt.close(fig)
    return gaps, kappa

if __name__ == "__main__":
    plot_beta_vs_width(w_ref=0.45)
    for pol in ("TE", "TM"):
        plot_fmax_vs_gap(pol=pol, fname=f"fmax_vs_gap_{pol}.png")
