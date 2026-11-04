"""
Full-vector Maxwell eigenmode solver (Yee-grid finite difference) for
SOI directional-coupler supermodes.  Units: micrometres.

Formulation
-----------
Fields ~ exp(-j*beta*z), derivatives normalised by k0, n = beta/k0.
Maxwell curl equations with H~ = -j*eta0*H give
    curl E = H~ ,   curl H~ = eps * E ,   div H~ = 0
Eliminating E, Hz yields an eigenproblem for n^2 in the transverse H field:
    n^2 Hx = eyy Hx + eyy Dy ezz^-1 (Dy Hx - Dx Hy) + Dx (Dx Hx + Dy Hy)
    n^2 Hy = exx Hy + Dy (Dx Hx + Dy Hy) + exx Dx ezz^-1 (Dx Hy - Dy Hx)
Yee layout (cell i,j):  Ez(i,j)  Ex(i+1/2,j)  Ey(i,j+1/2)
                        Hx(i,j+1/2)  Hy(i+1/2,j)  Hz(i+1/2,j+1/2)
so each derivative is a forward (f) or backward (b) difference that lands exactly
on the right sub-lattice.  Anisotropic sub-pixel averaging is used for eps.
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla

# ------------------------------------------------------------------ geometry
def make_eps_func(w, h, gap, n_core, n_sub, n_clad):
    """Two identical strips on a SiO2 substrate, symmetric about x = 0.
    Substrate: y < 0.  Cores: 0 <= y <= h.  Everything else: cladding."""
    xc = (gap + w) / 2.0
    def eps(X, Y):
        e = np.full(X.shape, n_clad**2)
        e[Y < 0] = n_sub**2
        for s in (-1, 1):
            m = (np.abs(X - s * xc) <= w / 2) & (Y >= 0) & (Y <= h)
            e[m] = n_core**2
        return e
    return eps

def _avg_eps(eps_func, xs, ys, dx, dy, mode, ss=4):
    """Sub-pixel averaged permittivity on cells centred at (xs, ys) (2-D arrays).
    mode: 'xx' harmonic in x / arithmetic in y; 'yy' the opposite; 'zz' arithmetic."""
    off = (np.arange(ss) + 0.5) / ss - 0.5
    acc = np.zeros((ss, ss) + xs.shape)
    for a, ox in enumerate(off):
        for b, oy in enumerate(off):
            acc[a, b] = eps_func(xs + ox * dx, ys + oy * dy)
    if mode == "xx":
        return np.mean(1.0 / np.mean(1.0 / acc, axis=0), axis=0)
    if mode == "yy":
        return np.mean(1.0 / np.mean(1.0 / acc, axis=1), axis=0)
    return acc.mean(axis=(0, 1))

# ------------------------------------------------------------------ solver
def solve(eps_func, x0, y0, Nx, Ny, dx, dy, wl, n_guess, nmodes=8):
    """Return n_eff (nmodes,), Hx, Hy (Nx, Ny, nmodes) and the node coordinate vectors."""
    k0 = 2 * np.pi / wl
    xn = x0 + dx * np.arange(Nx)          # node x
    yn = y0 + dy * np.arange(Ny)          # node y
    XN, YN = np.meshgrid(xn, yn, indexing="ij")

    exx = _avg_eps(eps_func, XN + dx / 2, YN, dx, dy, "xx").ravel()   # Ex (i+1/2, j)
    eyy = _avg_eps(eps_func, XN, YN + dy / 2, dx, dy, "yy").ravel()   # Ey (i, j+1/2)
    ezz = _avg_eps(eps_func, XN, YN, dx, dy, "zz").ravel()            # Ez (i, j)

    def fwd(N, d):  return sp.diags([-np.ones(N), np.ones(N - 1)], [0, 1]) / (d * k0)
    def bwd(N, d):  return sp.diags([np.ones(N), -np.ones(N - 1)], [0, -1]) / (d * k0)
    Ix, Iy = sp.identity(Nx), sp.identity(Ny)
    Dxf, Dxb = sp.kron(fwd(Nx, dx), Iy), sp.kron(bwd(Nx, dx), Iy)
    Dyf, Dyb = sp.kron(Ix, fwd(Ny, dy)), sp.kron(Ix, bwd(Ny, dy))

    Exx_, Eyy_, Ezi = sp.diags(exx), sp.diags(eyy), sp.diags(1.0 / ezz)

    Axx = Eyy_ + Eyy_ @ Dyf @ Ezi @ Dyb + Dxb @ Dxf
    Axy = -Eyy_ @ Dyf @ Ezi @ Dxb + Dxb @ Dyf
    Ayx = Dyb @ Dxf - Exx_ @ Dxf @ Ezi @ Dyb
    Ayy = Exx_ + Dyb @ Dyf + Exx_ @ Dxf @ Ezi @ Dxb
    A = sp.bmat([[Axx, Axy], [Ayx, Ayy]]).tocsc()

    vals, vecs = sla.eigs(A, k=nmodes, sigma=n_guess**2, which="LM")
    order = np.argsort(-vals.real)
    neff = np.sqrt(vals.real[order])
    vecs = vecs[:, order].real
    N = Nx * Ny
    hx, hy = vecs[:N], vecs[N:]
    nm = vecs.shape[1]
    # ---- reconstruct the remaining components (all derivatives already /k0) ----
    # Hz = -j*Hzr/..., Ex = j*Exr, Ey = j*Eyr, Ez real  (common phase factor j dropped
    # from Ex,Ey; Hz and Ez are therefore in quadrature with the transverse fields)
    Hzr = (Dxf @ hx + Dyf @ hy) / neff                       # at (i+1/2, j+1/2)
    Exr = (neff * hy - Dyb @ Hzr) / exx[:, None]             # at (i+1/2, j)
    Eyr = -(neff * hx - Dxb @ Hzr) / eyy[:, None]            # at (i, j+1/2)
    Ez = (Dxb @ hy - Dyb @ hx) / ezz[:, None]                # at (i, j)
    sh = (Nx, Ny, nm)
    fields = dict(Hx=hx.reshape(sh), Hy=hy.reshape(sh), Hz=Hzr.reshape(sh),
                  Ex=Exr.reshape(sh), Ey=Eyr.reshape(sh), Ez=Ez.reshape(sh))
    return neff, fields, xn, yn

# ------------------------------------------------------------------ classification
def classify(Hx, Hy):
    """Return (pol, parity) for each mode. Grid: Hy points are symmetric about x=0;
    Hx points are shifted by dx/2 so mirror index is i -> Nx-i."""
    out = []
    for m in range(Hx.shape[2]):
        hx, hy = Hx[:, :, m], Hy[:, :, m]
        ex, ey = np.sum(hx**2), np.sum(hy**2)
        if ey >= ex:                                   # quasi-TE (Ex, Hy dominant)
            pol = "TE"; f = hy; mir = f[::-1, :]
            p = np.sum(f * mir) / np.sum(f * f)
        else:                                          # quasi-TM (Ey, Hx dominant)
            pol = "TM"; f = hx[1:, :]; mir = f[::-1, :]
            p = np.sum(f * mir) / np.sum(f * f)
        out.append((pol, p, ey / (ex + ey)))
    return out

def supermodes(w=0.45, h=0.22, gap=0.2, wl=1.55,
               n_core=3.476, n_sub=1.444, n_clad=1.444,
               pol="TE", dx=0.02, dy=0.02, Wx=4.0, below=1.2, above=1.2, nmodes=8):
    Nx = int(round(Wx / dx))
    Ny = int(round((below + h + above) / dy))
    x0 = -Nx / 2 * dx
    y0 = -below
    eps = make_eps_func(w, h, gap, n_core, n_sub, n_clad)
    neff, F, xn, yn = solve(eps, x0, y0, Nx, Ny, dx, dy, wl,
                            n_guess=n_core * 0.8, nmodes=nmodes)
    Hx, Hy = F["Hx"], F["Hy"]
    info = classify(Hx, Hy)
    even = odd = None
    for i, (p, par, teFrac) in enumerate(info):
        if p != pol: continue
        if par > 0.5 and even is None: even = (neff[i], i)
        if par < -0.5 and odd is None: odd = (neff[i], i)
    return dict(neff=neff, F=F, Hx=Hx, Hy=Hy, x=xn, y=yn, info=info, even=even, odd=odd,
                w=w, h=h, gap=gap, dx=dx, dy=dy, rects=None)

def coupling(n_even, n_odd, wl):
    kappa = np.pi * (n_even - n_odd) / wl      # = (beta_e - beta_o)/2   [1/um]
    Lc = np.pi / (2 * kappa)                   # full-transfer length    [um]
    return kappa, Lc

if __name__ == "__main__":
    wl = 1.55
    for d in (0.04, 0.02, 0.0125):                         # convergence check
        r = supermodes(wl=wl, gap=0.2, dx=d, dy=d, pol="TE")
        ne, no = r["even"][0], r["odd"][0]
        k, Lc = coupling(ne, no, wl)
        print(f"d={d*1e3:5.1f} nm | n_even={ne:.5f} n_odd={no:.5f} "
              f"dn={ne-no:.5f} | kappa={k:.5f}/um  Lc={Lc:.2f} um")


# Yee-cell offsets (in units of dx, dy) of each component
_OFF = dict(Ex=(.5, 0), Ey=(0, .5), Ez=(0, 0), Hx=(0, .5), Hy=(.5, 0), Hz=(.5, .5))

def _plot_set(r, comps, labels, suptitle, fname, xlim=(-1.5, 1.5), ylim=(-0.8, 1.0)):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 3, figsize=(14, 7))
    rects = r.get("rects") or [(-(r["gap"] + 2*r["w"])/2, -r["gap"]/2, 0, r["h"]),
                               (r["gap"]/2, (r["gap"] + 2*r["w"])/2, 0, r["h"])]
    names = r.get("labels", ("Even", "Odd"))
    for row, key in enumerate(("even", "odd")):
        ne, i = r[key]
        vmax = max(np.abs(r["F"][c][:, :, i]).max() for c in comps)   # common scale per mode
        for col, (c, lab) in enumerate(zip(comps, labels)):
            a = axs[row, col]
            ox, oy = _OFF[c]
            x = r["x"] + ox * r["dx"]; y = r["y"] + oy * r["dy"]
            a.pcolormesh(x, y, r["F"][c][:, :, i].T / vmax, cmap="RdBu_r",
                         vmin=-1, vmax=1, shading="auto")
            for (xl, xr, yb, yt) in rects:
                a.add_patch(plt.Rectangle((xl, yb), xr - xl, yt - yb, fill=False, ec="k", lw=1))
            a.set_xlim(*xlim); a.set_ylim(*ylim); a.set_aspect("equal")
            a.set_title(f"{names[row]} (n_eff={ne:.4f}): {lab}", fontsize=10)
            if row == 1: a.set_xlabel("x (um)")
            if col == 0: a.set_ylabel("y (um)")
    fig.suptitle(suptitle, fontsize=12)
    plt.tight_layout(h_pad=2.0); plt.savefig(fname, dpi=130); plt.close(fig)

def plot_E(r, fname="E_fields.png"):
    _plot_set(r, ("Ex", "Ey", "Ez"), ("Ex", "Ey", "Ez (90° out of phase)"),
              "Electric field components (common scale per mode)", fname)

def plot_B(r, fname="B_fields.png"):
    # B = mu0*H for non-magnetic materials, so B has the same spatial pattern as H
    _plot_set(r, ("Hx", "Hy", "Hz"), ("Bx = mu0 Hx", "By = mu0 Hy", "Bz = mu0 Hz (90° out of phase)"),
              "Magnetic flux density components, B = mu0 H (common scale per mode)", fname)

def plot_all(r, prefix=""):
    plot_E(r, prefix + "E_fields.png")
    plot_B(r, prefix + "B_fields.png")

def plot_supermodes(r, fname="supermodes.png"):          # kept for backward compatibility
    plot_all(r, prefix=fname.rsplit(".", 1)[0] + "_")


# =====================================================================================
#  ASYMMETRIC COUPLER: two different waveguides, phase mismatch, TE / TM
# =====================================================================================
def make_eps_rects(rects, n_core, n_sub, n_clad):
    """rects = [(x_left, x_right, y_bottom, y_top), ...]; substrate is y < 0."""
    def eps(X, Y):
        e = np.full(X.shape, n_clad**2)
        e[Y < 0] = n_sub**2
        for (xl, xr, yb, yt) in rects:
            e[(X >= xl) & (X <= xr) & (Y >= yb) & (Y <= yt)] = n_core**2
        return e
    return eps

def _pick_modes(neff, F, pol, count):
    """Indices (sorted by n_eff, high->low) of the first `count` modes of polarization pol."""
    ex = (F["Hx"]**2).sum(axis=(0, 1)); ey = (F["Hy"]**2).sum(axis=(0, 1))
    te = ey / (ex + ey)
    idx = [i for i in range(len(neff)) if (te[i] >= 0.5) == (pol.upper() == "TE")]
    if len(idx) < count:
        raise RuntimeError(f"Found only {len(idx)} {pol} modes; increase nmodes or the window.")
    return idx[:count]

def coupler(w1=0.45, w2=0.50, h1=0.22, h2=0.22, gap=0.20, wl=1.55,
            n_core=3.476, n_sub=1.444, n_clad=1.444, pol="TE",
            dx=0.02, dy=0.02, margin=1.5, below=1.2, above=1.2, nmodes=10):
    """Two (possibly different) strips on SiO2.  gap = edge-to-edge distance.
    Solves, on the SAME grid, (1) isolated WG1, (2) isolated WG2, (3) the coupled pair.
    Returns n_eff of each, plus coupled-mode-theory quantities."""
    rect1 = (-gap / 2 - w1, -gap / 2, 0.0, h1)
    rect2 = (gap / 2, gap / 2 + w2, 0.0, h2)
    xmin, xmax, hmax = rect1[0], rect2[1], max(h1, h2)
    Nx = int(round((xmax - xmin + 2 * margin) / dx))
    Ny = int(round((below + hmax + above) / dy))
    x0 = (xmin + xmax) / 2 - Nx / 2 * dx
    y0 = -below

    def run(rects):
        eps = make_eps_rects(rects, n_core, n_sub, n_clad)
        return solve(eps, x0, y0, Nx, Ny, dx, dy, wl, n_guess=n_core * 0.8, nmodes=nmodes)

    # isolated waveguides (fundamental mode of the requested polarization)
    n_iso = []
    for rc in (rect1, rect2):
        ne, F, _, _ = run([rc])
        n_iso.append(ne[_pick_modes(ne, F, pol, 1)[0]])
    n1, n2 = n_iso

    # coupled structure: the two highest-index modes of this polarization are the supermodes
    ne, F, xn, yn = run([rect1, rect2])
    iA, iB = _pick_modes(ne, F, pol, 2)
    nA, nB = ne[iA], ne[iB]

    k0 = 2 * np.pi / wl
    b1, b2, bA, bB = k0 * n1, k0 * n2, k0 * nA, k0 * nB
    delta = (b1 - b2) / 2                  # half phase mismatch  [1/um]
    Delta = (bA - bB) / 2                  # half supermode splitting = sqrt(kappa^2 + delta^2)
    kappa2 = Delta**2 - delta**2
    kappa = np.sqrt(max(kappa2, 0.0))
    Fmax = kappa2 / Delta**2 if Delta > 0 else 0.0   # max power-transfer efficiency
    Lc = np.pi / (2 * Delta)                          # length of first power maximum
    return dict(pol=pol, wl=wl, n1=n1, n2=n2, nA=nA, nB=nB,
                beta1=b1, beta2=b2, betaA=bA, betaB=bB,
                delta=delta, dbeta=2 * delta, kappa=kappa, Delta=Delta, Fmax=Fmax, Lc=Lc,
                kappa_warning=(kappa2 < 0),
                # for the field plotting functions
                F=F, x=xn, y=yn, dx=dx, dy=dy, w=w1, h=h1, gap=gap,
                even=(nA, iA), odd=(nB, iB), rects=[rect1, rect2],
                labels=("Supermode A (higher n_eff)", "Supermode B (lower n_eff)"))

def power_transfer(res, z):
    """Power in waveguide 2 when launching into waveguide 1:  P2 = F sin^2(Delta z)."""
    return res["Fmax"] * np.sin(res["Delta"] * np.asarray(z))**2

def report(res):
    print(f"--- {res['pol']} polarisation, wavelength {res['wl']} um ---")
    print(f"isolated WG1  n_eff = {res['n1']:.5f}")
    print(f"isolated WG2  n_eff = {res['n2']:.5f}")
    print(f"supermode A   n_eff = {res['nA']:.5f}")
    print(f"supermode B   n_eff = {res['nB']:.5f}")
    print(f"phase mismatch  d_beta = beta1-beta2 = {res['dbeta']:.5f} /um   (delta = {res['delta']:.5f})")
    print(f"coupling coefficient kappa = {res['kappa']:.5f} /um")
    print(f"max power transfer  = {100*res['Fmax']:.1f} %   at  Lc = {res['Lc']:.2f} um")
    if res["kappa_warning"]:
        print("WARNING: supermode splitting < mismatch (numerical); refine the grid.")

def plot_power(res, zmax=None, fname="power_transfer.png"):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    zmax = zmax or 3 * res["Lc"]
    z = np.linspace(0, zmax, 600)
    P2 = power_transfer(res, z)
    plt.figure(figsize=(6.5, 3.6))
    plt.plot(z, P2, label="WG2"); plt.plot(z, 1 - P2, label="WG1")
    plt.xlabel("z (um)"); plt.ylabel("Normalised power"); plt.ylim(0, 1.02)
    plt.title(f"{res['pol']}: dbeta={res['dbeta']:.4f}/um, kappa={res['kappa']:.4f}/um")
    plt.legend(); plt.grid(alpha=.3); plt.tight_layout(); plt.savefig(fname, dpi=130); plt.close()
