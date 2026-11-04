"""
Length of a directional coupler needed to transfer a given fraction of the power
(default 30 %) into the second waveguide.  Full-vector Maxwell solver + coupled-mode theory.

How to use:  edit the PARAMETERS block, then run   python coupler_length.py
Needs dc_supermodes_fullvec.py in the same folder.   Units: micrometres.

Physics
  1. isolated WG1, isolated WG2 and the coupled pair are solved (full-vector, same grid)
        -> n1, n2 (phase mismatch d_beta = beta1 - beta2),  supermodes nA, nB
  2. delta = d_beta/2 ,  Delta = (betaA - betaB)/2 = sqrt(kappa^2 + delta^2)
        kappa = sqrt(Delta^2 - delta^2),   F = kappa^2/Delta^2   (max transfer)
  3. P2(z) = F sin^2(Delta z)   (light launched into WG1)
        first length with P2 = P_target :  L = asin( sqrt(P_target / F) ) / Delta
        (exists only if F >= P_target)
"""
import numpy as np
from dc_supermodes_fullvec import coupler

# =====================================================================================
#  >>>>>>>>>>>>>>>>>>>>>>>>>>>>   EDIT THESE PARAMETERS   <<<<<<<<<<<<<<<<<<<<<<<<<<<<
# =====================================================================================
PARAMS = dict(
    w1 = 0.45,        # um   width  of waveguide 1 (input waveguide)
    w2 = 0.45,        # um   width  of waveguide 2
    h1 = 0.22,        # um   height of waveguide 1
    h2 = 0.22,        # um   height of waveguide 2
    gap = 0.20,       # um   edge-to-edge gap between the waveguides
    wl = 1.55,        # um   wavelength
    n_core = 3.476,   # silicon
    n_sub  = 1.444,   # SiO2 substrate
    n_clad = 1.444,   # cladding (1.444 = SiO2, 1.0 = air)
    pol = "TE",       # "TE" or "TM"
    dx = 0.02, dy = 0.02,   # um   grid step (smaller = more accurate, much slower)
)
TARGET_POWER = 0.30   # fraction of the input power to be transferred to waveguide 2
# =====================================================================================

def length_for_power(target=TARGET_POWER, **params):
    """Return a dict with the coupler length for `target` power transfer and all inputs/outputs."""
    r = coupler(**params)
    F, Delta = r["Fmax"], r["Delta"]
    out = dict(params=params, target=target, n1=r["n1"], n2=r["n2"], nA=r["nA"], nB=r["nB"],
               dbeta=r["dbeta"], kappa=r["kappa"], Fmax=F, L_first_max=r["Lc"])
    if F >= target and Delta > 0:
        L = np.arcsin(np.sqrt(target / F)) / Delta
        out["length"] = L
        out["power_at_length"] = F * np.sin(Delta * L) ** 2
        out["reachable"] = True
    else:
        out["length"] = None
        out["power_at_length"] = None
        out["reachable"] = False
    return out

def print_report(res):
    p = res["params"]
    line = "=" * 62
    print(line); print("  DIRECTIONAL COUPLER - LENGTH FOR TARGET POWER TRANSFER"); print(line)
    print("INPUT PARAMETERS")
    print(f"  polarisation          : {p.get('pol', 'TE')}")
    print(f"  wavelength            : {p.get('wl', 1.55):g} um")
    print(f"  waveguide 1  (w x h)  : {p.get('w1')*1e3:.0f} nm x {p.get('h1', 0.22)*1e3:.0f} nm")
    print(f"  waveguide 2  (w x h)  : {p.get('w2')*1e3:.0f} nm x {p.get('h2', 0.22)*1e3:.0f} nm")
    print(f"  gap (edge to edge)    : {p.get('gap')*1e3:.0f} nm")
    print(f"  n core / sub / clad   : {p.get('n_core', 3.476)} / {p.get('n_sub', 1.444)} / {p.get('n_clad', 1.444)}")
    print(f"  grid step (dx, dy)    : {p.get('dx', 0.02)*1e3:.0f} nm, {p.get('dy', 0.02)*1e3:.0f} nm")
    print(f"  target power transfer : {res['target']*100:.1f} %")
    print("SOLVER RESULTS")
    print(f"  n_eff WG1 (isolated)  : {res['n1']:.5f}")
    print(f"  n_eff WG2 (isolated)  : {res['n2']:.5f}")
    print(f"  n_eff supermode A / B : {res['nA']:.5f} / {res['nB']:.5f}")
    print(f"  phase mismatch dbeta  : {res['dbeta']:.5f} 1/um   (beta1 - beta2)")
    print(f"  coupling coeff. kappa : {res['kappa']:.5f} 1/um")
    print(f"  max power transfer    : {res['Fmax']*100:.2f} %   (first maximum at z = {res['L_first_max']:.2f} um)")
    print("OUTPUT")
    if res["reachable"]:
        print(f"  coupler length for {res['target']*100:.0f}% transfer : {res['length']:.3f} um")
        print(f"  power transfer ratio at that length          : {res['power_at_length']*100:.2f} %  "
              f"(P2/P_in = {res['power_at_length']:.4f})")
        print(f"  power remaining in waveguide 1               : {(1-res['power_at_length'])*100:.2f} %")
    else:
        print(f"  TARGET NOT REACHABLE: maximum possible transfer is only {res['Fmax']*100:.2f} % "
              f"(< {res['target']*100:.0f} %).")
        print("  -> reduce the gap, reduce the width/height difference (phase mismatch), or lower the target.")
    print(line)

if __name__ == "__main__":
    result = length_for_power(TARGET_POWER, **PARAMS)
    print_report(result)
