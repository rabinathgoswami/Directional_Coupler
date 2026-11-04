from dc_supermodes_fullvec import supermodes, coupling, plot_supermodes
from dc_supermodes_fullvec import plot_all


wl = 1.55  # wavelength in um

r = supermodes(
    w=0.45,        # core width (um)
    h=0.22,        # core height (um)
    gap=0.20,      # gap between the two cores (um)
    wl=wl,
    n_core=3.476,  # silicon
    n_sub=1.444,   # SiO2 substrate
    n_clad=1.444,  # cladding: 1.444 for SiO2, 1.0 for air
    pol="TE",      # "TE" or "TM"
    dx=0.02, dy=0.02,  # grid step (um); smaller = more accurate but slower
)

n_even, n_odd = r["even"][0], r["odd"][0]
kappa, Lc = coupling(n_even, n_odd, wl)

print("n_eff even  =", n_even)
print("n_eff odd   =", n_odd)
print("kappa (1/um)=", kappa)
print("Coupling length Lc (um) =", Lc)


plot_all(r)   # saves E_fields.png and B_fields.png

import numpy as np
for g in [0.15, 0.2, 0.25, 0.3, 0.4]:
    q = supermodes(gap=g, wl=wl, dx=0.02, dy=0.02)
    k, L = coupling(q["even"][0], q["odd"][0], wl)
    print(f"gap={g*1000:.0f} nm  Lc={L:.1f} um")