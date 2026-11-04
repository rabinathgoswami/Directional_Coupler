from dc_supermodes_fullvec import coupler, report, plot_power, plot_all

for pol in ("TE", "TM"):
    r = coupler(w1=0.45, w2=0.50, h1=0.22, h2=0.22,
                gap=0.20, wl=1.55, pol=pol)
    report(r)
    plot_power(r, fname=f"power_transfer_{pol}.png")
    plot_all(r, f"{pol}_")      # saves TE_E_fields.png, TE_B_fields.png, TM_E_fields.png, ...