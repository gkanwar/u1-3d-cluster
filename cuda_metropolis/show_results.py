import analysis as al
import matplotlib.pyplot as plt
import numpy as np
import os
import paper_plt
paper_plt.load_latex_config()

fig, axes = plt.subplots(2,3, figsize=(6,4))
style = dict(marker='*', fillstyle='none', markersize=6)

all_e2s = np.arange(0.30, 1.25+1e-6, 0.05)
for L in [16, 32, 48, 64]:
    V = L**3

    e2s = []
    Es = []
    MCs = []
    MC2s = []
    MC2ps = []
    MCsus = []
    for e2 in all_e2s:
        fname = f'obs_trace_{e2:0.2f}_L{L}_cuda_E.dat'
        if not os.path.exists(fname):
            print(f'Skipping {fname} (does not exist)')
            continue
        E = np.fromfile(fname)
        MC = np.fromfile(f'obs_trace_{e2:0.2f}_L{L}_cuda_MC.dat')

        e2s.append(e2)
        Es.append(al.bootstrap(E / V, Nboot=1000, f=al.rmean))
        MCs.append(al.bootstrap(MC, Nboot=1000, f=al.rmean))
        MC2s.append(al.bootstrap(MC**2 / V, Nboot=1000, f=al.rmean))
        MC2ps.append(al.bootstrap(MC**2 / V**2, Nboot=1000, f=al.rmean))
        MCsus.append(al.bootstrap(MC, Nboot=1000, f=lambda MC: (al.rmean(MC**2) - al.rmean(np.abs(MC))**2)/V))

    e2s = np.array(e2s)
    Es = np.transpose(Es)
    MCs = np.transpose(MCs)
    MC2s = np.transpose(MC2s)
    MC2ps = np.transpose(MC2ps)
    MCsus = np.transpose(MCsus)

    al.add_errorbar(Es, xs=e2s, ax=axes[0,0], **style)
    al.add_errorbar(MCs, xs=e2s, ax=axes[0,1], **style)
    al.add_errorbar(MC2s, xs=e2s, ax=axes[1,0], **style)
    al.add_errorbar(MC2ps, xs=e2s, ax=axes[1,1], **style)
    al.add_errorbar(MCsus, xs=e2s, ax=axes[1,2], **style)

fig.set_tight_layout(True)

fig, ax = plt.subplots(1,1)
ax.plot(MC)
fig.set_tight_layout(True)

plt.show()
