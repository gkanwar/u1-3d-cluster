import analysis as al
import matplotlib.pyplot as plt
import numpy as np
import paper_plt
paper_plt.load_latex_config()

plt.rcParams['text.usetex'] = False

e2s = np.array([0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60])
L = 80
V = L**3

Es = []
MCs = []
MC2s = []
MC2ps = []
for e2 in e2s:
    E = np.fromfile(f'obs_trace_{e2:0.2f}_L{L}_cuda_E.dat')
    MC = np.fromfile(f'obs_trace_{e2:0.2f}_L{L}_cuda_MC.dat')

    Es.append(al.bootstrap(E / V, Nboot=1000, f=al.rmean))
    MCs.append(al.bootstrap(MC, Nboot=1000, f=al.rmean))
    MC2s.append(al.bootstrap(MC**2 / V, Nboot=1000, f=al.rmean))
    MC2ps.append(al.bootstrap(MC**2 / V**2, Nboot=1000, f=al.rmean))

Es = np.transpose(Es)
MCs = np.transpose(MCs)
MC2s = np.transpose(MC2s)
MC2ps = np.transpose(MC2ps)

fig, axes = plt.subplots(2,2, figsize=(6,6))
style = dict(marker='*', fillstyle='none', markersize=6)
al.add_errorbar(Es, xs=e2s, ax=axes[0,0], **style)
al.add_errorbar(MCs, xs=e2s, ax=axes[0,1], **style)
al.add_errorbar(MC2s, xs=e2s, ax=axes[1,0], **style)
al.add_errorbar(MC2ps, xs=e2s, ax=axes[1,1], **style)
fig.set_tight_layout(True)

fig, ax = plt.subplots(1,1)
ax.plot(MC)
fig.set_tight_layout(True)

plt.show()
