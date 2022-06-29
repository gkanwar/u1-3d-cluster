import argparse
import analysis as al
import matplotlib.pyplot as plt
import numpy as np
import paper_plt
paper_plt.load_latex_config()
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--e2', type=float, required=True)
parser.add_argument('--L', type=int, required=True)
args = parser.parse_args()

e2 = args.e2
L = args.L
V = L**3
binsize = 4

E = np.fromfile(f'obs_trace_{e2:0.2f}_L{L}_cluster_E.dat')
MC = np.fromfile(f'obs_trace_{e2:0.2f}_L{L}_cluster_MC.dat')

print('<E/V> =', al.bootstrap(E / V, Nboot=1000, f=al.rmean))
print('<MC> =', al.bootstrap(MC, Nboot=1000, f=al.rmean))
print('<MC^2/V> =', al.bootstrap(MC**2 / V, Nboot=1000, f=al.rmean))
print('<MC^2/V^2> =', al.bootstrap(MC**2 / V**2, Nboot=1000, f=al.rmean))

fig, axes = plt.subplots(2,1, figsize=(6,6))
style = dict(marker='*', fillstyle='none', markersize=6)
axes[0].plot(*al.bin_data(E, binsize=binsize))
axes[1].axhline(0.0, color='k', marker='')
axes[1].plot(*al.bin_data(MC, binsize=binsize))
axes[0].set_ylabel(r'$E$')
axes[1].set_ylabel(r'$M_C$')
fig.set_tight_layout(True)

binsizes = np.arange(20, 100, 20)
binned_MCs = []
binned_MC2s = []
for binsize in binsizes:
    binned_MCs.append(al.bootstrap(al.bin_data(MC, binsize=binsize)[1], Nboot=1000, f=al.rmean))
    binned_MC2s.append(al.bootstrap(al.bin_data(MC**2/V, binsize=binsize)[1], Nboot=1000, f=al.rmean))
binned_MCs = np.transpose(binned_MCs)
binned_MC2s = np.transpose(binned_MC2s)
fig, axes = plt.subplots(2,1, sharex=True)
al.add_errorbar(binned_MCs, xs=binsizes, ax=axes[0], marker='o', label=r'$\left< M_C \right>$')
al.add_errorbar(binned_MC2s, xs=binsizes, ax=axes[1], off=1, marker='o', label=r'$\left< M_C^2/V \right>$')
axes[1].set_xlabel('bin size')
fig.legend()
fig.set_tight_layout(True)

plt.show()
