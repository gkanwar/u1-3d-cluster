import analysis as al
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import scipy.optimize

def load_clust_hist(e2, L):
    V = L**3
    clust_hist = np.fromfile(
        f'clust_hist/obs_trace_{e2:0.2f}_L{L}_cluster_clust_hist.dat',
        dtype=np.int32
    ).reshape(V)
    return clust_hist

def power_law_fit(xs, ys):
    f = lambda xs, A, p: A * xs**(-p)
    popt, pcov = sp.optimize.curve_fit(f, xs, ys)
    fopt = lambda xs: f(xs, *popt)
    resid = fopt(xs) - ys
    chisq = np.sum(resid**2)
    chisq_per_dof = chisq / (len(ys) - len(popt))
    print(f'{chisq_per_dof=}')
    return {
        'params': popt,
        'f': fopt,
    }

def run_clust_check(L, axes):
    e2s = np.arange(0.30, 2.00+1e-6, 0.10)
    colors = plt.get_cmap('RdBu')(np.arange(len(e2s)) / len(e2s))
    binsize = 128
    for e2,color in zip(e2s, colors):
        clust_hist = load_clust_hist(e2, L)
        xs, clust_hist = al.bin_data(clust_hist, binsize=binsize)
        xs = (xs + binsize/2) / L**3
        clust_hist *= binsize # raw count
        axes[0].plot(xs[1:], clust_hist[1:], color=color) # truncate smallest bin
        axes[1].plot(xs, clust_hist, color=color, label=f'e2={e2:0.2f}')
        # fit small cluster size dist
        fit_xs, fit_ys = xs[1:7], clust_hist[1:7]
        res = power_law_fit(fit_xs, fit_ys)
        print(fit_xs, fit_ys, res['params'])
        axes[0].plot(
            fit_xs, res['f'](fit_xs), color='k', linewidth=0.5, linestyle='--',
            label=f'{res["params"][0]:.2f} x^(-{res["params"][1]:.2f})')
    for ax in axes:
        ax.set_yscale('log')
    axes[0].text(0.9, 0.95, f'L={L}', transform=axes[0].transAxes, ha='right', va='top', fontsize=16)
    axes[0].set_xscale('log')
    axes[0].legend(loc='lower left', fontsize=6, ncol=2)
    axes[1].legend(ncol=4)
    axes[0].set_xlim(binsize / L**3, 10*binsize / L**3)
    axes[0].set_ylim(1e-3 * L**3, 10 * L**3)
    axes[1].set_xlim(-0.01, 1.01)
    axes[0].set_xticklabels([], minor=True)

def main():
    fig, axes = plt.subplots(3,2, figsize=(9,12), gridspec_kw=dict(width_ratios=[0.3, 0.7]))
    run_clust_check(64, axes[0])
    run_clust_check(96, axes[1])
    run_clust_check(128, axes[2])
    axes[2,1].set_xlabel('cluster size / V')
    for ax in axes[:,0]:
        ax.set_ylabel('count')
    fig.set_tight_layout(True)
    fig.savefig(f'figs/tmp_clust_all.pdf')

if __name__ == '__main__': main()
