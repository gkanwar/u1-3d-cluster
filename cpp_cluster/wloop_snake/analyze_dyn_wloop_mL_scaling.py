### Take W(r,t) data from dynamic wloop Snake runs with fixed mL=8 and
### compute windowed sigma estimates.

import analysis as al
import itertools
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import matplotlib.style as mstyle
import numpy as np
import os
import paper_plt
# paper_plt.load_latex_config()
paper_plt.load_basic_config()
import scipy as sp
import scipy.optimize
import tqdm

BIAS = 0.01
raw_data_prefix = 'raw_obs'
data_prefix = 'data'
out_data_prefix = 'refined_obs'
figs_prefix = 'figs'
os.makedirs(out_data_prefix, exist_ok=True)
os.makedirs(figs_prefix, exist_ok=True)

def measure_log_ratio(Wt_hist_cfgs):
    Wt_hist = al.rmean(Wt_hist_cfgs)
    return np.log(Wt_hist[0]) - np.log(Wt_hist[-1])

def load_data(L, e2, x, *, tag, tag2, binsize, bias=BIAS, n_therm=0):
    prefix = (
        f'{raw_data_prefix}_{tag}_bias{bias:.02f}_{tag2}/'
        f'dyn_wloop_{tag}_stag_L{L}_{e2:.2f}_x{x}' )
    print(f'{prefix=}')
    # load
    Wt_hist = np.fromfile(f'{prefix}_Wt_hist.dat')
    Wt_hist = Wt_hist.reshape(-1, L+1)
    Wt_hist *= np.exp(-bias*np.arange(L+1))
    MC_hist = np.fromfile(f'{prefix}_MC.dat')
    # trim
    Wt_hist = Wt_hist[n_therm:]
    MC_hist = MC_hist[n_therm:]
    # analyze
    xs_bin, Wt_hist_bin = al.bin_data(Wt_hist, binsize=binsize)
    Wt_est = al.bootstrap(Wt_hist_bin, Nboot=100, f=al.rmean)
    ratio = al.bootstrap(Wt_hist_bin, Nboot=1000, f=measure_log_ratio)
    ratio = (ratio[0]/L, ratio[1]/L)
    xs_bin_MC, MC_hist_bin = al.bin_data(MC_hist, binsize=binsize)
    jack_ratio = [
        measure_log_ratio(np.roll(Wt_hist_bin, -i, axis=0)[1:])/L
        for i in range(len(Wt_hist_bin))]
    return {
        'Wt': Wt_est,
        'ratio': ratio,
        # 'mcmc_trace_0': (xs_bin, Wt_hist_bin[:,0]),
        # 'mcmc_trace_m1': (xs_bin, Wt_hist_bin[:,-1]),
        'mcmc_trace_MC': (xs_bin_MC, MC_hist_bin),
        'jack_trace_ratio': (xs_bin, jack_ratio)
    }

n_therm_by_L = {
    192: 2000,
    256: 4000,
}
    
def load_windows(e2, windows, *, L, bias, ML, binsize):
    tag = 'cuda'
    tag2 = f'mL{ML}'
    traces = []
    fig, axes = plt.subplots(
        2,1, figsize=(6,3), tight_layout=True,
        gridspec_kw=dict(height_ratios=[0.1,0.9], hspace=0))
    leg_ax, ax = axes
    for window_f in tqdm.tqdm(windows):
        window = window_f(L)
        print(f'{window=}')
        sigmas = []
        for x in window:
            try:
                n_therm = n_therm_by_L.get(L, 0)
                res = load_data(
                    L=L, e2=e2, x=x, tag=tag, tag2=tag2, binsize=binsize,
                    bias=bias, n_therm=n_therm)
                sigmas.append(res['ratio'])
                ax.plot(*res['mcmc_trace_MC'], label=f'$x={x}$')
            except FileNotFoundError as e:
                print(f'Warning: Missing {L=} {e2=} {x=}')
        sigmas = np.transpose(sigmas)
        if len(sigmas) > 0:
            window_sigma = (
                np.mean(sigmas[0]),
                # errors averaged in quadrature
                np.sqrt(np.sum(sigmas[1]**2))/len(sigmas[1]))
            traces.append(window_sigma)
        else:
            traces.append((float('nan'), float('nan')))
    leg_ax.axis('off')
    leg_ax.legend(*ax.get_legend_handles_labels(), ncol=4, loc='upper center', bbox_to_anchor=(0.5, 1.0))
    fig.savefig(f'{figs_prefix}/mcmc_MC_L{L}_{e2:0.2f}.pdf')
    return np.array(traces)


def main():
    """
    Use window-avg sigma_eff to build sigma vs e^2 curves.
    """

    Ls = np.array([64, 96, 128, 192, 256])
    ML_e2s = np.array([
        (6, [1.35, 1.29, 1.23, 1.12, 1.09], 'd', 'xkcd:gold'),
        (8, [1.68, 1.50, 1.39, 1.24, 1.18], 'o', 'k'),
        (10, [1.92, 1.65, 1.49, 1.33, 1.25], 's', 'xkcd:teal')
    ], dtype=[('ML', int), ('e2s', float, (5,)), ('marker', 'U128'), ('color', 'U128')])
    binsize = 100
    windows = [
        lambda L: np.arange(5*L//8-3, 5*L//8+1) 
   ]
    window_labels = [
        '$w=5/8L$'
    ]
    fig, axes = plt.subplots(
        2,1, figsize=(4,4), tight_layout=True,
        sharex=True, gridspec_kw=dict(hspace=0))
    style = dict(
        fillstyle='none', markersize=3, markeredgewidth=0.5, capsize=1.5,
        linestyle='')
    for ML_row  in ML_e2s:
        ML = ML_row['ML']
        e2s = ML_row['e2s']
        marker = ML_row['marker']
        color = ML_row['color']
        print(f'{ML_row=}')
        full_style = dict(marker=marker, color=color, **style)
        traces_sigma = []
        traces_sigma_over_m2 = []
        for L, e2 in zip(Ls, e2s):
            trace = load_windows(e2, windows, L=L, bias=BIAS, ML=ML, binsize=binsize)
            traces_sigma.append(trace)
            # traces_sigma_over_m2.append(L**2 * trace / ML**2)
            traces_sigma_over_m2.append(L * trace / (ML*e2))
        traces_sigma = np.transpose(traces_sigma, axes=(1,2,0))
        traces_sigma_over_m2 = np.transpose(traces_sigma_over_m2, axes=(1,2,0))
        xs = e2s
        for i, (t_sigma, t_sigma_over_m2, window, wlabel) in enumerate(
                zip(traces_sigma, traces_sigma_over_m2, windows, window_labels)):
            off = i*0.001
            al.add_errorbar(
                t_sigma, xs=xs, ax=axes[0], off=off, **full_style,
                label=f'ML={ML} {wlabel}')
            al.add_errorbar(
                t_sigma_over_m2, xs=xs, ax=axes[1], off=off, **full_style,
                label=f'ML={ML} {wlabel}')

    for ax in axes:
        ax.legend()
        # ax.set_yscale('log')
    axes[-1].set_xlabel(r'$a e^2$')
    axes[0].set_ylabel(r'$a^2 \sigma$')
    axes[1].set_ylabel(r'$\sigma/m e^2$')
    # ax.set_ylim(0, 0.020)
    fig.savefig(f'{figs_prefix}/windowed_e2_vs_sigma_mLALL.pdf')

if __name__ == '__main__':
    os.makedirs(figs_prefix, exist_ok=True)
    os.makedirs(data_prefix, exist_ok=True)
    main()
