### Take V'(r) data from analyze_dyn_wloop and extract string tension estimates.

import analysis as al
import itertools
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import matplotlib.style as mstyle
import matplotlib.backends.backend_pdf as mpdf
import numpy as np
import os
import paper_plt
paper_plt.load_latex_config()
# paper_plt.load_basic_config()
import scipy as sp
import scipy.optimize
import tqdm

data_prefix = 'refined_obs'
figs_prefix = 'figs'
ale_data_prefix = '../../alessandro'

def load_trace(e2, L):
    fname = f'{data_prefix}/dVr_{e2:0.1f}_L{L}.txt'
    dtype = {
        'names': ('xs', 'ys', 'yerrs'),
        'formats': (int, np.float64, np.float64)
    }
    try:
        return np.loadtxt(fname, dtype=dtype)
    except OSError:
        return np.array([], dtype=dtype)

def plot_sigma_eff(fname):
    e2s = np.concatenate([
        [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4],
        [1.5, 2.0, 2.5, 3.0]
    ])
    Ls = np.array([64, 96, 128, 192])
    markers = ['o', 's', 'x', '*']
    colors = ['xkcd:royal blue', 'xkcd:emerald green', 'xkcd:red', 'k']
    global_style = dict(linestyle='', capsize=1.5, markersize=4, fillstyle='none')
    pdf = mpdf.PdfPages(fname)
    for e2 in e2s:
        fig, axes = plt.subplots(1,2, figsize=(6,4), gridspec_kw=dict(
            width_ratios=(0.7, 0.3), wspace=0
        ))
        ax, leg_ax = axes
        for L,marker,color in zip(Ls, markers, colors):
            style = dict(marker=marker, color=color, **global_style)
            try:
                trace = load_trace(e2, L)
            except OSError:
                continue
            al.add_errorbar(
                (trace['ys'], trace['yerrs']), xs=trace['xs'], ax=ax,
                **style, label=f'$e^2={e2}$, $L={L}$'
            )
        leg_info = ax.get_legend_handles_labels()
        leg_ax.legend(*leg_info, fontsize=6)
        leg_ax.axis('off')
        ax.set_xlabel(r'$r$')
        pdf.savefig(fig)
        plt.close(fig)
    pdf.close()
    return fig

def compute_windows(trace, windows):
    xs = trace['xs']
    assert np.all(xs[:-1] < xs[1:]), 'xs must be sorted'
    window_ests = []
    for window in windows:
        inds = []
        for x in window:
            ind = np.searchsorted(xs, x)
            if ind < len(xs) and xs[ind] == x:
                inds.append(ind)
        if len(inds) > 0:
            inds = np.array(inds)
            ys = trace['ys'][inds]
            yerrs = trace['yerrs'][inds]
            window_ests.append((
                np.mean(ys),
                np.sqrt(np.sum(yerrs**2))/len(yerrs)))
        else:
            window_ests.append((float('nan'), float('nan')))
    return window_ests

def ratio_with_errs(est_x, est_y):
    """
    Compute mean and std of x/y using Gaussian err prop
    """
    mean_x, err_x = est_x
    mean_y, err_y = est_y
    mean = mean_x / mean_y
    err = mean * np.sqrt((err_x / mean_x)**2 + (err_y / mean_y)**2)
    return (mean, err)


def plot_sigma_vs_e2():

    fig, ax = plt.subplots(1,1, figsize=(3.75,3.0), tight_layout=True)
    fig2, ax2 = plt.subplots(1,1, figsize=(3.75,3.0), tight_layout=True)

    # Alessandro's masses
    masses = np.load(
        f'{ale_data_prefix}/stag_mass/massdata.npy',
        allow_pickle=True).item()

    # Tej's window Snake data
    e2s = np.concatenate([
        [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4],
        [1.5, 2.0, 2.5, 3.0]
    ])
    windows = [
        np.arange(21, 24+1),
        np.arange(33, 36+1),
        np.arange(45, 48+1),
        np.arange(61, 64+1),
        np.arange(93, 96+1)
    ]
    # cmap = plt.get_cmap('tab20c', 20)
    window_styles = [
        dict(label=r'$w=[21,24]$', color='#f6e0b5'),
        dict(label=r'$w=[33,36]$', color='#eea990'),
        dict(label=r'$w=[45,48]$', color='#aa6f73'),
        dict(label=r'$w=[61,64]$', color='#a39193'),
        dict(label=r'$w=[93,96]$', color='#66545e')
    ]
    style = dict(
        linestyle='-', fillstyle='none', markersize=3,
        markeredgewidth=0.5, capsize=2)
    all_traces_sigma = []
    all_traces_sigma_over_m2 = []
    for L, marker in [
            (64, 'x'), (96, 's'), (128, 'o'), (192, '*')]:
        mass_e2s, mass_mean, mass_err = masses[L]
        traces_sigma = []
        traces_sigma_over_m2 = []
        e2s_sigma_over_m2 = []
        for e2 in e2s:
            trace = load_trace(e2, L)
            ests = compute_windows(trace, windows)
            traces_sigma.append(ests)
            ind = np.where(np.isclose(mass_e2s, e2))[0]
            if len(ind) == 0: continue
            ind = ind[0]
            mass_est = (mass_mean[ind]**2, 2*mass_mean[ind]*mass_err[ind])
            traces_sigma_over_m2.append([
                ratio_with_errs(est, mass_est)
                for est in ests
            ])
            e2s_sigma_over_m2.append(e2)
        traces_sigma = np.transpose(traces_sigma, axes=(1,2,0))
        traces_sigma_over_m2 = np.transpose(traces_sigma_over_m2, axes=(1,2,0))
        for i, (t_sigma, t_sigma_over_m2, w_style, window) in enumerate(
                zip(traces_sigma, traces_sigma_over_m2, window_styles, windows)):
            if not np.any(np.isfinite(t_sigma[0])): continue
            full_style = dict(style)
            full_style.update(w_style)
            full_style['label'] = rf'$L={L}$, {full_style["label"]}'
            full_style['marker'] = marker
            all_traces_sigma.append(
                (i, L, e2s, t_sigma, full_style))
            all_traces_sigma_over_m2.append(
                (i, L, e2s_sigma_over_m2, t_sigma_over_m2, full_style))
    all_traces_sigma.sort()
    all_traces_sigma_over_m2.sort()
    for i in range(len(all_traces_sigma)):
        off = i*0.001
        _, _, xs, trace, full_style = all_traces_sigma[i]
        al.add_errorbar(trace, xs=xs, ax=ax, off=off, **full_style)
        _, _, xs, trace, full_style = all_traces_sigma_over_m2[i]
        al.add_errorbar(trace, xs=xs, ax=ax2, off=off, **full_style)

    # Alessandro's <H> data
    e2s = np.loadtxt(f'{ale_data_prefix}/stag_string_tension/couplings.txt')
    ale_style = dict(
        color='xkcd:royal blue', linestyle='-', fillstyle='none', capsize=2, markersize=3,
        markeredgewidth=0.5, alpha=0.8)
    for L, marker in [(64, 'x'), (96, 's')]:
        ys = np.loadtxt(f'{ale_data_prefix}/stag_string_tension/sigmaval{L}.txt')
        yerrs = np.loadtxt(f'{ale_data_prefix}/stag_string_tension/sigmaerr{L}.txt')
        style = dict(marker=marker, **ale_style)
        al.add_errorbar(
            (ys, yerrs), xs=e2s, ax=ax, **style,
            label=rf'$L = {L}$, $\langle H \rangle$, fit $\sigma$')
        mass_e2s, mass_mean, mass_err = masses[L]
        # assert np.all(np.sum(np.isclose(mass_e2s, e2)) > 0 for e2 in e2s)
        ratio_ests = []
        ratio_e2s = []
        for e2,y,yerr in zip(e2s, ys, yerrs):
            ind = np.where(np.isclose(mass_e2s, e2))[0]
            if len(ind) == 0: continue
            if e2 < 1.05: continue
            ind = ind[0]
            mass_est = (mass_mean[ind]**2, 2*mass_mean[ind]*mass_err[ind])            
            ratio_e2s.append(e2)
            ratio_ests.append(ratio_with_errs((y,yerr), mass_est))
        ratio_ests = np.transpose(ratio_ests)
        al.add_errorbar(
            ratio_ests, xs=ratio_e2s, ax=ax2, **style,
            label=rf'$L = {L}$, $\langle H \rangle$, fit $\sigma$')


    ax.legend(ncol=2, fontsize=6)
    ax.set_xlabel(r'$e^2$')
    ax.set_ylabel(r'$a^2 \sigma$', rotation=0)
    ax.set_ylim(3e-4, 5e-2)
    ax.set_yscale('log')
    fig.savefig(f'{figs_prefix}/windowed_e2_vs_sigma.pdf')

    ax2.legend(ncol=2, fontsize=6)
    ax2.set_xlabel(r'$e^2$')
    ax2.set_ylabel(r'$\sigma / m^2$', rotation=0)
    ax2.set_ylim(0, 8.5)
    # ax2.set_ylim(3e-4, 5e-2)
    # ax2.set_yscale('log')
    fig2.savefig(f'{figs_prefix}/windowed_e2_vs_sigma_over_m2.pdf')

def main():
    # plot_sigma_eff(f'{figs_prefix}/combined_sigma_eff.pdf')
    plot_sigma_vs_e2()

if __name__ == '__main__':
    main()
