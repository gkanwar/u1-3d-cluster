### Using the M and MC observables sensitive to {C,T}- and T-breaking,
### respectively, we can observe the finite volume scaling of the pseudocritical
### points and extract a location of the 2nd order critical point.

import analysis as al
import itertools
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import scipy.interpolate
import scipy.optimize
import scipy.stats
import os
import paper_plt
paper_plt.load_latex_config()

figs_prefix = 'figs/PoS'
use_kde = False

def load_sweep(sweep, *, kind, M_bins, MC_bins):
    Ls, e2s = sweep
    all_traces = []
    all_hists = [
        { 'M_hist': [], 'MC_hist': [] } for _ in e2s
    ]
    do_bin = lambda x: al.bin_data(x, binsize=50)[1]
    assert len(M_bins) == len(e2s) and len(MC_bins) == len(e2s)
    for L in Ls:
        V = L**3
        trace_e2s = []
        trace_M = []
        trace_M2 = []
        trace_MC = []
        trace_MC2 = []
        trace_MCsus = []
        trace_E = []
        for i, (e2, M_bin, MC_bin) in enumerate(zip(e2s, M_bins, MC_bins)):
            if kind == 'npy':
                fname = f'raw_obs/obs_trace_{e2:0.2f}_L{L}_mixed.npy'
                if not os.path.exists(fname):
                    print(f'Skipping {fname} (does not exist)')
                    continue
                print(f'Loading {fname}...')
                d = np.load(fname, allow_pickle=True).item()
                if d['version'] < 5:
                    print(f'Skipping {fname} (old version)')
                    continue

            elif kind == 'cpp':
                prefix = f'cpp_cluster/raw_obs/obs_trace_{e2:0.2f}_L{L}_cluster'
                if not os.path.exists(prefix + '_E.dat'):
                    print(f'Skipping prefix {prefix} (does not exist)')
                    continue
                print(f'Loading prefix {prefix}...')
                d = {}
                d['E'] = np.fromfile(prefix + '_E.dat')
                if os.path.exists(prefix + '_M.dat'):
                    d['M'] = np.fromfile(prefix + '_M.dat')
                else:
                    d['M'] = np.zeros_like(d['E'])
                d['MC'] = np.fromfile(prefix + '_MC.dat')

            else:
                raise RuntimeError(f'unknown {kind=}')
                    
            print(f'... loaded timeseries with {len(d["M"])} pts')
            trace_e2s.append(e2)
            # order param traces
            M = al.bootstrap(do_bin(d['M']), Nboot=1000, f=al.rmean)
            M2 = al.bootstrap(do_bin(d['M']**2), Nboot=1000, f=al.rmean)
            MC = al.bootstrap(do_bin(d['MC']), Nboot=1000, f=al.rmean)
            MC2 = al.bootstrap(do_bin(d['MC']**2), Nboot=1000, f=al.rmean)
            MCsus = al.bootstrap(
                d['MC'], Nboot=1000,
                f=lambda MC: al.rmean(MC**2) - al.rmean(np.abs(MC))**2)
            E = al.bootstrap(do_bin(d['E']), Nboot=1000, f=al.rmean)
            trace_M.append(M)
            trace_M2.append(M2)
            trace_MC.append(MC)
            trace_MC2.append(MC2)
            trace_MCsus.append(MCsus)
            trace_E.append(E)
            # symmetrized hists
            if use_kde:
                M_data = np.concatenate((d['M'], -d['M'])) / V
                MC_data = np.concatenate((d['MC'], -d['MC'])) / V
                print(M_bin[-1] - M_bin[0])
                M_kde = sp.stats.gaussian_kde(M_data, (M_bin[-1] - M_bin[0]))
                MC_kde = sp.stats.gaussian_kde(MC_data, (MC_bin[-1] - MC_bin[0]))
                all_hists[i]['M_hist'].append((L, M_kde))
                all_hists[i]['MC_hist'].append((L, MC_kde))
            else:
                M_hist_estimator = lambda x: np.histogram(
                    np.concatenate((x, -x)) / V, bins=M_bin)[0] / (2*len(x))
                MC_hist_estimator = lambda x: np.histogram(
                    np.concatenate((x, -x)) / V, bins=MC_bin)[0] / (2*len(x))
                M_hist = al.bootstrap(d['M'], Nboot=100, f=M_hist_estimator)
                MC_hist = al.bootstrap(d['MC'], Nboot=100, f=MC_hist_estimator)
                all_hists[i]['M_hist'].append((L, M_hist))
                all_hists[i]['MC_hist'].append((L, MC_hist))
            
        trace_e2s = np.array(trace_e2s)
        trace_M = np.transpose(trace_M)
        trace_M2 = np.transpose(trace_M2)
        trace_MC = np.transpose(trace_MC)
        trace_MC2 = np.transpose(trace_MC2)
        trace_MCsus = np.transpose(trace_MCsus)
        trace_E = np.transpose(trace_E)
        all_traces.append({
            'e2': trace_e2s,
            'M': trace_M,
            'M2': trace_M2,
            'MC': trace_MC,
            'MC2': trace_MC2,
            'MCsus': trace_MCsus,
            'E': trace_E,
        })
    return all_traces, all_hists

def fit_mc2_curve(e2s, MC2_means, MC2_errs, *, err_rescale=1.0):
    print(e2s)
    print(MC2_means)
    print(MC2_errs)
    MC2_errs *= err_rescale
    # simple fit
    f = lambda e2, A, B: A*np.exp(-B/e2)
    popt, pcov = sp.optimize.curve_fit(
        f, e2s, MC2_means, sigma=MC2_errs, bounds=[[0.0, 0.0], [np.inf, np.inf]])
    f_opt = lambda e2: f(e2, *popt)
    resid = MC2_means - f_opt(e2s)
    chisq = np.sum(resid**2 / MC2_errs**2)
    chisq_per_dof = chisq / (len(e2s) - len(popt))
    print(f'fit params = {popt}')
    print(f'fit {chisq_per_dof=}')
    out = {
        'f': f_opt,
        'params': popt,
        'chisq': chisq,
        'chisq_per_dof': chisq_per_dof
    }
    # fit with arbitary e2c
    f = lambda e2, e2c, A, B: A*np.exp(-B/(e2-e2c))
    popt, pcov = sp.optimize.curve_fit(
        f, e2s, MC2_means, sigma=MC2_errs,
        bounds=[[-0.3, 0.0, 0.0], [0.3, np.inf, np.inf]])
    f_opt = lambda e2: f(e2, *popt)
    resid = MC2_means - f_opt(e2s)
    chisq = np.sum(resid**2 / MC2_errs**2)
    chisq_per_dof = chisq / (len(e2s) - len(popt))
    print(f'fit v2 params = {popt}')
    print(f'fit v2 errs = {np.sqrt(np.diag(pcov))}')
    print(f'fit v2 {chisq_per_dof=}')
    return out

# TODO: range cmap colors?
# colors = {
#     8: 'xkcd:pink',
#     16: 'xkcd:green',
#     32: 'xkcd:blue',
#     48: 'xkcd:gray',
#     64: 'k',
#     80: 'xkcd:red',
#     96: 'xkcd:purple',
#     128: 'xkcd:forest green',
#     192: 'xkcd:cyan',
#     256: 'xkcd:magenta'
# }
cmap = plt.get_cmap('cividis_r')
colors = {}
for L in [8, 16, 32, 48, 64, 80, 96, 128, 192, 256]:
    colors[L] = cmap(L / 256)
markers = {
    8: 'o',
    16: 's',
    32: '^',
    48: 'x',
    64: 'v',
    80: '*',
    96: '.',
    128: 'p',
    192: 'h',
    256: 'P'
}

def concat_dicts(ds):
    keys = None
    for d in ds:
        if keys is None:
            keys = list(d.keys())
        assert set(keys) == set(d.keys())
    return {
        k: sum((d[k] for d in ds), start=[])
        for k in keys
    }

def plot_results():
    sweep1 = (
        # np.array([8, 16, 32, 48, 64, 80, 96, 128], dtype=int),
        np.array([32, 48, 64, 80, 96, 128]),
        np.arange(0.30, 1.80+1e-6,  0.05),
    )
    sweep2 = (
        np.array([192, 256], dtype=int),
        np.concatenate((
            np.arange(0.30, 0.55+1e-6, 0.05),
            np.arange(0.70, 1.80+1e-6, 0.10)
        ))
    )
    all_e2 = sorted(list(set(sweep1[1]) | set(sweep2[1])))
    def make_MC_bins(e2):
        peak_loc_ansatz = max(1e-3, np.sqrt(6.0)*np.exp(-3.27 / e2))
        return np.linspace(-1.5*peak_loc_ansatz, 1.5*peak_loc_ansatz, endpoint=True, num=101)
    M_bins_by_e2 = {
        e2: np.linspace(-1.5e-3, 1.5e-3, endpoint=True, num=51)
        for e2 in all_e2
    }
    MC_bins_by_e2 = {
        e2: make_MC_bins(e2) for e2 in all_e2
    }
    MC_bins_sweep1 = [MC_bins_by_e2[e2] for e2 in sweep1[1]]
    MC_bins_sweep2 = [MC_bins_by_e2[e2] for e2 in sweep2[1]]
    M_bins_sweep1 = [M_bins_by_e2[e2] for e2 in sweep1[1]]
    M_bins_sweep2 = [M_bins_by_e2[e2] for e2 in sweep2[1]]
    traces1, hists1 = load_sweep(
        sweep1, kind='npy', M_bins=M_bins_sweep1, MC_bins=MC_bins_sweep1)
    traces2, hists2 = load_sweep(
        sweep2, kind='cpp', M_bins=M_bins_sweep2, MC_bins=MC_bins_sweep2)

    ### Order param histograms
    all_bins_hists = [
        (M_bins_by_e2[e2], MC_bins_by_e2[e2],
         concat_dicts(
             [hists1[i] for i in np.flatnonzero(sweep1[1] == e2)] +
             [hists2[i] for i in np.flatnonzero(sweep2[1] == e2)]))
        for e2 in all_e2
    ]
    figsize = (5.8, 2.25)
    gridspec_kw = dict(width_ratios=[0.3, 0.7], right=0.88, wspace=0.35)
    legend_params = dict(
        ncol=1, loc='center left', bbox_to_anchor=[0.91, 0.5])
    # columnspacing=1.0, handletextpad=0.2
    fiducial_e2 = 0.70
    fiducial_label_pos = (0.93, 0.95)
    fid_color = 'xkcd:bright red'
    fig_M, axes_M = plt.subplots(1, 2, figsize=figsize, gridspec_kw=gridspec_kw)
    fig_MC, axes_MC = plt.subplots(1, 2, figsize=figsize, gridspec_kw=gridspec_kw)
    legend_M = None
    legend_MC = None

    for e2,(M_bins,MC_bins,hist) in zip(all_e2, all_bins_hists):
        if e2 != fiducial_e2: continue # silly skip...
        M_hists = hist['M_hist']
        MC_hists = hist['MC_hist']

        ax = axes_M[0]
        max_y = None
        for L, M_hist in M_hists:
            if use_kde:
                xs = np.linspace(M_bins[0], M_bins[-1], endpoint=True, num=250)
                ys = M_hist(xs)
                ax.plot(xs, ys, color=colors[L], label=rf'$L={L}$')
            else:
                xs = (M_bins[1:] + M_bins[:-1])/2
                ys = M_hist
                al.add_errorbar(
                    ys, xs=xs, ax=ax, color=colors[L], marker='.', label=rf'$L={L}$')
            if max_y is None or np.max(ys) > max_y:
                max_y = np.max(ys)
        ax.set_xlim(np.min(M_bins), np.max(M_bins))
        ax.set_ylim(-0.1 * max_y, 1.5 * max_y)
        ax.set_xlabel(rf'$ O_{{CS}} / V$')
        ax.set_ylabel(r'Prob.~density')
        legend_M = ax.get_legend_handles_labels()
        ax.text(*fiducial_label_pos, rf'$e^2 = {e2:0.2f}$',
                transform=ax.transAxes, ha='right', va='top', color=fid_color)

        ax = axes_MC[0]
        max_y = None
        for L, MC_hist in MC_hists:
            print('plot L =', L, np.sum(MC_hist))
            if use_kde:
                xs = np.linspace(MC_bins[0], MC_bins[-1], endpoint=True, num=250)
                ys = MC_hist(xs)
                ax.plot(xs, ys, color=colors[L], label=rf'$L={L}$')
            else:
                xs = (MC_bins[1:] + MC_bins[:-1])/2
                ys = MC_hist
                al.add_errorbar(
                    ys, xs=xs, ax=ax, color=colors[L], marker='.', label=rf'$L={L}$')
            if max_y is None or np.max(ys) > max_y:
                max_y = np.max(ys)
        ax.set_xlim(np.min(MC_bins), np.max(MC_bins))
        ax.set_ylim(-0.1 * max_y, 1.5 * max_y)
        ax.set_xlabel(rf'$ O_S / V$')
        ax.set_ylabel(r'Prob.~density')
        ax.text(*fiducial_label_pos, rf'$e^2 = {e2:0.2f}$',
                transform=ax.transAxes, ha='right', va='top', color=fid_color)
        legend_MC = ax.get_legend_handles_labels()

    for i,(L,trace) in enumerate(itertools.chain(
            zip(sweep1[0], traces1), zip(sweep2[0], traces2))):
        color, marker = colors[L], markers[L]
        V = L**3
        off = i*0.00
        style = dict(
            off=off, color=color, marker=marker,
            fillstyle='none', markersize=6, label=rf'$L={L}$')

        # filter out missing data
        M_inds = np.nonzero(trace['M2'][0])[0]
        print(f'{M_inds=}')
        
        # al.add_errorbar(trace['M'], ax=axes[0,0], **style)
        # al.add_errorbar(trace['M2'] / V, ax=axes[1,0], **style)
        al.add_errorbar(
            trace['M2'][:,M_inds] / V**2, xs=trace['e2'][M_inds],
            ax=axes_M[1], **style)
        # al.add_errorbar(trace['MC'], ax=axes[0,1], **style)
        # al.add_errorbar(trace['MC2'] / V, ax=axes[1,1], **style)
        al.add_errorbar(
            trace['MC2'] / V**2, xs=trace['e2'],
            ax=axes_MC[1], **style)
        # al.add_errorbar(trace['MCsus'] / V, ax=axes[1,2], **style)
        # al.add_errorbar(trace['E'] / V, ax=axes[0,2], **style)

        # Fit L >= 128 as the "infinite volume" curves
        if L >= 128:
            ti, tf = 5, 20
            res = fit_mc2_curve(
                trace['e2'][ti:tf], *(trace['MC2'][:,ti:tf] / V**2), err_rescale=1.0)
            fit_e2s = np.linspace(np.min(trace['e2']), np.max(trace['e2']), num=100)
            fit_fs = res['f'](fit_e2s)
            A, B = res['params']
            if L == 128:
                fit_handle, = axes_MC[1].plot(
                    fit_e2s, fit_fs, linewidth=3, linestyle='--', color='b',
                    alpha=0.5, zorder=3)
                fit_label = rf'${A:0.2f} e^{{ -{B:0.2f} / e^2 }}$'

    axes_M[1].axvline(fiducial_e2, color=fid_color, linewidth=0.5*paper_plt.pix_to_pt)
    axes_MC[1].axvline(fiducial_e2, color=fid_color, linewidth=0.5*paper_plt.pix_to_pt)
    axes_M[1].set_xlabel(r'$e^2$')
    axes_MC[1].set_xlabel(r'$e^2$')
    axes_M[1].set_ylabel(r'$\left<O_{SC}^2\right> / V^2$')
    axes_MC[1].set_ylabel(r'$\left<O_S^2\right> / V^2$')
    axes_M[1].set_yscale('log')
    axes_MC[1].set_yscale('log')
    axes_MC[1].set_ylim(2e-7, 3e-1)
    axes_MC[1].legend([fit_handle], [fit_label], facecolor='0.95', edgecolor='0.95')

    fig_M.legend(*legend_M, **legend_params)
    fig_MC.legend(*legend_MC, **legend_params)
    fig_M.savefig(f'{figs_prefix}/O_CS_hist_{fiducial_e2:0.2f}_and_sweep.pdf')
    fig_MC.savefig(f'{figs_prefix}/O_S_hist_{fiducial_e2:0.2f}_and_sweep.pdf')

if __name__ == '__main__':
    plot_results()
