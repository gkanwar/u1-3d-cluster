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
import os
import paper_plt
paper_plt.load_latex_config()


def load_sweep(sweep, *, kind, M_bins, MC_bins):
    Ls, e2s = sweep
    all_traces = []
    all_hists = [
        { 'M_hist': [], 'MC_hist': [] } for _ in e2s
    ]
    do_bin = lambda x: al.bin_data(x, binsize=20)[1]
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
            M_hist_estimator = lambda x: np.histogram(
                np.concatenate((x, -x)) / V, bins=M_bin)[0] / (2*len(x))
            MC_hist_estimator = lambda x: np.histogram(
                np.concatenate((x, -x)) / V, bins=MC_bin)[0] / (2*len(x))
            M_hist = al.bootstrap(d['M'], Nboot=100, f=M_hist_estimator)
            MC_hist = al.bootstrap(d['MC'], Nboot=100, f=MC_hist_estimator)
            # M_hist = np.histogram(
            #     np.concatenate((d['M'], -d['M'])) / V, bins=M_bin)[0] / (2*len(d['M']))
            # MC_hist = np.histogram(
            #     np.concatenate((d['MC'], -d['MC'])) / V, bins=MC_bin)[0] / (2*len(d['MC']))
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

def fit_mc2_curve(e2s, MC2_means, MC2_errs, *, fix_ec, err_rescale=1.0):
    print(e2s)
    print(MC2_means)
    print(MC2_errs)
    MC2_errs *= err_rescale
    g = lambda e2, A, B, e2c: A*np.exp(-B/(e2-e2c))
    if fix_ec:
        f = lambda e2, A, B: g(e2, A, B, 0.0)
        bounds = [[0.0]*2, [np.inf]*2]
        p0 = (1.0, 1.0)
    else:
        f = g
        bounds = [[0.0]*3, [np.inf]*3]
        p0 = (1.0, 1.0, 0.0)
    popt, pcov = sp.optimize.curve_fit(
        f, e2s, MC2_means, sigma=MC2_errs, bounds=bounds, p0=p0)
    f_opt = lambda e2: f(e2, *popt)
    resid = MC2_means - f_opt(e2s)
    chisq = np.sum(resid**2 / MC2_errs**2)
    chisq_per_dof = chisq / (len(e2s) - len(popt))
    print(f'fit params = {popt} ({np.sqrt(np.diag(pcov))})')
    print(f'fit {chisq_per_dof=}')
    return {
        'f': f_opt,
        'params': popt,
        'chisq': chisq,
        'chisq_per_dof': chisq_per_dof
    }

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
for L in [8, 16, 32, 48, 64, 80, 96, 128, 192, 256, 384]:
    colors[L] = cmap(L / 512)
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
    256: 'P',
    384: '>'
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
        np.array([128]), # 32, 48, 64, 80, 96
        np.arange(0.30, 1.80+1e-6,  0.05),
    )
    sweep2 = (
        np.array([192, 256], dtype=int),
        np.concatenate((
            np.arange(0.30, 0.55+1e-6, 0.05),
            np.arange(0.70, 1.80+1e-6, 0.10)
        ))
    )
    sweep3 = (
        np.array([384], dtype=int),
        np.arange(0.30, 0.55+1e-6, 0.05)
    )
    all_e2 = sorted(list(set(sweep1[1]) | set(sweep2[1]) | set(sweep3[1])))
    def make_MC_bins(e2):
        peak_loc_ansatz = max(1e-3, np.sqrt(6.0)*np.exp(-3.27 / e2))
        return np.linspace(-1.5*peak_loc_ansatz, 1.5*peak_loc_ansatz, endpoint=True, num=51)
    M_bins_by_e2 = {
        e2: np.linspace(-1.5e-3, 1.5e-3, endpoint=True, num=51)
        for e2 in all_e2
    }
    MC_bins_by_e2 = {
        e2: make_MC_bins(e2) for e2 in all_e2
    }
    MC_bins_sweep1 = [MC_bins_by_e2[e2] for e2 in sweep1[1]]
    MC_bins_sweep2 = [MC_bins_by_e2[e2] for e2 in sweep2[1]]
    MC_bins_sweep3 = [MC_bins_by_e2[e2] for e2 in sweep3[1]]
    M_bins_sweep1 = [M_bins_by_e2[e2] for e2 in sweep1[1]]
    M_bins_sweep2 = [M_bins_by_e2[e2] for e2 in sweep2[1]]
    M_bins_sweep3 = [M_bins_by_e2[e2] for e2 in sweep3[1]]
    traces1, hists1 = load_sweep(
        sweep1, kind='npy', M_bins=M_bins_sweep1, MC_bins=MC_bins_sweep1)
    traces2, hists2 = load_sweep(
        sweep2, kind='cpp', M_bins=M_bins_sweep2, MC_bins=MC_bins_sweep2)
    traces3, hists3 = load_sweep(
        sweep3, kind='cpp', M_bins=M_bins_sweep3, MC_bins=MC_bins_sweep3)

    ### Order param histograms
    all_bins_hists = [
        (M_bins_by_e2[e2], MC_bins_by_e2[e2],
         concat_dicts(
             [hists1[i] for i in np.flatnonzero(sweep1[1] == e2)] +
             [hists2[i] for i in np.flatnonzero(sweep2[1] == e2)] +
             [hists3[i] for i in np.flatnonzero(sweep3[1] == e2)]))
        for e2 in all_e2
    ]
    for e2,(M_bins,MC_bins,hist) in zip(all_e2, all_bins_hists):
        M_hists = hist['M_hist']
        MC_hists = hist['MC_hist']
        fig, ax = plt.subplots(1,1, figsize=(3.75, 3.0))
        print('bins', M_bins)
        # print('hist', M_hists[0][1].shape)
        print('Ls', [M_hist[0] for M_hist in M_hists])
        for L, M_hist in M_hists:
            xs = (M_bins[1:] + M_bins[:-1])/2
            ys = M_hist
            # ax.step(M_bins[1:], M_hist, color=colors[L], label=rf'$L={L}$')
            # ax.plot(xs, ys, color=colors[L], marker='.', label=rf'$L={L}$')
            al.add_errorbar(ys, xs=xs, ax=ax, color=colors[L], marker='.', label=rf'$L={L}$')
        ax.set_xlim(np.min(M_bins), np.max(M_bins))
        ax.set_xlabel(r'$\left< M \right> / V$')
        ax.legend()
        fig.savefig(f'figs/M_hist_{e2:0.2f}.pdf')
        plt.close(fig)
        fig, ax = plt.subplots(1,1, figsize=(3.75, 3.0))
        for L, MC_hist in MC_hists:
            print('plot L =', L, np.sum(MC_hist))
            xs = (MC_bins[1:] + MC_bins[:-1])/2
            ys = MC_hist
            # ax.step(MC_bins[1:], MC_hist, color=colors[L], label=rf'$L={L}$')
            # ax.bar(MC_bins[:-1], MC_hist, width=MC_bins[1]-MC_bins[0],
            #        alpha=0.3, color=colors[L], label=rf'$L={L}$')
            # ax.plot(xs, ys, color=colors[L], marker='.', label=rf'$L={L}$')
            al.add_errorbar(ys, xs=xs, ax=ax, color=colors[L], marker='.', label=rf'$L={L}$')
        ax.set_xlim(np.min(MC_bins), np.max(MC_bins))
        ax.set_xlabel(r'$\left< M_C \right> / V$')
        ax.legend()
        fig.savefig(f'figs/MC_hist_{e2:0.2f}.pdf')
        plt.close(fig)

    ### Joint order param sweep plots
    fig, axes = plt.subplots(3,3, figsize=(10,6))
    for i,(L,trace) in enumerate(itertools.chain(
            zip(sweep1[0], traces1), zip(sweep2[0], traces2), zip(sweep3[0], traces3))):
        color, marker = colors[L], markers[L]
        V = L**3
        off = i*0.00
        style = dict(xs=trace['e2'], off=off, color=color, marker=marker, fillstyle='none', markersize=6, label=rf'$L={L}$')
        al.add_errorbar(trace['M'], ax=axes[0,0], **style)
        al.add_errorbar(trace['M2'] / V, ax=axes[1,0], **style)
        al.add_errorbar(trace['M2'] / V**2, ax=axes[2,0], **style)
        al.add_errorbar(trace['MC'], ax=axes[0,1], **style)
        al.add_errorbar(trace['MC2'] / V, ax=axes[1,1], **style)
        ### FORNOW:
        al.add_errorbar((trace['MC2'] / V**2)**trace['e2'], ax=axes[2,1], **style)
        al.add_errorbar(trace['MCsus'] / V, ax=axes[1,2], **style)
        al.add_errorbar(trace['E'] / V, ax=axes[0,2], **style)

        # Fit L=256 as the "infinite volume" curve
        if L == 256:
            e2i = 0.50
            e2f = 0.80
            inds = np.nonzero((e2i - 1e-6 <= trace['e2']) & (trace['e2'] <= e2f + 1e-6))[0]
            print(f'{inds=}')
            # ti, tf = np.argmax(trace['e2'] >= e2i), np.argmin(trace['e2'] <= e2f)-1
            # print(f'{ti=} {tf=}')
            res = fit_mc2_curve(
                trace['e2'][inds], *(trace['MC2'][:,inds] / V**2), err_rescale=5.0, fix_ec=True)
            fit_e2s = np.linspace(np.min(trace['e2']), np.max(trace['e2']), num=100)
            fit_fs = res['f'](fit_e2s)
            A, B = res['params']
            ### FORNOW:
            fit_fs = fit_fs ** fit_e2s
            axes[2,1].plot(
                fit_e2s, fit_fs, linewidth=0.5, linestyle='--', color='b', alpha=0.5, zorder=3,
                label=rf'${A:0.2f} e^{{ -{B:0.2f} / e^2 }}$')
            res = fit_mc2_curve(
                trace['e2'][inds], *(trace['MC2'][:,inds] / V**2), err_rescale=5.0, fix_ec=False)
            fit_e2s = np.linspace(np.min(trace['e2']), np.max(trace['e2']), num=100)
            fit_fs = res['f'](fit_e2s)
            ### FORNOW:
            fit_fs = fit_fs ** fit_e2s
            A, B, e2c = res['params']
            axes[2,1].plot(
                fit_e2s, fit_fs, linewidth=0.5, linestyle='--', color='r', alpha=0.5, zorder=3,
                label=rf'${A:0.2f} e^{{ -{B:0.2f} / (e^2 - {e2c:0.3f}) }}$')

    # for i,(L,trace) in enumerate(zip(sweep2[0], traces2)):
    #     color, marker = colors[L], markers[L]
    #     V = L**3
    #     off = i*0.00
    #     style = dict(xs=trace['e2'], off=off, color=color, marker=marker, fillstyle='none', markersize=6, label=rf'$L={L}$')
    #     al.add_errorbar(trace['M'], ax=axes[0,0], **style)
    #     al.add_errorbar(trace['M2'] / V, ax=axes[1,0], **style)
    #     al.add_errorbar(trace['M2'] / V**2, ax=axes[2,0], **style)
    #     al.add_errorbar(trace['MC'], ax=axes[0,1], **style)
    #     al.add_errorbar(trace['MC2'] / V, ax=axes[1,1], **style)
    #     al.add_errorbar(trace['MC2'] / V**2, ax=axes[2,1], **style)
    #     al.add_errorbar(trace['MCsus'] / V, ax=axes[1,2], **style)
    #     al.add_errorbar(trace['E'] / V, ax=axes[0,2], **style)

    for ax in axes[2]:
        ax.set_xlabel(r'$e^2$')
    axes[0,0].set_ylabel(r'$\left<M\right>$')
    axes[0,1].set_ylabel(r'$\left<M_C\right>$')
    axes[0,2].set_ylabel(r'$\left<E / V\right>$')
    axes[0,2].set_yscale('log')
    axes[1,0].set_ylabel(r'$\left<M^2\right> / V$')
    axes[1,1].set_ylabel(r'$\left<M_C^2\right> / V$')
    axes[1,0].set_yscale('log')
    axes[1,1].set_yscale('log')
    axes[2,0].set_ylabel(r'$\left<M^2\right> / V^2$')
    axes[2,1].set_ylabel(r'$(\left<M_C^2\right> / V^2)^{e^2}$')
    axes[2,0].set_yscale('log')
    axes[2,1].set_yscale('log')
    axes[2,1].set_ylim(3e-4, 2e-1)
    axes[1,2].set_ylabel(r'$\frac{1}{V}(\left<M_C^2\right> - \left<|M_C|\right>^2)$')

    handles, labels = axes[2,1].get_legend_handles_labels()
    axes[2,2].legend(handles, labels)

    fig.set_tight_layout(True)
    fig.savefig('figs/phase_transition_sweep2.pdf')
    plt.close(fig)

if __name__ == '__main__':
    plot_results()
