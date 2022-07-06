### Using the M and MC observables sensitive to {C,T}- and T-breaking,
### respectively, we can observe the finite volume scaling of the pseudocritical
### points and extract a location of the 2nd order critical point.

import analysis as al
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import scipy.optimize
import os
import paper_plt
paper_plt.load_latex_config()


def load_sweep(sweep, *, kind):
    Ls, e2s = sweep
    all_traces = []
    do_bin = lambda x: al.bin_data(x, binsize=20)[1]
    for L in Ls:
        V = L**3
        trace_e2s = []
        trace_M = []
        trace_M2 = []
        trace_MC = []
        trace_MC2 = []
        trace_MCsus = []
        trace_E = []
        for e2 in e2s:
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
                prefix = f'cpp_cluster/obs_trace_{e2:0.2f}_L{L}_cluster'
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
            'E': trace_E
        })
    return all_traces

def fit_mc2_curve(e2s, MC2_means, MC2_errs, *, err_rescale=1.0):
    print(e2s)
    print(MC2_means)
    print(MC2_errs)
    MC2_errs *= err_rescale
    f = lambda e2, A, B: A*np.exp(-B/e2)
    popt, pcov = sp.optimize.curve_fit(
        f, e2s, MC2_means, sigma=MC2_errs, bounds=[[0.0, 0.0], [np.inf, np.inf]])
    f_opt = lambda e2: f(e2, *popt)
    resid = MC2_means - f_opt(e2s)
    chisq = np.sum(resid**2 / MC2_errs**2)
    chisq_per_dof = chisq / (len(e2s) - len(popt))
    print(f'fit params = {popt}')
    print(f'fit {chisq_per_dof=}')
    return {
        'f': f_opt,
        'params': popt,
        'chisq': chisq,
        'chisq_per_dof': chisq_per_dof
    }

def plot_results():
    sweep1 = (
        # np.array([8, 16, 32, 48, 64, 80, 96, 128], dtype=int),
        np.array([128]),
        np.arange(0.30, 1.80+1e-6,  0.05),
    )
    sweep2 = (
        np.array([192, 256], dtype=int),
        np.arange(0.30, 0.55+1e-6, 0.05)
    )
    traces1 = load_sweep(sweep1, kind='npy')
    traces2 = load_sweep(sweep2, kind='cpp')

    colors = {
        8: 'xkcd:pink',
        16: 'xkcd:green',
        32: 'xkcd:blue',
        48: 'xkcd:gray',
        64: 'k',
        80: 'xkcd:red',
        96: 'xkcd:purple',
        128: 'xkcd:forest green',
        192: 'xkcd:cyan',
        256: 'xkcd:magenta'
    }
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
    
    fig, axes = plt.subplots(3,3, figsize=(10,6))
    for i,(L,trace) in enumerate(zip(sweep1[0], traces1)):
        color, marker = colors[L], markers[L]
        V = L**3
        off = i*0.00
        style = dict(xs=trace['e2'], off=off, color=color, marker=marker, fillstyle='none', markersize=6, label=rf'$L={L}$')
        al.add_errorbar(trace['M'], ax=axes[0,0], **style)
        al.add_errorbar(trace['M2'] / V, ax=axes[1,0], **style)
        al.add_errorbar(trace['M2'] / V**2, ax=axes[2,0], **style)
        al.add_errorbar(trace['MC'], ax=axes[0,1], **style)
        al.add_errorbar(trace['MC2'] / V, ax=axes[1,1], **style)
        al.add_errorbar(trace['MC2'] / V**2, ax=axes[2,1], **style)
        al.add_errorbar(trace['MCsus'] / V, ax=axes[1,2], **style)
        al.add_errorbar(trace['E'] / V, ax=axes[0,2], **style)

        if L == 128:
            ti, tf = 4, 20
            res = fit_mc2_curve(
                trace['e2'][ti:tf], *(trace['MC2'][:,ti:tf] / V**2), err_rescale=5.0)
            fit_e2s = np.linspace(np.min(trace['e2']), np.max(trace['e2']), num=100)
            fit_fs = res['f'](fit_e2s)
            A, B = res['params']
            axes[2,1].plot(fit_e2s, fit_fs, linewidth=3, linestyle='--', color='b', alpha=0.5, zorder=3,
                           label=rf'${A:0.2f} e^{{ -{B:0.2f} / e^2 }}$')

    for i,(L,trace) in enumerate(zip(sweep2[0], traces2)):
        color, marker = colors[L], markers[L]
        V = L**3
        off = i*0.00
        style = dict(xs=trace['e2'], off=off, color=color, marker=marker, fillstyle='none', markersize=6, label=rf'$L={L}$')
        al.add_errorbar(trace['M'], ax=axes[0,0], **style)
        al.add_errorbar(trace['M2'] / V, ax=axes[1,0], **style)
        al.add_errorbar(trace['M2'] / V**2, ax=axes[2,0], **style)
        al.add_errorbar(trace['MC'], ax=axes[0,1], **style)
        al.add_errorbar(trace['MC2'] / V, ax=axes[1,1], **style)
        al.add_errorbar(trace['MC2'] / V**2, ax=axes[2,1], **style)
        al.add_errorbar(trace['MCsus'] / V, ax=axes[1,2], **style)
        al.add_errorbar(trace['E'] / V, ax=axes[0,2], **style)

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
    axes[2,1].set_ylabel(r'$\left<M_C^2\right> / V^2$')
    axes[2,0].set_yscale('log')
    axes[2,1].set_yscale('log')
    axes[1,2].set_ylabel(r'$\frac{1}{V}(\left<M_C^2\right> - \left<|M_C|\right>^2)$')

    handles, labels = axes[2,1].get_legend_handles_labels()
    axes[2,2].legend(handles, labels)

    fig.set_tight_layout(True)
    fig.savefig('figs/phase_transition_sweep1.pdf')

if __name__ == '__main__':
    plot_results()
