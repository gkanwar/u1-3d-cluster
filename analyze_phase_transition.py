### Using the M and MC observables sensitive to {C,T}- and T-breaking,
### respectively, we can observe the finite volume scaling of the pseudocritical
### points and extract a location of the 2nd order critical point.

import analysis as al
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import paper_plt
paper_plt.load_latex_config()


def load_sweep(sweep):
    Ls, e2s = sweep
    all_traces = []
    do_bin = lambda x: al.bin_data(x, binsize=100)[1]
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
            fname = f'raw_obs/obs_trace_{e2:0.2f}_L{L}_mixed.npy'
            if not os.path.exists(fname):
                print(f'Skipping {fname} (does not exist)')
                continue
            print(f'Loading {fname}...')
            d = np.load(fname, allow_pickle=True).item()
            if d['version'] < 5:
                print(f'Skipping {fname} (old version)')
                continue
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

def plot_results():
    sweep1 = (
        np.array([8, 16, 32, 48, 64, 80], dtype=int),
        np.arange(0.30, 1.80+1e-6,  0.05),
    )
    colors1 = ['xkcd:pink', 'xkcd:green', 'xkcd:blue', 'xkcd:gray', 'k', 'xkcd:red']
    markers1 = ['o', 's', '^', 'x', 'v', '*']
    
    traces1 = load_sweep(sweep1)
    fig, axes = plt.subplots(3,3, figsize=(10,6))
    for i,(L,trace,color,marker) in enumerate(zip(sweep1[0], traces1, colors1, markers1)):
        V = L**3
        off = i*0.003
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

    handles, labels = axes[0,0].get_legend_handles_labels()
    axes[2,2].legend(handles, labels)

    fig.set_tight_layout(True)
    fig.savefig('figs/phase_transition_sweep1.pdf')

if __name__ == '__main__':
    plot_results()
