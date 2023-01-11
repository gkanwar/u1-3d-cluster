import analysis as al
import matplotlib.pyplot as plt
import numpy as np
import os
import paper_plt
paper_plt.load_latex_config()
import scipy as sp
import scipy.optimize

def est_fwd_sigma(Pfs):
    return -np.log(al.rmean(Pfs))
def est_bwd_sigma(Pbs):
    return np.log(al.rmean(Pbs))
def est_avg_sigma(Pfs, Pbs):
    return (np.log(al.rmean(Pbs)) - np.log(al.rmean(Pfs))) / 2


### VERSION 1: Based on static Wilson loop sims with various x,t geoms
def compute_sigma(e2, *, x, L):
    ts = [3,4,5]
    prefix = 'raw_obs/raw_obs_stag'
    Pfs = []
    Pbs = []
    MCs = []
    for t in ts:
        fname = f'{prefix}_L{L}_{e2:0.2f}_x{x}_t{t}_Pf.dat'
        if os.path.exists(fname):
            Pfs.append(np.fromfile(fname))
        else:
            Pfs.append(None)
        fname = f'{prefix}_L{L}_{e2:0.2f}_x{x}_t{t}_Pb.dat'
        if os.path.exists(fname):
            Pbs.append(np.fromfile(fname))
        else:
            Pbs.append(None)
        fname = f'{prefix}_L{L}_{e2:0.2f}_x{x}_t{t}_MC.dat'
        if os.path.exists(fname):
            MCs.append(np.fromfile(fname))
        else:
            MCs.append(None)

    trace_ts_fwd = []
    trace_ts_bwd = []
    trace_fwd = []
    trace_bwd = []
    trace_ts_avg = []
    trace_avg = []

    for t,Pf,Pb in zip(ts, Pfs, Pbs):
        print(f't = {t}')
        # pt1_boots = list(map(
        #     lambda x: est_fwd_sigma(x[0]),
        #     al.bootstrap_gen(Pfs[i], Nboot=1000)))
        # ax.hist(pt1_boots, histtype='step')
        # pt1 = np.mean(pt1_boots), np.std(pt1_boots)
        if Pf is not None:
            pt1 = al.bootstrap(Pf, Nboot=1000, f=est_fwd_sigma)
            print(pt1)
            trace_ts_fwd.append(t+1)
            trace_fwd.append(pt1)
        if Pb is not None:
            pt2 = al.bootstrap(Pb, Nboot=1000, f=est_bwd_sigma)
            print(pt2)
            trace_ts_bwd.append(t + 0.02)
            trace_bwd.append(pt2)
        if Pf is not None and Pb is not None:
            pt3 = al.bootstrap(Pf, Pb, Nboot=1000, f=est_avg_sigma)
            print(pt3)
            trace_ts_avg.append(t + 0.5)
            trace_avg.append(pt3)

    # Missing data guard
    if len(trace_ts_fwd) == 0 and len(trace_ts_bwd) == 0 and len(trace_ts_avg) == 0:
        return None

    fig, ax = plt.subplots(1,1, figsize=(6,4))
    if len(trace_ts_fwd) > 0:
        al.add_errorbar(
            np.transpose(trace_fwd), xs=trace_ts_fwd, ax=ax,
            linestyle='', marker='o', markersize=6, capsize=2,
            fillstyle='none'
        )
    if len(trace_ts_bwd) > 0:
        al.add_errorbar(
            np.transpose(trace_bwd), xs=trace_ts_bwd, ax=ax,
            linestyle='', marker='d', markersize=6, capsize=2,
            fillstyle='none'
        )
    if len(trace_ts_avg) > 0:
        al.add_errorbar(
            np.transpose(trace_avg), xs=trace_ts_avg, ax=ax,
            linestyle='', marker='^', markersize=6, capsize=2,
            fillstyle='none'
        )

    fig.savefig(f'figs/raw_L{L}_{e2:0.2f}_x{x}.pdf')
    plt.close(fig)

    fig, ax = plt.subplots(1,1, figsize=(6,4))
    for t,MC in zip(ts, MCs):
        if MC is None: continue
        ax.plot(*al.bin_data(MC, binsize=100), label=rf'$t = {t}$')
    ax.legend()
    fig.savefig(f'figs/MC_L{L}_{e2:0.2f}_x{x}.pdf')
    plt.close(fig)

    if len(trace_avg) > 0:
        out_trace = trace_avg
    elif len(trace_fwd) > 0:
        out_trace = trace_fwd
    else:
        assert len(trace_bwd) > 0
        out_trace = trace_bwd
    sigma_mean = np.mean(np.transpose(out_trace)[0])
    sigma_err = np.sqrt(np.sum(np.transpose(out_trace)[1]**2) / len(out_trace))
    return sigma_mean, sigma_err

### VERSION 2: Based on dynamical wilson loop simulations
def compute_sigma_v2(e2, *, x, L):
    Nboot = 1000
    binsize = 10
    fit_pad = 3
    SIGMA_BIAS = 0.05
    Lt = L

    def fit_wloop(ts, Wt_est):
        f = lambda ts,A,B: A*np.exp(-B*ts)
        popt, pcov = sp.optimize.curve_fit(f, ts, Wt_est[0][ts], sigma=Wt_est[1][ts])
        return {
            'f': lambda ts: f(ts, *popt),
            'popt': popt
        }

    def est_sigma(Wt_hist):
        Wt = np.stack((al.rmean(Wt_hist), Wt_est[1]))
        res = fit_wloop(np.arange(fit_pad, Lt+1-fit_pad), Wt)
        return res['popt'][1]

    fname = f'raw_obs/dyn_wloop_stag_L{L}_{e2:0.2f}_x{x}_Wt_hist.dat'
    Wt_hist = np.fromfile(fname, dtype=np.float64).reshape(-1, Lt+1)
    counts = np.sum(Wt_hist, axis=-1)
    assert np.all(counts == counts[0])
    Wt_hist = Wt_hist * np.exp(-SIGMA_BIAS * np.arange(Lt+1))
    Wt_est = al.bootstrap(al.bin_data(Wt_hist, binsize=binsize)[1], Nboot=Nboot, f=al.rmean)
    print(f'{Wt_est=}')

    res = fit_wloop(np.arange(fit_pad, Lt+1-fit_pad), Wt_est)
    print(f'{res["popt"]=}')

    sigma = al.bootstrap(al.bin_data(Wt_hist, binsize=binsize)[1], Nboot=Nboot, f=est_sigma)
    print(f'{sigma=}')

    ts = np.arange(Lt+1)
    fig, ax = plt.subplots(1,1)
    al.add_errorbar(Wt_est, xs=ts, ax=ax, marker='s')
    ax.plot(ts, res['f'](ts), color='r', linestyle='-', marker='')
    ax.set_yscale('log')
    fig.savefig(f'figs/dyn_loop_L{L}_{e2:0.2f}_x{x}.pdf')
    plt.close(fig)

    return sigma


def fit_sigma_exp(e2s, sigma_over_e):
    ys, yerrs = sigma_over_e
    f = lambda e2,A,B: A*np.exp(-B/e2)
    popt,pcov = sp.optimize.curve_fit(f, e2s, ys, sigma=yerrs)
    return {
        'f': lambda e2: f(e2, *popt),
        'popt': popt,
        'pcov': pcov,
        'perr': np.sqrt(np.diag(pcov)),
    }

def fit_sigma_linear(e2s, sigma_over_e):
    ys, yerrs = sigma_over_e
    f = lambda e2,A,B: A/e2 + B
    popt,pcov = sp.optimize.curve_fit(f, e2s, ys, sigma=yerrs)
    return {
        'f': lambda e2: f(e2, *popt),
        'popt': popt,
        'pcov': pcov,
        'perr': np.sqrt(np.diag(pcov)),
    }

def main():
    fig, axes = plt.subplots(2,2, figsize=(6,6), sharex='col', sharey='row')

    all_e2s = np.array([
        0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00,
        1.10, 1.20, 1.30, 1.40, 1.60, 1.80, 2.00,
        2.50, 3.00, 3.50, 4.00
    ])

    x = 6
    fit_L = 16

    Ls = {
        16: dict(marker='o', color='k'),
        32: dict(marker='s', color='xkcd:purple')
    }

    for L,L_style in Ls.items():
        sigma_trace = []
        e2s = []
        for e2 in all_e2s:
            # sigma_est = compute_sigma(e2, x=x, L=L)
            sigma_est = compute_sigma_v2(e2, x=x, L=L)
            if sigma_est is None: continue
            e2s.append(e2)
            sigma_trace.append(sigma_est)
        e2s = np.array(e2s)
        sigma_trace = np.transpose(sigma_trace)
        sigma_over_e_trace = sigma_trace / np.sqrt(e2s)

        style = dict(**L_style, markersize=6, fillstyle='none', linestyle='')
        al.add_errorbar(sigma_over_e_trace, xs=1/e2s, ax=axes[0,0], **style)
        al.add_errorbar(sigma_over_e_trace, xs=e2s, ax=axes[0,1], **style)
        al.add_errorbar(sigma_trace, xs=1/e2s, ax=axes[1,0], **style)
        al.add_errorbar(sigma_trace, xs=e2s, ax=axes[1,1], **style)

        if L != fit_L: continue

        # fit 1: sigma = A*e*exp(-B/e^2)
        inds = np.arange(7)
        res = fit_sigma_exp(e2s[inds], sigma_over_e_trace[:,inds])
        A, B = res['popt']
        min_e2, max_e2 = np.min(e2s[inds]), np.max(e2s[inds])
        fit_e2s = np.linspace(min_e2, max_e2, num=100, endpoint=True)
        fit_ys = res['f'](fit_e2s)
        print(res)
        style = dict(color='r', linestyle='-')
        axes[0,0].plot(
            1/fit_e2s, fit_ys, **style, label=rf'Fit ${A:.2f} e \exp(-{B:.2f}/e^2)$')
        axes[0,1].plot(fit_e2s, fit_ys, **style)
        axes[1,0].plot(1/fit_e2s, np.sqrt(fit_e2s)*fit_ys, **style)
        axes[1,1].plot(fit_e2s, np.sqrt(fit_e2s)*fit_ys, **style)

        # fit 2: sigma = A*e + B/e
        inds = np.arange(7)
        res = fit_sigma_linear(e2s[inds], sigma_over_e_trace[:,inds])
        A, B = res['popt']
        min_e2, max_e2 = np.min(e2s[inds]), np.max(e2s[inds])
        fit_e2s = np.linspace(min_e2, max_e2, num=100, endpoint=True)
        fit_ys = res['f'](fit_e2s)
        print(res)
        style = dict(color='xkcd:forest green', linestyle='-')
        axes[0,0].plot(
            1/fit_e2s, fit_ys, **style, label=rf'Fit ${A:.2f} e +  {B:.2f}/e$')
        axes[0,1].plot(fit_e2s, fit_ys, **style)
        axes[1,0].plot(1/fit_e2s, np.sqrt(fit_e2s)*fit_ys, **style)
        axes[1,1].plot(fit_e2s, np.sqrt(fit_e2s)*fit_ys, **style)

    for ax in axes.flatten():
        ax.set_yscale('log')
        # ax.set_ylim(0, 0.1)
    axes[-1,0].set_xlabel(r'$1/e^2$')
    axes[-1,1].set_xlabel(r'$e^2$')
    axes[0,0].set_ylabel(r'$\sigma / e$')
    axes[1,0].set_ylabel(r'$\sigma$')
    axes[0,0].legend()
    fig.savefig('figs/analyze_sigma.pdf')

if __name__ == '__main__': main()
