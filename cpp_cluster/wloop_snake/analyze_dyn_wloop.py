import analysis as al
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import paper_plt
paper_plt.load_latex_config()
import scipy as sp
import scipy.optimize

BIAS = 0.03
raw_data_prefix = 'raw_obs_v2'
data_prefix = 'data'
figs_prefix = 'figs'

def sigma_eff(Wt_hist_cfgs):
    Wt_hist = al.rmean(Wt_hist_cfgs)
    return -np.log(Wt_hist[1:]) + np.log(Wt_hist[:-1])

def measure_log_ratio(Wt_hist_cfgs):
    Wt_hist = al.rmean(Wt_hist_cfgs)
    return np.log(Wt_hist[0]) - np.log(Wt_hist[-1])

def load_data(L, e2, x, version, *, BIAS=BIAS):
    Wt_hist = np.fromfile(
        f'{raw_data_prefix}/dyn_wloop_v{version}_stag_L{L}_{e2:.1f}_x{x}_Wt_hist.dat', dtype=np.float64
    ).reshape(-1, L+1)
    Wt_hist *= np.exp(-BIAS*np.arange(L+1))
    xs_bin, Wt_hist_bin = al.bin_data(Wt_hist, binsize=100)
    Wt_est = al.bootstrap(Wt_hist_bin, Nboot=1000, f=al.rmean)
    sigma = al.bootstrap(Wt_hist_bin, Nboot=1000, f=sigma_eff)
    ratio = al.bootstrap(Wt_hist_bin, Nboot=1000, f=measure_log_ratio)
    ratio = (ratio[0]/L, ratio[1]/L)
    return {
        'Wt': Wt_est,
        'sigma': sigma,
        'ratio': ratio,
        'mcmc_trace_0': (xs_bin, Wt_hist_bin[:,0]),
        'mcmc_trace_m1': (xs_bin, Wt_hist_bin[:,-1])
    }

def make_fitter(f, g, pnames, **extra_kwargs):
    def do_fit(xs, ys_with_errs):
        ys, sigma = ys_with_errs
        popt, pcov = sp.optimize.curve_fit(
            f, xs, ys, sigma=sigma, **extra_kwargs)
        fopt = lambda x: f(x, *popt)
        resid = fopt(xs) - ys
        chisq = np.sum(resid**2 / sigma**2)
        chisq_per_dof = chisq / (len(xs) - len(popt))
        print(f'Fit {chisq=} {chisq_per_dof=}')
        return {
            'f': fopt,
            'g': lambda x: g(x, *popt),
            'popt': popt,
            'pdict': {k: v for k,v in zip(pnames, popt)},
            'chisq': chisq,
            'chisq_per_dof': chisq_per_dof
        }
    return do_fit

fit_sigma_eff_1exp = make_fitter(
    lambda r, sigma, alpha, beta: sigma + alpha*np.exp(-beta*r),
    lambda r, sigma, alpha, beta: sigma*r + alpha*np.exp(-beta*r)/(np.exp(-beta) - 1),
    ['sigma', 'alpha', 'beta'],
    p0=[0.01, 0.0, 0.1],
    bounds=[[0.0, 0.0, 0.0], [np.inf, np.inf, np.inf]],
    maxfev=10000)
fit_sigma_eff_luscher = make_fitter(
    lambda r, sigma, gamma: sigma + gamma/(r*(r+1)),
    lambda r, sigma, gamma: sigma*r - gamma/r,
    ['sigma', 'gamma'],
    p0=[np.pi/24, 0.01],
    bounds=[[-np.inf, 0.0], [np.inf, np.inf]],
    maxfev=10000)
fit_sigma_eff_power = make_fitter(
    lambda r, sigma, beta, gamma: sigma + gamma*(np.exp(-beta*(r+1))/np.sqrt(r+1) - np.exp(-beta*r)/np.sqrt(r)),
    lambda r, sigma, beta, gamma: sigma*r + gamma*np.exp(-beta*r)/np.sqrt(r),
    ['sigma', 'beta', 'gamma'],
    p0=[0.01, 0.1, 0.0],
    bounds=[[0.0, 0.0, -np.inf], [np.inf, np.inf, np.inf]],
    maxfev=10000)
    
    

def main1():
    e2 = 1.0
    
    fig1, ax1 = plt.subplots(1,1)
    fig2, ax2 = plt.subplots(1,1)
    fig3, ax3 = plt.subplots(1,1)
    style = dict(marker='o', markersize=4, fillstyle='none', linestyle='', capsize=3)

    trace = []
    L = 96
    xs = np.array([4,6,8,10,12,14,16,18,20,22,24,26])
    for x in xs:
        res = load_data(L=L, e2=e2, x=x, version=2)
        ratio = res['ratio']
        trace.append(ratio)
        al.add_errorbar(res['Wt'], ax=ax1, label=f'L={L}, ratio={ratio[0]:.3f}+/-{ratio[1]:.3f}')
        al.add_errorbar(res['sigma'], ax=ax2, label=f'L={L}')
    trace = np.transpose(trace)
    res = fit_sigma_eff_1exp(xs[1:], trace[:,1:])
    print(f'Fit L={L} params {res["popt"]}')
    al.add_errorbar(trace, xs=xs, ax=ax3, **style, label=f'L={L}')
    xs_fit = np.linspace(4, 36, num=100)
    ax3.plot(xs_fit, res['f'](xs_fit), color='k', linewidth=0.5)

    trace = []
    L = 64
    xs = np.array([6,7,8,9,10,12,14,16,18,20,22,24,26,28,30,32])
    for x in xs:
        res = load_data(L=L, e2=e2, x=x, version=2)
        ratio = res['ratio']
        trace.append(ratio)
        al.add_errorbar(res['Wt'], ax=ax1, label=f'L={L}, ratio={ratio[0]:.3f}+/-{ratio[1]:.3f}')
        al.add_errorbar(res['sigma'], ax=ax2, label=f'L={L}')
    trace = np.transpose(trace)
    res = fit_sigma_eff_1exp(xs[4:], trace[:,4:])
    print(f'Fit L={L} params {res["popt"]}')
    al.add_errorbar(trace, xs=xs, ax=ax3, **style, label=f'L={L}')
    xs_fit = np.linspace(4, 36, num=100)
    ax3.plot(xs_fit, res['f'](xs_fit), color='k', linewidth=0.5)

    trace = []
    L = 32
    xs = np.array([6,7,8,9])
    for x in xs:
        res = load_data(L=L, e2=e2, x=x, version=2)
        ratio = res['ratio']
        trace.append(ratio)
        al.add_errorbar(res['Wt'], ax=ax1, label=f'L={L}, ratio={ratio[0]:.3f}+/-{ratio[1]:.3f}')
        al.add_errorbar(res['sigma'], ax=ax2, label=f'L={L}')
    trace = np.transpose(trace)
    al.add_errorbar(trace, xs=xs, ax=ax3, **style, label=f'L={L}')

    trace = []
    L = 16
    xs = np.array([6])
    for x in xs:
        res = load_data(L=L, e2=e2, x=x, version=2)
        ratio = res['ratio']
        trace.append(ratio)
        al.add_errorbar(res['Wt'], ax=ax1, label=f'L={L}, ratio={ratio[0]:.3f}+/-{ratio[1]:.3f}')
        al.add_errorbar(res['sigma'], ax=ax2, label=f'L={L}')
    trace = np.transpose(trace)
    al.add_errorbar(trace, xs=xs, ax=ax3, **style, label=f'L={L}')

    ax1.legend()
    ax2.legend()
    ax3.legend()
    fig1.savefig(f'{figs_prefix}/compare_{e2:0.1f}_wloops.pdf')
    fig2.savefig(f'{figs_prefix}/compare_{e2:0.1f}_sigma_xt.pdf')
    fig3.savefig(f'{figs_prefix}/compare_{e2:0.1f}_sigma_x.pdf')

def make_plots_for_e2_sweep(e2, *, L, xs, global_ax, fit_kind, **extra_style):
    fig1, axes1 = plt.subplots(2,1, figsize=(6, 8), tight_layout=True)
    fig2, ax2 = plt.subplots(1,1)
    ncols3 = 4
    nrows3 = 1+(len(xs)-1)//ncols3
    fig3, axes3 = plt.subplots(
        nrows3, ncols3, figsize=(9, 2*nrows3), gridspec_kw=dict(
            hspace=0.01, wspace=0.05
        ))
    style = dict(markersize=4, fillstyle='none', linestyle='', **extra_style)

    trace = []
    cmap = plt.get_cmap('viridis')
    colors = cmap(np.arange(len(xs))/len(xs))
    for x,color,ax3 in zip(xs, colors, axes3.flatten()):
        res = load_data(L=L, e2=e2, x=x, version=2)
        ax3.plot(*res['mcmc_trace_0'], linestyle='-', marker='', color='b')
        ax3.plot(*res['mcmc_trace_m1'], linestyle='-', marker='', color='r')
        ax3.set_xticks([])
        ax3.set_yscale('log')
        # ax3.set_yticks([])
        ax3.text(0.1, 0.9, f'$x={x}$', ha='left', va='top', transform=ax3.transAxes)
        ratio = res['ratio']
        trace.append(ratio)
        al.add_errorbar(
            res['Wt'], ax=axes1[0],
            label=rf'$L={L}$, $100\Delta f = {100*ratio[0]:.2f}({int(10000*ratio[1])})$',
            color=color
        )
        al.add_errorbar(
            (res['Wt'][0]*np.exp(BIAS*np.arange(L+1)),
             res['Wt'][1]*np.exp(BIAS*np.arange(L+1))),
            ax=axes1[1], color=color)
        # al.add_errorbar(res['Wt'], ax=ax1, label=f'L={L}, ratio={ratio[0]:.3f}+/-{ratio[1]:.3f}')
        al.add_errorbar(res['sigma'], ax=ax2, label=f'L={L}')
    trace = np.transpose(trace)
    fit_min = 8
    fit_max = L//2-4
    xs_fit_input = xs[fit_min:fit_max+1]
    ys_fit_input = trace[:,fit_min:fit_max+1]
    xs_fit = np.linspace(6, 36, num=100)
    if fit_kind == '1exp':
        res = fit_sigma_eff_1exp(xs_fit_input, ys_fit_input)
        pdict = res['pdict']
        print(f'Fit L={L} with 1-exp: params {pdict}')
        fit_str = (
            f'${pdict["sigma"]:0.4f} + {pdict["alpha"]:0.4f} '
            rf'\exp(-{pdict["beta"]:0.4f} r)$')
    elif fit_kind == 'luscher':
        res = fit_sigma_eff_luscher(xs_fit_input, ys_fit_input)
        pdict = res['pdict']
        print(f'Fit L={L} with luscher: params {pdict}')
        fit_str = (
            f'${pdict["sigma"]:0.4f} + {pdict["gamma"]:0.4f}/(r(r+1))$')
    elif fit_kind == 'power':
        res = fit_sigma_eff_power(xs_fit_input, ys_fit_input)
        pdict = res['pdict']
        print(f'Fit L={L} with power: params {pdict}')
        fit_str = (
            rf'${pdict["sigma"]:0.4f} + {pdict["gamma"]:0.4f}'
            rf'\exp(-{pdict["beta"]:0.4f} r)/\sqrt{{r}}$')
    else:
        raise RuntimeError(f'unknown {fit_kind=}')
    global_ax.axvspan(fit_min-0.25, fit_max+0.25, color='0.9', ec='none')
    al.add_errorbar(trace, xs=xs, ax=global_ax, **style, label=f'$e^2={e2:0.1f}$ data')
    fit_color = style['color'] if 'color' in style else 'k'
    global_ax.plot(
        xs_fit, res['f'](xs_fit), color=fit_color, linewidth=0.5,
        label=(
            f'$e^2={e2:0.1f}$ fit: {fit_str} '
            f'[chisq/dof={res["chisq_per_dof"]:.2f}]'))

    axes1[0].legend(ncol=2)
    axes1[0].text(0.2, 0.9, 'Reweighted', transform=axes1[0].transAxes)
    axes1[1].text(0.2, 0.9, 'Unweighted', transform=axes1[1].transAxes)
    ax2.legend(ncol=2)
    fig1.savefig(f'{figs_prefix}/compare_{e2:0.1f}_wloops.pdf')
    fig2.savefig(f'{figs_prefix}/compare_{e2:0.1f}_sigma_xt.pdf')
    fig3.savefig(f'{figs_prefix}/compare_{e2:0.1f}_mcmc.pdf')
    plt.close(fig1)
    plt.close(fig2)
    plt.close(fig3)

def make_plots_for_e2_integrated_E(
        e2, *, L, xmin, xmax, global_ax1, global_ax2, fit_kind, **extra_style):
    style = dict(markersize=4, fillstyle='none', linestyle='', **extra_style)
    style_ale = dict(style)
    style_ale['color'] = 'k'

    xs = np.arange(xmin, xmax+1)
    xs_E = np.arange(xmin, xmax+2)
    trace = []
    trace_E = []
    E = (0.0, 0.0)
    for x in xs:
        res = load_data(L=L, e2=e2, x=x, version=2)
        ratio = res['ratio']
        trace.append(ratio)
        trace_E.append(E)
        E = (E[0] + ratio[0], np.sqrt(E[1]**2 + ratio[1]**2))
    trace_E.append(E)
    trace_E = np.transpose(trace_E)
    np.savetxt(
        f'{data_prefix}/e2_{e2:0.1f}_L{L}_E.txt',
        np.transpose(np.concatenate((xs_E[np.newaxis,:], trace_E), axis=0)),
        fmt=['%d','%.4e','%.4e']
    )
    fname_ale = f'{data_prefix}/ale_e2_{e2:0.1f}_L{L}_E.txt'
    if os.path.exists(fname_ale):
        trace_E_ale = np.loadtxt(fname_ale)
        xs_E_ale = trace_E_ale[:,0]
        trace_E_ale = np.transpose(trace_E_ale[:,1:])
        zero_ind = np.argmax(np.isclose(xs_E_ale, xmin))
        trace_E_ale[0,:] -= trace_E_ale[0,zero_ind]
    else:
        fname_ale = None
    trace = np.transpose(trace)

    # plot data
    al.add_errorbar(trace, xs=xs, ax=global_ax1, **style, label=f'$e^2={e2:0.1f}$ data')
    al.add_errorbar(trace_E, xs=xs_E, ax=global_ax2, **style, label=f'$e^2={e2:0.1f}$ data')
    if fname_ale is not None:
        al.add_errorbar(trace_E_ale, xs=xs_E_ale, ax=global_ax2, **style_ale, label=f'$e^2={e2:0.1f}$ Alessandro data')

    # fit data
    fit_x = 5
    fit_start = np.argmin((xs - fit_x)**2)
    xs_fit_input = xs[fit_start:]
    ys_fit_input = trace[:,fit_start:]
    xs_fit = np.linspace(xmin, xmax+4, num=100)
    if fit_kind == '1exp':
        res = fit_sigma_eff_1exp(xs_fit_input, ys_fit_input)
        pdict = res['pdict']
        print(f'Fit L={L} with 1-exp: params {pdict}')
        fit_str = (
            rf'${pdict["sigma"]:0.4f} + {pdict["alpha"]:0.4f} '
            rf'\exp(-{pdict["beta"]:0.4f} r)$')
        fit_E_str = (
            rf'${pdict["sigma"]:0.4f} r + \frac{{{pdict["alpha"]:0.4f}}}'
            rf'{{e^{{-{pdict["beta"]:0.4f}}} - 1}} \exp(-{pdict["beta"]:0.4f} r)$')
    elif fit_kind == 'luscher':
        res = fit_sigma_eff_luscher(xs_fit_input, ys_fit_input)
        pdict = res['pdict']
        print(f'Fit L={L} with luscher: params {pdict}')
        fit_str = (
            rf'${pdict["sigma"]:0.4f} + {pdict["gamma"]:0.4f}/(r(r+1))$')
        fit_E_str = (
            rf'${pdict["sigma"]:0.4f} r - {pdict["gamma"]:0.4f}/r$')
    else:
        raise RuntimeError(f'unknown {fit_kind=}')

    # plot fit
    fit_color = style['color'] if 'color' in style else 'k'
    global_ax1.plot(
        xs_fit, res['f'](xs_fit), color=fit_color, linewidth=0.5,
        label=(
            f'$e^2={e2:0.1f}$ fit: {fit_str} '
            f'[chisq/dof={res["chisq_per_dof"]:.2f}]'))
    gs_fit = res['g'](xs_fit)
    gs_fit_off = trace_E[0,fit_start]
    fit_start = np.argmin((xs_fit - fit_x)**2)
    gs_fit += gs_fit_off - gs_fit[fit_start]
    global_ax2.plot(
        xs_fit, gs_fit, color=fit_color, linewidth=0.5,
        label=(
            f'$e^2={e2:0.1f}$ fit: {fit_E_str} '
            f'[chisq/dof={res["chisq_per_dof"]:.2f}]'))


def main2(*, fit_kind):
    fig, ax = plt.subplots(1,1, figsize=(8,4))
    cmap = plt.get_cmap('RdBu')
    e2_style = {
        0.6: dict(color=cmap(0.1), marker='o'),
        # 0.7: dict(color=cmap(0.2), marker='s'),
        # 0.8: dict(color=cmap(0.3), marker='v'),
        # 0.9: dict(color=cmap(0.4), marker='^'),
        # 1.0: dict(color=cmap(0.5), marker='x'),
        # 1.1: dict(color=cmap(0.6), marker='d'),
        # 1.2: dict(color=cmap(0.7), marker='p'),
        # 1.3: dict(color=cmap(0.9), marker='h'),
        # 1.4: dict(color=cmap(1.0), marker='*')
    }
    xs = np.arange(0, 32+1, 1)
    L = 64
    for e2,style in e2_style.items():
        make_plots_for_e2_sweep(
            e2, L=L, xs=xs, global_ax=ax, fit_kind=fit_kind, **style)
    ax.set_xlabel(r'$r$')
    ax.set_xlim(-0.2, 40)
    ax.set_ylim(0, 0.05)
    ax.legend(ncol=2)
    ax.text(0.02, 0.98, f'Fit kind: {fit_kind}', ha='left', va='top', transform=ax.transAxes)
    fig.savefig(f'{figs_prefix}/compare_all_e2_sigma_x_fit_{fit_kind}.pdf')
    plt.close(fig)

def main3(e2, *, fit_kind):
    fig, axes = plt.subplots(2,1, figsize=(8,6), sharex=True)
    xmin = 0
    xmax = 32
    L = 64
    style = dict(color='r', marker='o')
    make_plots_for_e2_integrated_E(
        e2, L=L, xmin=xmin, xmax=xmax, global_ax1=axes[0], global_ax2=axes[1],
        fit_kind=fit_kind, **style)
    axes[1].set_xlabel(r'$r$')
    axes[1].set_xlim(-0.2, 40)
    axes[1].set_ylabel(r'$E$')
    # axes[0].set_ylim(0, 1.00)
    axes[0].text(0.02, 0.98, f'Fit kind: {fit_kind}', ha='left', va='top', transform=axes[0].transAxes)
    axes[0].set_ylabel(r'$\sigma_{\mathrm{eff}}$')
    for ax in axes:
        ax.legend(ncol=2)
    fig.savefig(f'{figs_prefix}/compare_e2_{e2:.1f}_sigma_x_all_x_fit_{fit_kind}.pdf')
    plt.close(fig)

if __name__ == '__main__':
    os.makedirs(figs_prefix, exist_ok=True)
    os.makedirs(data_prefix, exist_ok=True)
    # main1()
    main2(fit_kind='1exp')
    # main2(fit_kind='luscher')
    # main2(fit_kind='power')
    # for e2 in [0.6, 0.7, 0.8, 0.9, 1.0]:
    #     main3(e2, fit_kind='1exp')
    #     main3(e2, fit_kind='luscher')
