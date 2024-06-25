import analysis as al
import matplotlib.pyplot as plt
import numpy as np
import os
import paper_plt
# paper_plt.load_basic_config()
paper_plt.load_latex_config()
import scipy as sp
import scipy.optimize
import tqdm

topdir = 'raw_obs_more_mom'
figdir = 'figs'

max_abs_n = 6
max_n2 = max_abs_n**2
wvs = []
for n0 in range(-max_abs_n, max_abs_n+1):
    for n1 in range(n0, max_abs_n+1):
        if n0**2 + n1**2 <= max_n2:
            wvs.append((n0,n1))

def int_to_fname_str(n):
    if n >= 0:
        return str(n)
    else:
        return 'm'+str(-n)

def load_all_Ch_mom(prefix, *, L):
    all_Ch_mom = []
    all_Ch_mom_stag = []
    for wv in wvs:
        n = wv
        n_str = '_'.join(map(int_to_fname_str, n))
        fname = f'{prefix}_Ch_mom_{n_str}.dat'
        Ch_mom = np.fromfile(fname, dtype=np.complex128).reshape(-1, L)
        all_Ch_mom.append(Ch_mom)
        n_stag = (wv[0] + L//2, wv[1] + L//2)
        n_str = '_'.join(map(int_to_fname_str, n_stag))
        fname = f'{prefix}_Ch_mom_{n_str}.dat'
        Ch_mom = np.fromfile(fname, dtype=np.complex128).reshape(-1, L)
        all_Ch_mom_stag.append(Ch_mom)
    return np.array(all_Ch_mom), np.array(all_Ch_mom_stag)

def build_corr(Ch_mom):
    L = Ch_mom.shape[-1]
    Ch_mom_t = []
    for dt in range(L):
        Ch_mom_t.append(np.mean(np.conj(Ch_mom) * np.roll(Ch_mom, -dt, axis=-1), axis=-1))
    return np.transpose(Ch_mom_t)

def fit_cosh(y, yerr, *, xs, L):
    f = lambda t,A,E: A*np.cosh(E*(t - L/2))
    if y[0]/y[L//2] > 0:
        E0 = np.log(y[0]/y[L//2])/(L/2)
        A0 = y[0] / np.cosh(E0*L/2)
        p0 = [A0, E0]
    else:
        p0 = [y[0], 0.0]
    popt, pcov = sp.optimize.curve_fit(
        f, xs, y, sigma=yerr, p0=p0,
        bounds=([0, 0], [np.inf, 1.0]))
    fopt = lambda t: f(t, *popt)
    resid = y - fopt(xs)
    chisq = np.sum((resid / yerr)**2)
    chisq_per_dof = chisq / (len(y) - len(popt))
    return {
        'popt': popt,
        'f': fopt,
        'chisq': chisq,
        'chisq_per_dof': chisq_per_dof
    }

def fit_exp(y, yerr, *, xs, L):
    f = lambda t,A,E: A*np.exp(-t*E)
    if y[0]/y[-1] > 0:
        E0 = min(3.0, np.log(y[0]/y[-1])/(xs[-1] - xs[0]))
        A0 = y[0]
        p0 = [A0, E0]
    else:
        p0 = [y[0], 0.0]
    popt, pcov = sp.optimize.curve_fit(
        f, xs, y, sigma=yerr, p0=p0,
        bounds=([0, 0], [np.inf, 5.0]))
    fopt = lambda t: f(t, *popt)
    resid = y - fopt(xs)
    chisq = np.sum((resid / yerr)**2)
    chisq_per_dof = chisq / (len(y) - len(popt))
    return {
        'popt': popt,
        'f': fopt,
        'chisq': chisq,
        'chisq_per_dof': chisq_per_dof
    }

def load_and_analyze_trace(L, e2):
    prefix = f'{topdir}/obs_trace_{e2:0.2f}_L{L}_cluster'
    Ch_mom, Ch_mom_stag = load_all_Ch_mom(prefix, L=L)
    print(f'{Ch_mom.shape=}')

    def wv_to_k2(wv):
        k2 = 0.0
        for ni in wv:
            ki = 2*np.pi*ni/L
            k2 += ki**2
        return k2
    k2s = np.array(list(map(wv_to_k2, wvs)))
    print(list(zip(wvs, k2s)))
    Cks = list(map(build_corr, Ch_mom))
    Cks_stag = list(map(build_corr, Ch_mom_stag))

    wv_to_xs = []
    wv_to_xs_stag = []
    xs = []
    xs_stag = []
    Ck_errs = [] 
    Ck_stag_errs = []
    print('Computing full-ens errors...')
    for wv, k2, Ck, Ck_stag in zip(tqdm.tqdm(wvs), k2s, Cks, Cks_stag):
        if k2 == 0:
            # dummy values
            wv_to_xs.append(None)
            wv_to_xs_stag.append(None)
            Ck_errs.append(None)
            Ck_stag_errs.append(None)
            continue
        n = wv
        n_stag = (wv[0] + L//2, wv[1] + L//2)
        x = 2 - np.cos(2*np.pi*n[0]/L) - np.cos(2*np.pi*n[1]/L)
        x_stag = 2 - np.cos(2*np.pi*n_stag[0]/L) - np.cos(2*np.pi*n_stag[1]/L)
        for i,y in enumerate(xs):
            if np.isclose(x, y, rtol=1e-3):
                wv_to_xs.append(i)
                break
        else:
            wv_to_xs.append(len(xs))
            xs.append(x)
        for i,y in enumerate(xs_stag):
            if np.isclose(x_stag, y, rtol=1e-3):
                wv_to_xs_stag.append(i)
                break
        else:
            wv_to_xs_stag.append(len(xs_stag))
            xs_stag.append(x_stag)
        # xs.append()
        # xs_stag.append()
        Ck_est = al.bootstrap(Ck, Nboot=1000, f=al.rmean)
        Ck_stag_est = al.bootstrap(Ck_stag, Nboot=1000, f=al.rmean)
        Ck_errs.append(Ck_est[1])
        Ck_stag_errs.append(Ck_stag_est[1])
    print('Done.')

    def do_analysis_on_inds(inds, *, stag):
        # Eks = []
        if stag:
            Eks = [[] for _ in range(len(xs_stag))]
            my_Cks = Cks_stag
            my_Ck_errs = Ck_stag_errs
            my_wv_to_xs = wv_to_xs_stag
            fit_kind = fit_exp
            tf = 3
        else:
            Eks = [[] for _ in range(len(xs))]
            my_Cks = Cks
            my_Ck_errs = Ck_errs
            my_wv_to_xs = wv_to_xs
            fit_kind = fit_cosh
            tf = len(Cks[0])

        # fig, axes = plt.subplots(2,1, figsize=(6,5))
        for wv, xi, k2, Ck, Ck_err in zip(wvs, my_wv_to_xs, k2s, my_Cks, my_Ck_errs):
            if k2 == 0: continue
            ts = np.arange(L)

            fit_boot = lambda y: fit_kind(
                al.rmean(y)[:tf], Ck_err[:tf], xs=ts[:tf], L=L)["popt"]
            fit_params = fit_boot(Ck[inds])
            E_fit = fit_params[1]
            E2_fit = E_fit**2
            A_fit = fit_params[0]
            Eks[xi].append(E2_fit)
            # print(f'Fit E (unstag) = {E_fit}')
            # al.add_errorbar(Ck_est, ax=axes[0], label=f'{k2=}')
            # fit_ys = A_fit[0]*np.cosh(E_fit[0]*(ts-L/2))
            # axes[0].plot(ts, fit_ys, label=f'{k2=}')
            
            # stag
            # Ck_est_stag = al.bootstrap(Ck_stag, Nboot=1000, f=al.rmean)
            # tf = 3
            # fit_boot_exp = lambda y: fit_exp(
            #     al.rmean(y)[:tf], Ck_est_stag[1][:tf], xs=ts[:tf], L=L)["popt"]
            # fit_params = al.bootstrap(Ck_stag, Nboot=20, f=fit_boot_exp)
            # E_fit = (fit_params[0][1], fit_params[1][1])
            # E2_fit = (E_fit[0]**2, 2*E_fit[1]*E_fit[0])
            # A_fit = (fit_params[0][0], fit_params[1][0])
            # Eks_stag.append(E2_fit)
            # print(f'Fit E (stag) = {E_fit}')
            # al.add_errorbar(Ck_est_stag, ax=axes[1], label=f'{k2=}')
            # fit_ys = A_fit[0]*np.exp(-E_fit[0]*ts[:tf])
            # axes[1].plot(ts[:tf], fit_ys, label=f'{k2=}')
        # for ax in axes:
        #     ax.set_yscale('log')
        # fig.savefig(f'{figdir}/analyze_Ch_mom_{e2:0.2f}_L{L}.pdf')
        out = []
        for Eks_i in Eks:
            out.append(np.mean(Eks_i))
        # Eks = np.array(Eks)
        # Eks_stag = np.transpose(Eks_stag)
        if stag:
            assert len(out) == len(xs_stag)
        else:
            assert len(out) == len(xs)
        return np.array(out)

    print('Bootstrapping...')

    Eks = []
    Eks_stag = []
    for boot_inds in al.bootstrap_gen(np.arange(len(Cks[0])), Nboot=20):
        Ek_boot = do_analysis_on_inds(boot_inds, stag=False)
        Ek_stag_boot = do_analysis_on_inds(boot_inds, stag=True)
        Eks.append(Ek_boot)
        Eks_stag.append(Ek_stag_boot)
    Eks = np.mean(Eks, axis=0), np.std(Eks, axis=0)
    Eks_stag = np.mean(Eks_stag, axis=0), np.std(Eks_stag, axis=0)
    print(Eks[0].shape)
    print(len(xs))

    return {
        'unstag': (xs, Eks),
        'stag': (xs_stag, Eks_stag)
    }
    

def main():
    L = 96
    fig, ax = plt.subplots(1,1, figsize=(4,3))
    # e2s = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    e2s = [0.60, 0.70, 0.80, 0.90]
    # cmap = plt.get_cmap('viridis')
    style = dict(markersize=3, fillstyle='none', linestyle='')
    markers = ['o', 's', 'd', 'x']
    colors = ['k', 'tab:blue', 'tab:red', '0.5']
    for i, (e2, marker, color) in enumerate(zip(e2s, markers, colors)):
        cache_fname = f'{topdir}/cache_{e2:0.2f}_L{L}.npy'
        if not os.path.exists(cache_fname):
            res = load_and_analyze_trace(L, e2)
            np.save(cache_fname, res)
        res = np.load(cache_fname, allow_pickle=True).item()
        xs, Eks = res['unstag']
        inds = np.nonzero(2*np.array(xs) < 0.13)
        xs = 2*np.array(xs)[inds]
        more_style = dict(
            # color=cmap(i / len(e2s)),
            marker=marker,
            color=color,
            label=f'$e^2 = {e2:0.2f}$')
        Eks = (
            4*np.sinh(np.sqrt(Eks[0][inds]) / 2)**2,
            Eks[1][inds]
        )
        al.add_errorbar(Eks, xs=xs, ax=ax, **more_style, **style)
    ax.plot([0,1], [0,1], color='k', marker='', linestyle='-')
    ax.legend(ncol=2)
    ax.set_ylim(0, 0.165)
    ax.set_xlim(0, 0.165)
    ax.set_aspect(1.0)
    # ax.set_xlabel(r'$4-2\cos(a k_1)-2\cos(a k_2)$')
    # ax.set_ylabel(r'$4 \sinh(a E/ 2)^2$')
    ax.set_xlabel(r'$\widehat{k}^2$')
    ax.set_ylabel(r'$\widehat{E}^2$')
    fig.savefig(f'{figdir}/analyze_E_vs_k.pdf')
    # fig, axes = plt.subplots(3,1, figsize=(6,6))
    # style = dict(marker='o', markersize=4, fillstyle='none', linestyle='')
    # al.add_errorbar(Eks, xs=xs, ax=axes[0], color='k', **style)
    # al.add_errorbar(Eks_stag, xs=xs_stag, ax=axes[0], color='r', **style)
    # al.add_errorbar(Eks, xs=xs, ax=axes[1], color='k', **style)
    # al.add_errorbar(Eks_stag, xs=xs_stag, ax=axes[2], color='r', **style)
    # for ax in axes:
    #     ax.set_xlabel(r'$2-\cos(k_0)-\cos(k_1)$')
    #     ax.set_ylabel(r'$E^2$')
    # fig.savefig(f'{figdir}/analyze_E_vs_k.pdf')


if __name__ == '__main__':
    main()
