import analysis as al
import matplotlib as mpl
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import scipy.optimize

# def compute_Ch(Ch_mom):
#     L = Ch_mom.shape[-1]
#     Ch = []
#     for t in range(L):
#         Ch.append(np.mean(Ch_mom * np.roll(Ch_mom, -t, axis=-1), axis=-1))
#     Ch = np.transpose(Ch)
#     return Ch

# def get_shift(A, m, L):
#     return A/L*(np.exp(m) + 1)/(np.exp(m) - 1)*(1 - np.exp(-m*L))    

# def fit_Ch(ts, Ch_est, *, L, p0=None, verbose=False):
#     mean, err = Ch_est
#     f = lambda ts,A,m: (
#         A*(np.exp(-m*ts) + np.exp(-m*(L - ts))) - get_shift(A, m, L)
#     )
#     if p0 is None:
#         p0 = [1.0, 0.01]
#     popt, pcov = sp.optimize.curve_fit(
#         f, ts, mean, sigma=err, maxfev=100000, p0=p0,
#         bounds=[[0.0, 0.0], [np.inf,  np.inf]])
#     fopt = lambda ts: f(ts, *popt)
#     resid = fopt(ts) - mean
#     chisq = np.sum(resid**2 / err**2)
#     chisq_per_dof = chisq / (len(mean) - len(popt))
#     if verbose: print(f'{chisq=} {chisq_per_dof=} {popt=}')
#     return {
#         'params': popt,
#         'fopt': fopt,
#         'f': f
#     }

# def load_and_fit(fname, *, L, ax):
#     ts = np.arange(L)
#     Ch_mom = np.fromfile(fname).reshape(-1, L) / L**2
#     Ch = compute_Ch(Ch_mom)
#     Ch_mean, Ch_err = Ch_est = al.bootstrap(Ch, Nboot=1000, f=al.rmean)

#     # fit window
#     t_fit = ts[4:-4]

#     # main fit
#     res = fit_Ch(t_fit, (Ch_mean[t_fit], Ch_err[t_fit]), L=L, verbose=True)
#     p0 = res['params']
#     print(f'{p0=}')

#     # boot fits
#     m_boots = []
#     for Ch_boot, in al.bootstrap_gen(Ch, Nboot=100):
#         Ch_boot = np.mean(Ch_boot, axis=0)
#         res_boot = fit_Ch(t_fit, (Ch_boot[t_fit], Ch_err[t_fit]), L=L, p0=p0)
#         m_boots.append(res_boot['params'][1])
#     m_est = np.mean(m_boots, axis=0), np.std(m_boots, axis=0)
#     shift = get_shift(res['params'][0], res['params'][1], L)
#     al.add_errorbar((Ch_est[0] + shift, Ch_est[1]), xs=ts, ax=ax, marker='o', capsize=2)
#     print('m = ', res['params'][1])
#     ax.plot(ts, res['fopt'](ts) + shift, color='r', zorder=3)
#     print(m_est)
#     return m_est

# def load_and_fit(fname, *, L, ax):
#     ts = np.arange(L)
#     Ch_mom = np.fromfile(fname).reshape(-1, L) / L**2
#     Ch = compute_Ch(Ch_mom)
#     Ch_mean, Ch_err = Ch_est = al.bootstrap(Ch, Nboot=1000, f=al.rmean)

#     # fit window
#     t_fit = ts[4:-4]

#     # main fit
#     res = fit_Ch(t_fit, (Ch_mean[t_fit], Ch_err[t_fit]), L=L, verbose=True)
#     p0 = res['params']
#     print(f'{p0=}')

#     # boot fits
#     m_boots = []
#     for Ch_boot, in al.bootstrap_gen(Ch, Nboot=100):
#         Ch_boot = np.mean(Ch_boot, axis=0)
#         res_boot = fit_Ch(t_fit, (Ch_boot[t_fit], Ch_err[t_fit]), L=L, p0=p0)
#         m_boots.append(res_boot['params'][1])
#     m_est = np.mean(m_boots, axis=0), np.std(m_boots, axis=0)
#     shift = get_shift(res['params'][0], res['params'][1], L)
#     al.add_errorbar((Ch_est[0] + shift, Ch_est[1]), xs=ts, ax=ax, marker='o', capsize=2)
#     print('m = ', res['params'][1])
#     ax.plot(ts, res['fopt'](ts) + shift, color='r', zorder=3)
#     print(m_est)
#     return m_est

# def main_old():
#     L = 64

#     e2s = np.arange(0.30, 1.00+1e-6, 0.10)
#     fig, ax = plt.subplots(1,1)
#     # ax.set_yscale('log')
#     all_m_est = []
#     for e2 in e2s:
#         m_est = load_and_fit(f'raw_obs/obs_trace_{e2:0.2f}_L{L}_cluster_Ch_mom.dat', L=L, ax=ax)
#         all_m_est.append(m_est)
#     all_m_est = np.transpose(all_m_est)
#     fig, ax = plt.subplots(1,1)
#     al.add_errorbar(all_m_est, xs=e2s, ax=ax, marker='o')
#     plt.show()

def compute_shifted_C(Ch_mom, hsq, *, m2, L, max_ds):
    assert Ch_mom.shape[0] == hsq.shape[0]
    V = L**3
    h = np.sum(Ch_mom, axis=-1)
    sbar = np.round(-h / V).astype(int)
    print('sbar', sbar)

    num = 0
    den = 0
    all_ds = np.arange(-max_ds, max_ds+1e-6, 0.5)
    # for numerical stability, divide out a constant
    min_hs_sq = None
    for ds in all_ds:
        s = sbar + ds
        hs_sq = (hsq + 2*s*h + s**2 * V)
        if min_hs_sq is None:
            min_hs_sq = np.min(hs_sq)
        else:
            min_hs_sq = min(min_hs_sq, np.min(hs_sq))
    print(f'{min_hs_sq=}')
    for ds in all_ds:
        s = sbar + ds
        hs_sq = (hsq + 2*s*h + s**2 * V)
        assert np.all(hs_sq >= 0.0), hs_sq
        fs = np.exp(-m2/2 * (hs_sq - min_hs_sq))
        print(fs[0])
        O = np.array([
            np.mean(Ch_mom * np.roll(Ch_mom, -t, axis=-1), axis=-1)
            for t in range(L)])
        Os = O + 2*s*L**2*(h/L) + s**2 * L**4
        num = num + np.transpose(Os*fs)
        den = den + fs

    C_est = al.bootstrap(num, den, Nboot=1000, f=lambda num, den: al.rmean(num) / al.rmean(den))
    return C_est

def main():
    L = 64
    e2s = np.arange(0.30, 0.70+1e-6, 0.10)
    fig, ax = plt.subplots(1,1, figsize=(14,8))
    markers = ['o', 's', '^']
    # colors = ['xkcd:brick red', 'xkcd:royal blue', 'xkcd:forest green',
    #           'xkcd:purple', 'xkcd:orange', 'xkcd:emerald green', 'xkcd:light red',
    #           'xkcd:magenta']*5
    colors = plt.get_cmap('RdBu')(np.arange(len(e2s)) / len(e2s))
    for e2,color in zip(e2s, colors):
        Ch_mom = np.fromfile(f'raw_obs/obs_trace_{e2:0.2f}_L{L}_cluster_Ch_mom.dat').reshape(-1, L)
        hsq = np.fromfile(f'raw_obs/obs_trace_{e2:0.2f}_L{L}_cluster_hsq.dat')
        max_ds = 60
        m2s = [3e-2, 1e-2]
        for m2,marker in zip(m2s, markers):
            C_est = compute_shifted_C(Ch_mom, hsq, m2=m2, L=L, max_ds=max_ds)
            avg_C0 = C_est[0][0]
            C_est = (C_est[0] / avg_C0, C_est[1] / avg_C0)
            al.add_errorbar(C_est, ax=ax, marker=marker, color=color, label=f'e2={e2:.02f} (m2={m2},max_ds={max_ds})')
    ax.legend()
    fig.savefig(f'figs/tmp_Ch_L{L}.pdf')
    plt.show()

def compute_C(Ch_mom):
    L = Ch_mom.shape[-1]
    O = np.array([
        np.mean(np.conj(Ch_mom) * np.roll(Ch_mom, -t, axis=-1), axis=-1)
        for t in range(L)])
    return O

def compute_shiftavg_C(Ch_mom, hsq, *, L):
    assert Ch_mom.shape[0] == hsq.shape[0]
    V = L**3
    h = np.sum(Ch_mom, axis=-1)
    hbar = h / V

    O = compute_C(Ch_mom)
    Osub = -2*L*V * hbar**2 + L**4 * hsq / V + O
    Osub = np.transpose(Osub)

    return Osub

# def estimate_const(C_est, ts):
#     Cs = C_est[0][ts] / C_est[0][0]
#     f = lambda t,A,B,m: A + B*np.exp(-m*t)
#     popt, pcov = sp.optimize.curve_fit(
#         f, ts, Cs, maxfev=10000,
#         p0=[Cs[-1], 0.0, 0.0]
#     )
#     fopt = lambda t: f(t, *popt)
#     resid = fopt(ts) - Cs
#     chisq = np.sum(resid**2)
#     chisq_per_dof = chisq / (len(Cs) - len(popt))
#     print(f'{popt=} {chisq=} {chisq_per_dof=}')
#     return popt[0] * C_est[0][0]

def do_basic_exp_fit(C_est):
    mean, err = C_est
    ts = np.arange(len(mean))
    fit_ts = ts[3:16]
    f = lambda t,A,m: A*np.exp(-m*t)
    fit_mean, fit_err = mean[fit_ts], err[fit_ts]
    popt, pcov = sp.optimize.curve_fit(f, fit_ts, fit_mean, sigma=fit_err)
    fopt = lambda t: f(t, *popt)
    resid = fopt(fit_ts) - fit_mean
    chisq = np.sum(resid**2 / fit_err**2)
    chisq_per_dof = chisq / (len(fit_mean) - len(popt))
    # print(f'{chisq=} {chisq_per_dof=}')
    return {
        'params': popt,
        'f': fopt,
    }

# s = 10
# hbar = np.sum(Ch_mom, axis=-1) / L**3
# shift_Ch_mom = Ch_mom + L**2 * s
# shift_hsq = hsq + 2*s*L**3*hbar + s**2 * L**3
# shift_C = compute_shiftavg_C(shift_Ch_mom, shift_hsq, L=L)
# print(shift_C[0], C[0])
# assert np.allclose(shift_C, C), 'not shift invariant!'

def load_and_fit_periodic(L, e2s):
    fig, ax = plt.subplots(1,1, figsize=(14,8))
    colors = plt.get_cmap('RdBu')(np.arange(len(e2s)) / len(e2s))
    masses = []
    for e2,color in zip(e2s, colors):
        # periodic BCs
        Ch_mom = np.fromfile(
            f'raw_obs/obs_trace_{e2:0.2f}_L{L}_cluster_Ch_mom.dat',
            dtype=np.complex128).reshape(-1, L)
        hsq = np.fromfile(f'raw_obs/obs_trace_{e2:0.2f}_L{L}_cluster_hsq.dat')
        C = compute_shiftavg_C(Ch_mom, hsq, L=L)
        C_est = al.bootstrap(C, Nboot=1000, f=al.rmean)

        avg_C0 = C_est[0][0]
        C_est = (C_est[0] / avg_C0, C_est[1] / avg_C0)
        C_sub = C_est[0][L//2]
        C_est = (C_est[0] - C_sub, C_est[1])

        m_boots = []
        for C_boot, in al.bootstrap_gen(al.bin_data(C, binsize=20)[1], Nboot=1000):
            C_boot = al.rmean(C_boot) / avg_C0
            C_boot -= C_boot[L//2]
            res = do_basic_exp_fit((C_boot, C_est[1]))
            m_boots.append(res["params"][1])
        res = do_basic_exp_fit(C_est)
        m = res["params"][1]
        fit_f = res['f']
        masses.append((np.mean(m_boots), np.std(m_boots)))
        print(f'm={masses[-1]}')

        al.add_errorbar(C_est, ax=ax, marker='o', color=color, label=f'e2={e2:.02f}')
        ts = np.arange(L)
        ax.plot(ts, fit_f(ts), color=color)
    ax.set_yscale('log')
    ax.legend()
    fig.savefig(f'figs/tmp_Ch_sub_L{L}.pdf')

    masses = np.transpose(masses)
    return masses

def load_and_fit_cperiodic(L, e2s):
    fig, ax = plt.subplots(1,1, figsize=(14,8))
    colors = plt.get_cmap('RdBu')(np.arange(len(e2s)) / len(e2s))
    masses = []
    for e2,color in zip(e2s, colors):
        # periodic BCs
        Ch_mom1 = np.fromfile(
            f'raw_obs/obs_trace_{e2:0.2f}_L{L}_cluster_cper_Ch_mom1.dat',
            dtype=np.complex128).reshape(-1, L)
        C = compute_C(Ch_mom1)
        C_est = al.bootstrap(C, Nboot=1000, f=al.rmean)

        avg_C0 = C_est[0][0]
        C_est = (C_est[0] / avg_C0, C_est[1] / avg_C0)
        # C_sub = C_est[0][L//2]
        # C_est = (C_est[0] - C_sub, C_est[1])

        m_boots = []
        for C_boot, in al.bootstrap_gen(al.bin_data(C, binsize=20)[1], Nboot=1000):
            C_boot = al.rmean(C_boot) / avg_C0
            # C_boot -= C_boot[L//2]
            res = do_basic_exp_fit((C_boot, C_est[1]))
            m_boots.append(res["params"][1])
        res = do_basic_exp_fit(C_est)
        m = res["params"][1]
        fit_f = res['f']
        masses.append((np.mean(m_boots), np.std(m_boots)))
        print(f'm={masses[-1]}')

        al.add_errorbar(C_est, ax=ax, marker='o', color=color, label=f'e2={e2:.02f}')
        ts = np.arange(L)
        ax.plot(ts, fit_f(ts), color=color)
    ax.set_yscale('log')
    ax.legend()
    fig.savefig(f'figs/tmp_Ch_cper_L{L}.pdf')

    masses = np.transpose(masses)
    return masses
    

def main2():
    e2s = np.arange(0.70, 2.00+1e-6, 0.10)
    mass_fig, mass_ax = plt.subplots(1,1)
    colors = ['xkcd:brick red', 'xkcd:light red', 'xkcd:magenta']
    for L,color in zip([64, 96, 128], colors):
        style = dict(markersize=6, fillstyle='none', color=color)
        masses = load_and_fit_periodic(L, e2s)
        al.add_errorbar(masses, xs=e2s, ax=mass_ax, marker='o', **style, label=f'L={L}')
        try:
            masses = load_and_fit_cperiodic(L, e2s)
            al.add_errorbar(masses, xs=e2s, ax=mass_ax, marker='x', **style, label=f'L={L} (cper)')
        except FileNotFoundError as e:
            print(f'Skipping cper L={L} ...', repr(e))
            
    mass_ax.set_xlabel('e^2')
    mass_ax.set_ylabel('m')
    mass_ax.legend()
    mass_fig.savefig(f'figs/tmp_Ch_sub_mass.pdf')
    
    plt.show()

if __name__ == '__main__': main2()
