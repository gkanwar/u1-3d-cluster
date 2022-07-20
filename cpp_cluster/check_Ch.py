import analysis as al
import matplotlib as mpl
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import paper_plt
paper_plt.load_latex_config()
import scipy as sp
import scipy.optimize

Nboot=100

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
#     Ch_mean, Ch_err = Ch_est = al.bootstrap(Ch, Nboot=Nboot, f=al.rmean)

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
#     Ch_mean, Ch_err = Ch_est = al.bootstrap(Ch, Nboot=Nboot, f=al.rmean)

#     # fit window
#     t_fit = ts[4:-4]

#     # main fit
#     res = fit_Ch(t_fit, (Ch_mean[t_fit], Ch_err[t_fit]), L=L, verbose=True)
#     p0 = res['params']
#     print(f'{p0=}')

#     # boot fits
#     m_boots = []
#     for Ch_boot, in al.bootstrap_gen(Ch, Nboot=Nboot):
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

    C_est = al.bootstrap(num, den, Nboot=Nboot, f=lambda num, den: al.rmean(num) / al.rmean(den))
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
            al.add_errorbar(C_est, ax=ax, marker=marker, color=color, label=f'$e^2={e2:.02f}$ ($m^2={m2}$,maxds={max_ds})')
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
    # V1
    Osub_v1 = -2*L*V * hbar**2 + L**4 * hsq / V + O
    Osub_v1 = np.transpose(Osub_v1)
    # V2 (see Evertz paper)
    Osub_v2 = -L**4 * hsq / V + O
    Osub_v2 = np.transpose(Osub_v2)

    return Osub_v1, Osub_v2

def compute_Cl(Cl_mom):
    L = Cl_mom.shape[-1]
    Cl_mom_pospar = Cl_mom[:,0] + Cl_mom[:,1]
    Cl_mom_negpar = Cl_mom[:,0] - Cl_mom[:,1]
    O_pospar = np.transpose([
        np.mean(Cl_mom_pospar * np.roll(Cl_mom_pospar, -t, axis=-1), axis=-1)
        for t in range(L)
    ])
    O_negpar = np.transpose([
        np.mean(Cl_mom_negpar * np.roll(Cl_mom_negpar, -t, axis=-1), axis=-1)
        for t in range(L)
    ])
    return O_pospar, O_negpar

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
    popt, pcov = sp.optimize.curve_fit(f, fit_ts, fit_mean, sigma=fit_err, maxfev=10000)
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
    fig2, ax2 = plt.subplots(1,1, figsize=(14,8))
    colors = plt.get_cmap('RdBu')(np.arange(len(e2s)) / len(e2s))
    masses = []
    masses2 = []
    masses_l = []
    e2s_l = []
    print(f'== {L=} ==')
    for e2,color in zip(e2s, colors):
        # load data
        Cl_mom = None
        try:
            Cl_mom = np.fromfile(
                f'raw_obs/obs_trace_{e2:0.2f}_L{L}_cluster_Cl_mom.dat',
                dtype=np.float64).reshape(-1, 2, L)
        except FileNotFoundError as e:
            print(f'Skipping Cl ... {repr(e)}')
        Ch_mom = np.fromfile(
            f'raw_obs/obs_trace_{e2:0.2f}_L{L}_cluster_Ch_mom.dat',
            dtype=np.complex128).reshape(-1, L)
        hsq = np.fromfile(f'raw_obs/obs_trace_{e2:0.2f}_L{L}_cluster_hsq.dat')

        # compute C
        C, C2 = compute_shiftavg_C(Ch_mom, hsq, L=L)
        C_est = al.bootstrap(C, Nboot=Nboot, f=al.rmean)
        avg_C0 = np.abs(C_est[0][0])
        C_est = (C_est[0] / avg_C0, C_est[1] / avg_C0)
        C_sub = C_est[0][L//2]
        C_est = (C_est[0] - C_sub, C_est[1])
        C2_est = al.bootstrap(C, Nboot=Nboot, f=al.rmean)
        avg_C20 = np.abs(C2_est[0][0])
        C2_est = (C2_est[0] / avg_C20, C2_est[1] / avg_C20)
        C2_sub = C2_est[0][L//2]
        C2_est = (C2_est[0] - C2_sub, C2_est[1])

        # estimate C masses
        m_boots = []
        for C_boot, in al.bootstrap_gen(al.bin_data(C, binsize=20)[1], Nboot=Nboot):
            C_boot = al.rmean(C_boot) / avg_C0
            C_boot -= C_boot[L//2]
            res = do_basic_exp_fit((C_boot, C_est[1]))
            m_boots.append(res["params"][1])
        res = do_basic_exp_fit(C_est)
        m = res["params"][1]
        fit_f = res['f']
        masses.append((np.mean(m_boots), np.std(m_boots)))
        print(f'm={masses[-1]}')        
        m2_boots = []
        for C2_boot, in al.bootstrap_gen(al.bin_data(C2, binsize=20)[1], Nboot=Nboot):
            C2_boot = al.rmean(C2_boot) / avg_C20
            C2_boot -= C2_boot[L//2]
            res = do_basic_exp_fit((C2_boot, C2_est[1]))
            m2_boots.append(res["params"][1])
        res = do_basic_exp_fit(C2_est)
        m2 = res["params"][1]
        fit_f2 = res['f']
        masses2.append((np.mean(m2_boots), np.std(m2_boots)))
        print(f'm2={masses2[-1]}')        

        # plot C and fit
        al.add_errorbar(C_est, ax=ax, marker='o', color=color, label=f'$e^2={e2:.02f}$')
        al.add_errorbar(C2_est, ax=ax, marker='s', color=color)
        ts = np.arange(L)
        ax.plot(ts, fit_f(ts), color=color)
        ax.plot(ts, fit_f2(ts), color=color, linestyle='--')

        # skip Cl if not available
        if Cl_mom is None: continue
        e2s_l.append(e2)
        
        # compute Cl
        Cl_plus, Cl_minus = compute_Cl(Cl_mom)
        Cl = -Cl_minus
        Cl_est = al.bootstrap(Cl, Nboot=Nboot, f=al.rmean)
        avg_Cl0 = np.abs(Cl_est[0][1])
        Cl_est = (Cl_est[0] / avg_Cl0, Cl_est[1] / avg_Cl0)
        # Cl_sub = Cl_est[0][L//2]
        # Cl_est = (Cl_est[0] - Cl_sub, Cl_est[1])

        # estimate Cl masses
        ml_boots = []
        for Cl_boot, in al.bootstrap_gen(al.bin_data(Cl, binsize=20)[1], Nboot=Nboot):
            Cl_boot = al.rmean(Cl_boot) / avg_C0
            # Cl_boot -= Cl_boot[L//2]
            res = do_basic_exp_fit((Cl_boot, Cl_est[1]))
            ml_boots.append(res["params"][1])
        res = do_basic_exp_fit(Cl_est)
        ml = res["params"][1]
        fit_fl = res['f']
        masses_l.append((np.mean(ml_boots), np.std(ml_boots)))
        print(f'ml={masses_l[-1]}')

        # plot Cl and fit
        al.add_errorbar(Cl_est, ax=ax2, marker='o', color=color, label=f'$e^2={e2:.02f}$')
        ts = np.arange(L)
        ax2.plot(ts, fit_fl(ts), color=color)
    ax.set_yscale('log')
    ax.legend()
    ax2.set_yscale('log')
    ax2.legend()
    fig.savefig(f'figs/tmp_Ch_sub_L{L}.pdf')
    fig2.savefig(f'figs/tmp_Cl_sub_L{L}.pdf')

    masses = np.transpose(masses)
    masses2 = np.transpose(masses2)
    masses_l = np.transpose(masses_l) if len(masses_l) > 0 else None
    return {
        'masses': (e2s, masses),
        'masses2': (e2s, masses2),
        'masses_l': (e2s_l, masses_l)
    }

def load_and_fit_cperiodic(L, e2s):
    fig, ax = plt.subplots(1,1, figsize=(14,8))
    colors = plt.get_cmap('RdBu')(np.arange(len(e2s)) / len(e2s))
    masses = []
    for e2,color in zip(e2s, colors):
        # C-periodic BCs
        Ch_mom1 = np.fromfile(
            f'raw_obs/obs_trace_{e2:0.2f}_L{L}_cluster_cper_Ch_mom1.dat',
            dtype=np.complex128).reshape(-1, L)
        C = compute_C(Ch_mom1)
        C_est = al.bootstrap(C, Nboot=Nboot, f=al.rmean)

        avg_C0 = np.abs(C_est[0][0])
        C_est = (C_est[0] / avg_C0, C_est[1] / avg_C0)
        # C_sub = C_est[0][L//2]
        # C_est = (C_est[0] - C_sub, C_est[1])

        m_boots = []
        for C_boot, in al.bootstrap_gen(al.bin_data(C, binsize=20)[1], Nboot=Nboot):
            C_boot = al.rmean(C_boot) / avg_C0
            # C_boot -= C_boot[L//2]
            res = do_basic_exp_fit((C_boot, C_est[1]))
            m_boots.append(res["params"][1])
        res = do_basic_exp_fit(C_est)
        m = res["params"][1]
        fit_f = res['f']
        masses.append((np.mean(m_boots), np.std(m_boots)))
        print(f'm={masses[-1]}')

        al.add_errorbar(C_est, ax=ax, marker='o', color=color, label=f'$e^2={e2:.02f}$')
        ts = np.arange(L)
        ax.plot(ts, fit_f(ts), color=color)
    ax.set_yscale('log')
    ax.legend()
    fig.savefig(f'figs/tmp_Ch_cper_L{L}.pdf')

    masses = np.transpose(masses)
    return masses
    

def main2():
    e2s = np.arange(0.70, 2.00+1e-6, 0.10)
    Ls = [64, 96, 128, 192]
    mass_fig, mass_ax = plt.subplots(1,1)
    colors = ['xkcd:gray', 'xkcd:navy blue', 'xkcd:forest green', 'xkcd:red']
    for L,color in zip(Ls, colors):
        style = dict(markersize=6, fillstyle='none', color=color)
        res = load_and_fit_periodic(L, e2s)
        al.add_errorbar(res['masses'][1], xs=e2s, ax=mass_ax, marker='o', **style, label=f'$L={L}$ [h,v1]')
        al.add_errorbar(res['masses2'][1], xs=e2s, ax=mass_ax, marker='^', **style, label=f'$L={L}$ [h,v2]')
        if res['masses_l'][1] is not None:
            al.add_errorbar(
                res['masses_l'][1], xs=res['masses_l'][0],
                ax=mass_ax, marker='s', **style, label=f'$L={L}$ [glue,-]')
        try:
            masses = load_and_fit_cperiodic(L, e2s)
            al.add_errorbar(masses, xs=e2s, ax=mass_ax, marker='x', **style, label=f'$L={L}$ (cper)')
        except FileNotFoundError as e:
            print(f'Skipping cper L={L} ...', repr(e))
            
    mass_ax.set_xlabel('$e^2$')
    mass_ax.set_ylabel('$m$')
    mass_ax.legend()
    mass_fig.savefig(f'figs/tmp_Ch_sub_mass.pdf')
    
    plt.show()

if __name__ == '__main__': main2()
