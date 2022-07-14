### Quick check that masses look like a nice curve

import analysis as al
import matplotlib
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import scipy.optimize

def plot_error_vs_binning(x):
    binsizes = np.array([1, 2, 4, 8, 16, 32, 64])
    ests = []
    for binsize in binsizes:
        ts, xbin = al.bin_data(x, binsize=binsize)
        ests.append(al.bootstrap(xbin, Nboot=1000, f=al.rmean))
    ests = np.transpose(ests)
    print(ests.shape)
    al.add_errorbar(ests, xs=binsizes, ax=plt, marker='o')
    plt.show()

def lighten_color(c):
    rgb = matplotlib.colors.to_rgb(c)
    rgb_light = tuple(min(1.0, 1.5*x) for x in rgb)
    return rgb_light

def load_and_fit(fname, L, ax, style):
    Cl = -np.fromfile(fname).reshape(-1, L)
    Cl /= np.mean(Cl[:,1])
    # plot_error_vs_binning(Cl[:,4])
    
    Cl_boots = list(map(lambda Cl: al.rmean(Cl[0]), al.bootstrap_gen(Cl, Nboot=1000)))
    Cl_covar = al.covar_from_boots(Cl_boots)
    Cl_mean = np.mean(Cl_boots, axis=0)

    f = lambda r,A,B,m: np.exp(-m*r)*(
        2*A/r - B*np.exp(-m)/(r+1) - B*np.exp(m)/(r-1)) # TODO: What is the right fit form?
    rmin, rmax = 4, 15
    all_rs = np.arange(Cl.shape[-1])
    rs = all_rs[rmin:rmax:2]
    ys = Cl_mean[rs]
    covar = Cl_covar[np.ix_(rs, rs)]
    popt, pcov = sp.optimize.curve_fit(
        f, rs, ys, sigma=covar,
        bounds=[[0.0, -np.inf, -np.inf], [np.inf, np.inf, np.inf]])
    resid = f(rs, *popt) - ys
    chisq = resid.T @ np.linalg.inv(covar) @ resid
    chisq_per_dof = chisq / (len(ys) - len(popt))
    print(f'{chisq=}')
    print(f'{chisq_per_dof=}')
    print(f'mass = {popt[2]}')

    prefactor = lambda rs: 1.0
    al.add_errorbar(
        (prefactor(all_rs)*Cl_mean,
         prefactor(all_rs)*np.sqrt(np.einsum('ii->i', Cl_covar))),
        ax=ax, marker='o', linestyle='-', **style)
    fine_rs = np.linspace(1.1, 15.0, num=100)
    style2 = dict(**style)
    style2['color'] = lighten_color(style['color'])
    ax.plot(fine_rs, prefactor(fine_rs)*f(fine_rs, *popt), linestyle='-', zorder=3,
            **style2)
    return popt[2]

def load_and_fit_momproj(fname, L, ax, style):
    l_momproj = np.fromfile(fname).reshape(-1, L)
    Cl = -np.transpose([
        np.mean(l_momproj * np.roll(l_momproj, -t, axis=-1), axis=-1) for t in range(L)
    ])
    
    Cl_boots = list(map(lambda Cl: al.rmean(Cl[0]), al.bootstrap_gen(Cl, Nboot=1000)))
    Cl_covar = al.covar_from_boots(Cl_boots)
    Cl_mean = np.mean(Cl_boots, axis=0)
    print('Cl_mean', Cl_mean)

    f = lambda r,A,m: A*np.exp(-m*r)
    rmin, rmax = 4, 15
    all_rs = np.arange(Cl.shape[-1])
    rs = all_rs[rmin:rmax]
    ys = Cl_mean[rs]
    covar = al.shrink_covar(Cl_covar[np.ix_(rs, rs)], lam=0.1)
    _eigs = np.linalg.eigvals(covar)
    print(np.min(_eigs), np.max(_eigs))
    popt, pcov = sp.optimize.curve_fit(
        f, rs, ys, sigma=covar,
        bounds=[[0.0, 0.0], [np.inf, np.inf]], maxfev=10000)
    resid = f(rs, *popt) - ys
    chisq = resid.T @ np.linalg.inv(covar) @ resid
    chisq_per_dof = chisq / (len(ys) - len(popt))
    print(f'{chisq=}')
    print(f'{chisq_per_dof=}')
    print(f'mass = {popt[1]}')

    prefactor = lambda rs: 1.0
    al.add_errorbar(
        (prefactor(all_rs)*Cl_mean,
         prefactor(all_rs)*np.sqrt(np.einsum('ii->i', Cl_covar))),
        ax=ax, marker='o', linestyle='-', **style)
    fine_rs = np.linspace(1.1, 15.0, num=100)

    style2 = dict(**style)
    style2['color'] = lighten_color(style['color'])
    ax.plot(fine_rs, prefactor(fine_rs)*f(fine_rs, *popt), linestyle='-', zorder=3,
            **style2)
    return popt[1]

e2s = np.arange(0.40, 1.00+1e-6, 0.10)
L = 64
ms = []
fig, ax = plt.subplots(1,1, figsize=(8,8))
colors = ['xkcd:brick red', 'xkcd:purple', 'xkcd:blue', 'xkcd:forest green', 'xkcd:emerald green', 'xkcd:bright red', 'xkcd:orange', 'xkcd:pink']
for e2,color in zip(e2s, colors):
    print(e2)
    # m = load_and_fit(f'raw_obs/obs_trace_{e2:0.2f}_L{L}_cluster_Cl.dat', L, ax, dict(
    #     label=f'{e2:0.2f}_L{L}', color=color
    # ))
    m = load_and_fit_momproj(f'raw_obs/obs_trace_{e2:0.2f}_L{L}_cluster_Cl_mom.dat', L, ax, dict(
        label=f'{e2:0.2f}_L{L}', color=color
    ))
    ms.append(m)
ax.set_yscale('log')
ax.grid()
# ax.set_ylim(-3e-3, 3e-3)
ax.legend()
fig.savefig(f'figs/tmp_Cl_all.pdf')

fig, ax = plt.subplots(1,1)
ax.plot(e2s, ms)
ax.set_yscale('log')
fig.set_tight_layout(True)
fig.savefig('figs/mass_curve.pdf')
# plt.show()
