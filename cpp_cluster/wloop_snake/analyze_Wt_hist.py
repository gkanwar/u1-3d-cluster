import analysis as al
import matplotlib.pyplot as plt
import numpy as np
import paper_plt
paper_plt.load_latex_config()
import scipy as sp
import scipy.optimize
import sys

SIGMA_BIAS = 0.05

Nboot = 1000
binsize = 10
fit_pad = 6

def fit_sigma(ts, Wt_est):
    f = lambda ts,A,B: A*np.exp(-B*ts)
    popt, pcov = sp.optimize.curve_fit(f, ts, Wt_est[0][ts], sigma=Wt_est[1][ts])
    return {
        'f': lambda ts: f(ts, *popt),
        'popt': popt
    }

def est_sigma(Wt_hist_cfgs):
    Wt = np.stack((al.rmean(Wt_hist_cfgs), Wt_est[1]))
    res = fit_sigma(np.arange(fit_pad, Lt+1-fit_pad), Wt)
    return res['popt'][1]

def sigma_eff(Wt_hist_cfgs):
    Wt_hist = al.rmean(Wt_hist_cfgs)
    return -np.log(Wt_hist[1:]) + np.log(Wt_hist[:-1])

Lt = int(sys.argv[1])
prefix = sys.argv[2]

fig, ax = plt.subplots(1,1)
ts = np.arange(Lt+1)
x_styles = {
    # 3: 'tab:brown',
    # 4: 'tab:green',
    # 5: 'xkcd:cyan',
    6: 'tab:blue',
    7: 'tab:orange',
    8: 'tab:red',
    9: 'tab:purple',
    10: 'xkcd:cyan',
    11: 'xkcd:emerald green'
}
style = dict(marker='o', markersize=4, fillstyle='none', linestyle='')
for i,(x,color) in enumerate(x_styles.items()):
    fname = f'{prefix}_x{x}_Wt_hist.dat'
    
    Wt_hist = np.fromfile(fname, dtype=np.float64).reshape(-1, Lt+1)
    counts = np.sum(Wt_hist, axis=-1)
    assert np.all(counts == counts[0])
    Wt_hist = Wt_hist * np.exp(-SIGMA_BIAS * np.arange(Lt+1))
    Wt_hist_bin = al.bin_data(Wt_hist, binsize=binsize)[1]
    Wt_est = al.bootstrap(Wt_hist_bin, Nboot=Nboot, f=al.rmean)
    print(f'{Wt_est=}')
    sigma_eff_est = al.bootstrap(Wt_hist_bin, Nboot=Nboot, f=sigma_eff)
    print(f'{sigma_eff_est=}')

    res = fit_sigma(np.arange(fit_pad, Lt+1-fit_pad), Wt_est)
    print(f'{res["popt"]=}')

    sigma = al.bootstrap(al.bin_data(Wt_hist, binsize=binsize)[1], Nboot=Nboot, f=est_sigma)
    print(f'{sigma=}')

    # al.add_errorbar(Wt_est, xs=ts, ax=ax, marker='s', label=f'x={x}', color=color)
    # ax.plot(ts, res['f'](ts), color=color, linestyle='-', marker='')

    al.add_errorbar(
        sigma_eff_est, xs=ts[1:], ax=ax, off=0.03*i, label=f'x={x}', color=color, **style)
    ax.fill_between(
        [-1+i, Lt+1+i], [sigma[0]+sigma[1]]*2, [sigma[0]-sigma[1]]*2,
        color=color, alpha=0.3
    )

ax.legend()
# ax.set_yscale('log')
fig.savefig('tmp.pdf')
plt.close(fig)
