### Testing MCMC efficiency by various approaches in the following toy problem:
###
### Z = sum_{x_i} exp(-(e^2/2) * [
###     sum_i (x_i + s_i)^2 +
###     sum_<ij> (x_i - x_j)^2
### ])
###
### Suppose the Wilson "loop" spans the range of spins [0,t). Then a direct
### Gaussian calculation of the path integral yields,
###
### <W> = exp(-(e^2/2)*(4 |c|^2 - 2 sum_i c_{i+1} c_i - 2 sum_{i<t} c_i))
### c_i = K^{-1}_{ij} omega_j
### omega_j = {1 if j < t else 0}
### K^{-1}_{ij} = sum_{k} e^{(2 pi /L) i k (i-j)} / (6 - 4 cos(2 pi k / L))

from dataclasses import dataclass
import analysis as al
import matplotlib.pyplot as plt
import numpy as np
import tqdm

@dataclass
class Wloop:
    t: int

Nboot = 1000

# METAPARAMS
DX = 2
DW = 1
BIAS = 0.0

def compute_mean_x(wloop, L):
    omega = np.zeros(L)
    omega[:wloop.t] = 1.0
    omega_tilde = np.fft.fft(omega)
    ks = np.arange(L)
    K = 6 - 4*np.cos(2*np.pi*ks / L)
    Ki = 1/K
    c_tilde = 2 * Ki * omega_tilde
    c = np.fft.ifft(c_tilde)
    assert np.allclose(np.imag(c), 0.0)
    c = np.real(c)
    return c

def exact_W(wloop, L, *, e2):
    c = -compute_mean_x(wloop, L)
    print(f'{wloop.t=} {c=}')
    V = 3*np.sum(c**2) - 2*np.sum(np.roll(c, -1)*c) + 2*np.sum(c[:wloop.t]) + wloop.t
    return np.exp(-(e2/2)*V)

def update_x(x, wloop, *, e2):
    L, = x.shape
    acc = 0
    for i in range(L):
        S1 = (x[i] - x[(i-1)%L])**2 + (x[i] - x[(i+1)%L])**2
        S2 = (x[i]+1)**2 if i < wloop.t else x[i]**2
        old_S = (e2/2)*(S1 + S2)
        sign = 2*np.random.randint(2)-1
        dx = np.random.randint(1,DX+1)
        xi_p = x[i] + sign*dx
        S1_p = (xi_p - x[(i-1)%L])**2 + (xi_p - x[(i+1)%L])**2
        S2_p = (xi_p+1)**2 if i < wloop.t else xi_p**2
        new_S = (e2/2)*(S1_p + S2_p)
        if np.random.random() < np.exp(-new_S + old_S):
            x[i] = xi_p
            acc += 1.0/L
    return acc

### NOTE: Not clear there is a nice way to sample these Gaussian *integers*
def update_x_v2(x, wloop, *, e2):
    L, = x.shape
    phi = np.random.normal(size=L)
    ks = np.arange(L)
    phi /= np.sqrt(e2*(3 - 2*np.cos(2*np.pi*ks/L)))
    assert L % 2 == 0, 'specialized to even L'
    # print(ks[L//2+1:], ks[L//2-1:0:-1])
    phi[L//2+1:] = phi[L//2-1:0:-1]
    xp = np.fft.ifft(phi, norm='ortho')
    # print(xp)
    assert np.allclose(np.imag(xp), 0.0)
    xp = np.real(xp)
    mean_x = compute_mean_x(wloop, L)
    x_f = xp + mean_x
    print(x_f)
    assert np.allclose(x_f % 1.0, 0.0) # FAILS
    x[:] = np.rint(x_f).astype(int)
    return 1.0

def update_W(x, wloop, *, e2):
    L, = x.shape
    sign = 2*np.random.randint(2)-1
    dW = sign*np.random.randint(1,DW+1)

    if wloop.t + dW < 0 or wloop.t + dW > L:
        return False

    if sign > 0:
        xts = slice(wloop.t,wloop.t+dW)
    else:
        xts = slice(wloop.t+dW,wloop.t)
    dS = sign*((e2/2)*np.abs(dW) + e2*np.sum(x[xts]) - BIAS)
    acc = np.random.random() < np.exp(-dS)
    if acc:
        wloop.t += dW
    return acc

def compute_W_weights(x, wloop, *, e2):
    L, = x.shape
    log_ws = -(e2/2) - e2*x + BIAS
    log_wtots = np.insert(np.cumsum(log_ws), 0, 0)
    wtots = np.exp(log_wtots) / np.sum(np.exp(log_wtots))
    return wtots

def update_W_v2(x, wloop, *, e2):
    wtots = compute_W_weights(x, wloop, e2=e2)
    wloop.t = np.random.choice(np.arange(len(wtots)), p=wtots)
    return True

def update_Wt_hist(x, wloop, Wt_hist_bin, *, e2):
    wtots = compute_W_weights(x, wloop, e2=e2)
    Wt_hist_bin += wtots

def run_metropolis(*, L, e2, n_iter, n_therm, n_bin_meas):
    x = np.zeros(L, dtype=int)
    wloop = Wloop(t=0)
    Wt_hist_bin = np.zeros(L+1, dtype=np.float64)
    Wt_hist = []
    acc_x = 0.0
    acc_W = 0.0
    n_update_x = 1
    for i in tqdm.tqdm(range(-n_therm, n_iter)):
        acc_xi = 0
        for j in range(n_update_x):
            acc_xij = update_x(x, wloop, e2=e2)
            acc_xi += acc_xij / n_update_x
        acc_Wi = update_W_v2(x, wloop, e2=e2)
        acc_x += acc_xi / (n_therm + n_iter)
        acc_W += acc_Wi / (n_therm + n_iter)

        update_Wt_hist(x, wloop, Wt_hist_bin, e2=e2)
        # Wt_hist_bin[wloop.t] += 1
        
        if (i+1) % n_bin_meas == 0:
            Wt_hist.append(Wt_hist_bin)
            Wt_hist_bin = np.zeros(L+1, dtype=np.float64)
    print(f'{acc_x=} {acc_W=}')

    return {
        'Wt_hist': np.array(Wt_hist)
    }


def main():
    L = 16
    e2 = 1.0
    # binsize = 50
    # do_bin = lambda x: al.bin_data(x, binsize=binsize)[1]

    n_iter = 250000
    n_therm = 25000

    fig, ax = plt.subplots(1,1)

    ### EXACT
    W1 = np.array([exact_W(Wloop(t=t), L=L, e2=e2) for t in range(0, L+1)])
    ax.plot(W1, marker='s', label='exact')

    ### V1
    global DW
    DW = 1
    res = run_metropolis(L=L, e2=e2, n_iter=n_iter, n_therm=n_therm, n_bin_meas=1)
    W2 = res['Wt_hist'] * np.exp(-np.arange(L+1)*BIAS)
    W2 *= np.mean(W1)/np.mean(W2)
    binsizes = [1,10,30,100,300,1000]
    for i,binsize in enumerate(binsizes):
        W2_bin = al.bin_data(W2, binsize=binsize)[1]
        W2_est = al.bootstrap(W2_bin, Nboot=Nboot, f=al.rmean)
        al.add_errorbar(
            W2_est, ax=ax, marker='o', label=f'MC 1 (binsize={binsize})',
            linestyle='', elinewidth=0.5, capsize=1.5, off=i*0.05, markersize=2)

    ### V2
    # DW = 8
    # res = run_metropolis(L=L, e2=e2, n_iter=n_iter, n_therm=n_therm, n_bin_meas=100)
    # W3 = res['Wt_hist'] * np.exp(-np.arange(L+1)*BIAS)
    # W3 *= np.mean(W1)/np.mean(W3)
    # W3_est = al.bootstrap(W3, Nboot=Nboot, f=al.rmean)
    # al.add_errorbar(W3_est, ax=ax, marker='o', label='MC 2')

    # np.save('tmp_Wt_hist.npy', res['Wt_hist'])

    ax.set_yscale('log')
    ax.legend()
    fig.savefig('toy_Wt_hist.pdf')
    #plt.show()
    

if __name__ == '__main__': main()
