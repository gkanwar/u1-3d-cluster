import analysis as al
import itertools
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
import pyvista.examples
import scipy as sp
import scipy.interpolate
import scipy.optimize
import scipy.ndimage
import tqdm


def get_checkerboard_mask(p, *, shape):
    arrs = [np.arange(Lx) for Lx in shape]
    return sum(np.ix_(*arrs)) % 2 == p

def meas_wloops(hs, e2s, colors):
    fig, ax = plt.subplots(1,1, figsize=(3.375, 2.5))
    for h,e2,color in zip(hs, e2s, colors):
        m_spacelike_plaqs = np.roll(h, -1, axis=-1) - h
        logweights_spacelike_plaqs = -e2 * ((m_spacelike_plaqs + 1)**2 - m_spacelike_plaqs**2)
        Ws = [np.ones(len(h))]
        As = [0]
        plot_As = [0]
        count = {}
        for Ax in range(1, 6):
            for At in range(1, 6):
                W = np.exp(np.sum(logweights_spacelike_plaqs[:, :Ax, :At], axis=(1,2)))
                W = np.mean(W, axis=-1)
                A = Ax * At
                if A > 10: continue
                Ws.append(W)
                if A not in count:
                    count[A] = 0
                plot_As.append(A + 0.07 * count[A])
                As.append(A)
                count[A] += 1
        Ws = np.transpose(Ws)
        Ws_est = al.bootstrap(Ws, Nboot=100, f=al.rmean)
        # plot data
        al.add_errorbar(
            Ws_est, xs=plot_As, ax=ax, marker='o', fillstyle='none', markersize=4,
            capsize=2, linestyle='none', elinewidth=0.5, color=color,
            label=f'e2={e2:0.2f}')
        # fit and plot fit
        def linear_f(x, m, C):
            return x*m + C
        A0 = 1
        As = np.array(As)
        inds = np.argsort(As)
        As = As[inds]
        log_Ws = np.log(Ws_est[0][inds])
        log_Ws_errs = Ws_est[1][inds] / Ws_est[0][inds]
        popt, _ = sp.optimize.curve_fit(linear_f, As[A0:], log_Ws[A0:], sigma=log_Ws_errs[A0:], p0=[-0.5, 1.0])
        print('popt', popt)
        best_f = lambda x: linear_f(x, *popt)
        interp_As = np.linspace(min(As), max(As))
        ax.plot(interp_As, np.exp(best_f(interp_As)), color=color, linewidth=0.5,
                label=f'fit sigma={-popt[0]}')
    ax.set_yscale('log')
    ax.set_xlabel('A')
    ax.legend(ncol=2)
    fig.set_tight_layout(True)
    fig.savefig('figs/corrs_all_e2.pdf')


def meas_plaqs(hs, e2s, Ls):
    Nd = 3
    L_inds = {}
    for L in Ls:
        if L not in L_inds:
            L_inds[L] = (Ls == L)
    all_Ps = []
    for h,e2 in zip(hs, e2s):
        P_avg = 0
        for i in range(Nd):
            m = np.roll(h, -1, axis=i+1) - h
            P = np.exp(-e2 * ((m+1)**2 - m**2))
            P_avg = P_avg + np.mean(P, axis=(1,2,3)) / Nd
        print(P_avg.shape)
        P_est = al.bootstrap(P_avg, Nboot=100, f=al.rmean)
        all_Ps.append(P_est)
    all_Ps = np.transpose(all_Ps)
    print(all_Ps)
    fig, ax = plt.subplots(1,1, figsize=(3.375, 2.5))
    for L,inds in L_inds.items():
        al.add_errorbar(
            all_Ps[:,inds], xs=e2s[inds], ax=ax,
            label=f'<P> L={L}', marker='o', fillstyle='none', linestyle='',
            linewidth=0.5, capsize=2)
    ax.set_xlabel('e^2')
    ax.legend()
    fig.set_tight_layout(True)
    fig.savefig('figs/plaqs_all_e2.pdf')

def get_bin_midpoints(bins):
    bins = np.array(bins)
    return (bins[1:] + bins[:-1])/2

# find repeated points and collapse into single reweighted point
def combine_scatter_weights(pts, weights):
    assert len(pts) == len(weights)
    tot_weights = {}
    for pt,weight in zip(pts, weights):
        pt = tuple(pt)
        if pt not in tot_weights:
            tot_weights[pt] = 0
        tot_weights[pt] += weight
    print(f'combine reduced from {len(pts)} pts to {len(tot_weights)} pts')
    pts = list(tot_weights.keys())
    weights = [tot_weights[pt] for pt in pts]
    return np.array(pts), np.array(weights)

def meas_Ms(hs, e2s, Ls):
    Nd = 3
    L_inds = {}
    for L in Ls:
        if L not in L_inds:
            L_inds[L] = np.array(list(sorted(
                np.argwhere(Ls == L)[:,0], key=lambda i: e2s[i])))
            
    all_Ms = []
    all_MTs = []
    all_aMs = []
    all_aMTs = []
    all_M2s = []
    all_B4s = []
    bins_MTs = np.linspace(-0.10, 0.10, num=20, endpoint=True)
    binmids_MTs = get_bin_midpoints(bins_MTs)
    bins_Ms = np.linspace(-0.10, 0.10, num=20, endpoint=True)
    binmids_Ms = get_bin_midpoints(bins_Ms)
    hist_Ms = []
    hist_MTs = []
    scatter_Ms = []
    scatter_MTs = []
    for h in tqdm.tqdm(hs):
        mask = get_checkerboard_mask(0, shape=h[0].shape)
        Ms = np.array([
            np.mean(hi[mask]) - np.mean(hi[~mask])
            for hi in h])
        MTs = np.array([
            np.mean((hi[mask] - np.mean(hi[mask]))**2) - np.mean((hi[~mask] - np.mean(hi[~mask]))**2)
            for hi in h])
        M2s = np.array([
            (np.mean(hi[mask]) - np.mean(hi[~mask]))**2
            for hi in h])
        M4s = np.array([
            (np.mean(hi[mask]) - np.mean(hi[~mask]))**4
            for hi in h])
        M_est = al.bootstrap(Ms, Nboot=100, f=al.rmean)
        MT_est = al.bootstrap(MTs, Nboot=100, f=al.rmean)
        aM_est = al.bootstrap(np.abs(Ms), Nboot=100, f=al.rmean)
        aMT_est = al.bootstrap(np.abs(MTs), Nboot=100, f=al.rmean)
        M2_est = al.bootstrap(M2s, Nboot=100, f=al.rmean)
        B4_est = al.bootstrap(M4s, M2s, Nboot=100, f=lambda M4,M2: al.rmean(M4) / al.rmean(M2)**2)
        all_Ms.append(M_est)
        all_MTs.append(MT_est)
        all_aMs.append(aM_est)
        all_aMTs.append(aMT_est)
        all_M2s.append(M2_est)
        all_B4s.append(B4_est)
        scatter_MTs.append(MTs)
        MT_hist, _ = np.histogram(MTs, bins=bins_MTs)
        hist_MTs.append(MT_hist / len(MTs))
        scatter_Ms.append(Ms)
        M_hist, _ = np.histogram(Ms, bins=bins_Ms)
        hist_Ms.append(M_hist / len(Ms))
    all_Ms = np.transpose(all_Ms)
    all_MTs = np.transpose(all_MTs)
    all_aMs = np.transpose(all_aMs)
    all_aMTs = np.transpose(all_aMTs)
    all_M2s = np.transpose(all_M2s)
    all_B4s = np.transpose(all_B4s)
    hist_MTs = np.transpose(hist_MTs)
    hist_Ms = np.transpose(hist_Ms)

    # histograms/scatter plots
    for L in sorted(L_inds.keys()):
        inds = L_inds[L]
        
        fig, axes = plt.subplots(1,2, figsize=(6.5, 3.5))
        ax = axes[0]
        e2s_L = e2s[inds]
        xs, ys = np.meshgrid(e2s_L, binmids_MTs)
        zs = hist_MTs[:,inds] # nbins x ne2s
        f = sp.interpolate.interp2d(e2s_L, binmids_MTs, zs, kind='linear')
        xs_fine = np.linspace(np.min(e2s_L), np.max(e2s_L), num=100, endpoint=True)
        ys_fine = np.linspace(np.min(bins_MTs), np.max(bins_MTs), num=100, endpoint=True)
        zs_fine = f(xs_fine, ys_fine)
        ax.imshow(
            zs_fine, aspect='auto', interpolation='none', origin='lower',
            cmap='binary',
            extent=(np.min(xs_fine), np.max(xs_fine), np.min(ys_fine), np.max(ys_fine))
        )
        ax.set_xlabel('e2')
        ax.set_ylabel('MT')
        # ax = axes[1]
        # ys_zoom = sp.ndimage.zoom(binmids_Ms, bin_zoom)
        # # xs, ys = np.meshgrid(e2s[inds], ys_zoom)
        # zs = hist_Ms[:,inds] # nbins x ne2s
        # # zs = sp.ndimage.zoom(zs, zoom=[bin_zoom, 1.0])
        # zs = sp.ndimage.gaussian_filter(zs, sigma=[0.0, e2_smooth])
        # ax.imshow(
        #     zs, aspect='auto', interpolation='none', origin='lower',
        #     cmap='binary',
        #     extent=(np.min(e2s[inds]), np.max(e2s[inds]), np.min(bins_Ms), np.max(bins_Ms))
        # )
        # ax.set_xlabel('e2')
        # ax.set_ylabel('M')
        fig.set_tight_layout(True)
        fig.savefig(f'figs/Ms_hist_L{L}.pdf')

        # fig, axes = plt.subplots(1,2, figsize=(6.5, 3.5))
        # ax = axes[0]
        # arrs = [np.stack((
        #     np.full(scatter_MTs[i].shape, e2s[i]), scatter_MTs[i],
        #     np.full(scatter_MTs[i].shape, 1/len(scatter_MTs[i])))
        # ) for i in inds]
        # xs, ys, ws = np.concatenate(arrs, axis=-1)
        # ax.hist2d(xs, ys, weights=ws, bins=[40,40], range=[
        #     [np.min(e2s[inds]), np.max(e2s[inds])],
        #     [np.min(bins_MTs), np.max(bins_MTs)]
        # ])
        # ax.set_xlabel('e2')
        # ax.set_ylabel('MT')
        # ax = axes[1]
        # arrs = [np.stack((
        #     np.full(scatter_Ms[i].shape, e2s[i]), scatter_Ms[i],
        #     np.full(scatter_Ms[i].shape, 1/len(scatter_Ms[i])))
        # ) for i in inds]
        # xs, ys, ws = np.concatenate(arrs, axis=-1)
        # ax.hist2d(xs, ys, weights=ws, bins=[40, 40], range=[
        #     [np.min(e2s[inds]), np.max(e2s[inds])],
        #     [np.min(bins_Ms), np.max(bins_Ms)]
        # ])
        # ax.set_xlabel('e2')
        # ax.set_ylabel('M')
        # fig.set_tight_layout(True)
        # fig.savefig(f'figs/Ms_hist2_L{L}.pdf')

        # fig, axes = plt.subplots(1,2, figsize=(6.5, 3.5))
        # ax = axes[0]
        # pt_size = 10.0
        # arrs = [np.stack((
        #     np.full(scatter_MTs[i].shape, e2s[i]), scatter_MTs[i],
        #     np.full(scatter_MTs[i].shape, pt_size/len(scatter_MTs[i])))
        # ) for i in inds]
        # xs, ys, ws = np.concatenate(arrs, axis=-1)
        # pts, ws = combine_scatter_weights(np.transpose((xs, ys)), ws)
        # ax.scatter(*np.transpose(pts), s=ws)
        # ax.set_xlim(np.min(e2s[inds]), np.max(e2s[inds]))
        # ax.set_ylim(np.min(bins_MTs), np.max(bins_MTs))
        # ax.set_xlabel('e2')
        # ax.set_ylabel('MT')
        # ax = axes[1]
        # arrs = [np.stack((
        #     np.full(scatter_MTs[i].shape, e2s[i]), scatter_MTs[i],
        #     np.full(scatter_MTs[i].shape, pt_size/len(scatter_MTs[i])))
        # ) for i in inds]
        # xs, ys, ws = np.concatenate(arrs, axis=-1)
        # pts, ws = combine_scatter_weights(np.transpose((xs, ys)), ws)
        # ax.scatter(*np.transpose(pts), s=ws)
        # ax.set_xlim(np.min(e2s[inds]), np.max(e2s[inds]))
        # ax.set_ylim(np.min(bins_Ms), np.max(bins_Ms))
        # ax.set_xlabel('e2')
        # ax.set_ylabel('M')
        # fig.set_tight_layout(True)
        # fig.savefig(f'figs/Ms_scatter_L{L}.pdf')

    # order parameters plot
    fig, axes = plt.subplots(3,2, figsize=(6.75, 6.5), sharex=True, squeeze=False)
    marker = ''
    linestyle = '-'
    capsize = 0
    for L in sorted(L_inds.keys()):
        inds = L_inds[L]
        al.add_errorbar(
            all_aMs[:,inds], xs=e2s[inds], ax=axes[0,0],
            label=f'<|M|> L={L}', marker=marker, linestyle=linestyle,
            linewidth=0.5, capsize=capsize)
        al.add_errorbar(
            all_Ms[:,inds], xs=e2s[inds], ax=axes[1,0],
            label=f'<M> L={L}', marker=marker, linestyle=linestyle,
            linewidth=0.5, capsize=capsize)
        al.add_errorbar(
            all_M2s[:,inds], xs=e2s[inds], ax=axes[2,0],
            label=f'M2 L={L}', marker=marker, linestyle=linestyle,
            linewidth=0.5, capsize=capsize)
        al.add_errorbar(
            all_aMTs[:,inds], xs=e2s[inds], ax=axes[0,1],
            label=f'<|MT|> L={L}', marker=marker, linestyle=linestyle,
            linewidth=0.5, capsize=capsize)
        al.add_errorbar(
            all_MTs[:,inds], xs=e2s[inds], ax=axes[1,1],
            label=f'<MT> L={L}', marker=marker, linestyle=linestyle,
            linewidth=0.5, capsize=capsize)
        al.add_errorbar(
            all_B4s[:,inds], xs=e2s[inds], ax=axes[2,1],
            label=f'B4 L={L}', marker=marker, linestyle=linestyle,
            linewidth=0.5, capsize=capsize)
    for ax in axes[-1]:
        ax.set_xlabel('e^2')
        ax.set_xlim(-0.01, 0.61)
    for ax in axes.flatten():
        ax.legend(ncol=2, fontsize=7)
        ax.grid()
    axes[0,0].set_ylim(-0.002, 0.04)
    axes[1,0].set_ylim(-0.01, 0.01)
    axes[0,1].set_ylim(-0.01, 0.07)
    axes[1,1].set_ylim(-0.04, 0.04)
    fig.set_tight_layout(True)
    fig.savefig('figs/Ms_all_e2.pdf')

def meas_corrs(h):
    corr = 0
    N = 4
    for x in range(0, h.shape[0], h.shape[0]//N):
        for y in range(0, h.shape[1], h.shape[1]//N):
            for z in range(0, h.shape[2], h.shape[2]//N):
                corr_src = np.roll(h, (-x,-y,-z), axis=(1,2,3)) - h[:,x:x+1,y:y+1,z:z+1]
                corr = corr + corr_src / N**3

    ds_to_corrs = {}
    for dx in range(-4, 4+1):
        for dy in range(-4, 4+1):
            for dz in range(-4, 4+1):
                d2 = dx**2 + dy**2 + dz**2
                if d2 > 16: continue
                dx,dy,dz = sorted([abs(dx),abs(dy),abs(dz)])
                key = dx,dy,dz
                if key not in ds_to_corrs:
                    ds_to_corrs[key] = []
                ds_to_corrs[key].append(corr[:,dx,dy,dz])
    d2s = []
    corr_ests = []
    for (dx,dy,dz),corr in ds_to_corrs.items():
        d2 = dx**2 + dy**2 + dz**2
        corr_est = al.bootstrap(np.exp(1j*corr), Nboot=100, f=al.rmean)
        ### TODO!!!

def energy(h):
    assert len(h.shape) == 3, 'specialized for 3d'
    Nd = 3
    E = 0.0
    for i in range(Nd):
        mp = h - np.roll(h, -1, axis=i)
        E += np.sum(mp**2)
    return E

def run_multipoint_mag(hs, e2s, *, Os, L):
    E_means = []
    E_stds = []
    for h in hs:
        E = np.array([energy(hi) for hi in h])
        E_means.append(np.mean(E))
        E_stds.append(np.std(E))
    # var_i can be estimated by
    # (dg * std[E_i] * Z/Z_i)^2 / N_i
    # best coeffs given by (normalized) 1/var_i
    ### TODO!!!

def measure_D(h):
    Nd = len(h.shape)
    D = 0
    for i in range(Nd):
        D = D + (np.roll(h, -2, axis=i) - h)**2
    mask = get_checkerboard_mask(1, shape=h.shape)
    D[mask] *= -1
    D = np.sign(D)
    # smear over one lattice site
    # comps = []
    # for i in range(Nd):
    #     comps.append((np.roll(D, -1, axis=i) + D + np.roll(D, 1, axis=i)) / 3)
    # D = np.mean(comps, axis=0)
    # smear over one cube
    comps = []
    for dxs in itertools.product(range(2), repeat=Nd):
        comp = D
        for i in range(Nd):
            comp = np.roll(comp, -dxs[i], axis=i)
        comps.append(comp)
    D = np.mean(comps, axis=0)
    return D

def plot_domains(h, *, e2, prefix):
    assert len(h.shape) == 3, 'specialized for 3d'
    L = h.shape[0]
    D = measure_D(h)
    # 2D option
    fig, axes = plt.subplots(2,2, figsize=(6,6))
    for i in range(4):
        ax = axes.flatten()[i]
        ax.matshow(D[i*L//4], cmap='bwr', vmin=-1.0, vmax=1.0)
        ax.set_title(f'slice {i*L//4}')
    fig.set_tight_layout(True)
    fig.savefig(prefix + '.pdf')
    plt.close(fig)
    # 3D option
    grid = pv.UniformGrid()
    grid.dimensions = np.array(D.shape) + 1
    grid.cell_data['values'] = D.flatten(order='F')
    grid.plot(
        cmap='bwr', clim=[-1,1], show_edges=False,
        cpos=[(2.5*L, 2.5*L, 1.5*L), (0,0,0), (0,0,1)],
        text=f'e2={e2:0.2f}, L={L}',
        off_screen=True, screenshot=prefix + '.jpg')

def main_mag():
    e2s = []
    Ls = []
    hs = []

    print('Loading all data...')

    # L = 8
    sweep_e2s = np.concatenate((
        np.arange(0.25, 0.55+1e-6, 0.01),
        np.arange(0.60, 1.50+1e-6, 0.1)))
    e2s.extend(sweep_e2s)
    Ls.extend([8]*len(sweep_e2s))
    hs.extend([
        np.load(f'ens_{e2:0.2f}_L8_mixed.npy')/2 for e2 in sweep_e2s])
    """

    # L = 16
    sweep_e2s = np.concatenate((
        np.arange(0.25, 0.55+1e-6, 0.01),
        np.arange(0.60, 1.50+1e-6, 0.1)))
    e2s.extend(sweep_e2s)
    Ls.extend([16]*len(sweep_e2s))
    hs.extend([
        np.load(f'ens_{e2:0.2f}_L16_mixed.npy')/2 for e2 in sweep_e2s])

    # L = 32
    sweep_e2s = np.concatenate((
        np.arange(0.25, 0.55+1e-6, 0.01),
        np.arange(0.60, 1.50+1e-6, 0.1)))
    e2s.extend(sweep_e2s)
    Ls.extend([32]*len(sweep_e2s))
    hs.extend([
        np.load(f'ens_{e2:0.2f}_L32_mixed.npy')/2 for e2 in sweep_e2s])

    # L = 64
    sweep_e2s = np.concatenate((
        np.arange(0.25, 0.55+1e-6, 0.01),
        np.arange(0.60, 1.50+1e-6, 0.1)))
    e2s.extend(sweep_e2s)
    Ls.extend([64]*len(sweep_e2s))
    hs.extend([
        np.load(f'ens_{e2:0.2f}_L64_mixed.npy')/2 for e2 in sweep_e2s])
    """

    """
    # L = 96
    sweep_e2s = np.array([x for x in np.arange(0.25, 0.50+1e-6, 0.05) if x not in sweep_e2s])
    e2s.extend(sweep_e2s)
    Ls.extend([96]*len(sweep_e2s))
    hs.extend([
        np.load(f'ens_{e2:0.2f}_L96_mixed.npy')/2 for e2 in sweep_e2s])
    """

    print('Running measurements...')

    e2s = np.array(e2s)
    Ls = np.array(Ls)
    # meas_plaqs(hs, e2s, Ls)
    meas_Ms(hs, e2s, Ls)

    ### OLD data
    # sweep_e2s = np.arange(0.6, 3.0+1e-6, 0.1)
    # e2s.extend(sweep_e2s)
    # Ls.extend([8]*len(sweep_e2s))
    # hs.extend([
    #     np.load(f'ens_{e2:0.2f}_clust.npy')/2 for e2 in sweep_e2s])
    # sweep_e2s = np.array([0.15, 0.20, 0.60, 0.65, 0.70]) #np.arange(0.15, 0.7+1e-6, 0.05)
    # e2s.extend(sweep_e2s)
    # Ls.extend([64]*len(sweep_e2s))
    # hs.extend([
    #     np.load(f'ens_{e2:0.2f}_L64_clust.npy')/2 for e2 in sweep_e2s])
    # sweep_e2s = np.arange(0.6, 3.0+1e-6, 0.1)
    # e2s.extend(sweep_e2s)
    # Ls.extend([16]*len(sweep_e2s))
    # hs.extend([
    #     np.load(f'ens_{e2:0.2f}_L16_clust.npy')/2 for e2 in sweep_e2s])
    
def main_wloops():
    e2s = [0.7, 0.6, 0.5, 0.4]
    Ls = [8]*4
    hs = [
        np.load(f'ens_{e2:0.2f}_clust.npy')/2 for e2 in [0.7, 0.6, 0.5, 0.4]]

    hs.append(np.load('ens_b0.20_L32.npy') / 2)
    e2s.append(0.2)
    Ls.append(32)
    hs.append(np.load('ens_b0.10_L64.npy') / 2)
    e2s.append(0.1)
    Ls.append(64)

    colors = ['forestgreen', 'lightseagreen', 'dodgerblue', 'mediumblue', 'orangered', 'deeppink']
    meas_wloops(hs, e2s, colors)


def main_domains():
    for e2 in tqdm.tqdm(np.arange(0.25, 1.50+1e-6, 0.05)):
        for L in [16, 32, 64, 96]:
            try:
                h = np.load(f'ens_{e2:0.2f}_L{L}_mixed.npy')/2
                plot_domains(h[-1], e2=e2, prefix=f'figs/domains_{e2:0.2f}_L{L}')
            except Exception as e:
                print(f'Skipping ({e2:0.2f},{L}) ({e})')
        
if __name__ == '__main__':
    # main_wloops()
    main_mag()
    # main_domains()
