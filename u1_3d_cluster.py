import analysis as al
import functools
import itertools
import matplotlib.pyplot as plt
import numpy as np
import sys
import time
import tqdm

from cluster.cluster import sample_flip_mask as sample_flip_mask_cext
from u1_lib import *

# 3D cluster algorithm for the dual U(1) Wilson-type theory with half-integer quanta.
# The dual model consists of height variables which are integer (half-integer) on
# even (odd) sites of the lattice. These can be efficiently updated using a cluster
# algorithm performing height flips.

# For convenience, cfg represents 2*h so that all elements can be an integer.
# Whenever actually computing weights with cfg, we first must extract h = cfg/2.

def sample_bonds(cfg, *, e2, h_star):
    assert len(cfg.shape) == 3, 'specialized for 3d'
    bonds = []
    hx = cfg / 2
    for i in range(3):
        hy = np.roll(hx, -1, axis=i)
        R = np.exp(-2*e2*(h_star - hx)*(h_star - hy))
        p_bond = 1 - np.minimum(1, R)
        assert p_bond.shape == cfg.shape
        bonds.append(np.random.random(size=p_bond.shape) < p_bond)
    return np.stack(bonds, axis=0)

def sample_flip_mask(bonds, rand_bits):
    assert len(bonds.shape) == 3+1, 'specialized for 3d'
    assert bonds.shape[0] == 3, 'specialized for 3d'
    assert len(rand_bits.shape) == 1 and rand_bits.shape[0] >= np.prod(bonds.shape[1:]), \
        'rand_bits must be 1d string of enough bits'
    Nd = 3
    shape = bonds.shape[1:]
    labels = np.zeros(shape, dtype=int)
    flip_mask = -1*np.ones(shape, dtype=int)
    cur_label = 0
    comp_sizes = []
    rand_ind = 0
    for x in itertools.product(*[range(Li) for Li in shape]):
        if labels[x] != 0: continue
        cur_flip = rand_bits[rand_ind] % 2
        rand_ind += 1
        cur_label += 1
        labels[x] = cur_label
        flip_mask[x] = cur_flip
        comp_size = 0
        queue = [x]
        while len(queue) > 0:
            comp_size += 1
            y = queue.pop()
            for i in range(Nd):
                y_fwd = shift_x(y, 1, axis=i, shape=shape)
                y_bwd = shift_x(y, -1, axis=i, shape=shape)
                if bonds[(i,) + y] and labels[y_fwd] != cur_label:
                    assert labels[y_fwd] == 0
                    labels[y_fwd] = cur_label
                    flip_mask[y_fwd] = cur_flip
                    queue.append(y_fwd)
                if bonds[(i,) + y_bwd] and labels[y_bwd] != cur_label:
                    assert labels[y_bwd] == 0
                    labels[y_bwd] = cur_label
                    flip_mask[y_bwd] = cur_flip
                    queue.append(y_bwd)
        comp_sizes.append(comp_size)
    assert np.sum(comp_sizes) == np.prod(shape)
    assert np.all(flip_mask != -1)
    for i in range(Nd):
        assert np.all(np.logical_or(~bonds[i], labels == np.roll(labels, -1, axis=i)))
    return {
        'mask': (flip_mask == 1),
        'labels': labels,
        'sizes': comp_sizes
    }

def cluster_update(cfg, *, e2):
    assert len(cfg.shape) == 3, 'specialized for 3d'
    x = tuple(np.random.randint(Li) for Li in cfg.shape)
    cfg_star = cfg[x]
    h_star = cfg_star / 2
    _start1 = time.time()
    bonds = sample_bonds(cfg, e2=e2, h_star=h_star)
    _end1 = time.time()
    _start2a = time.time()
    rand_bits = np.random.randint(2, size=int(np.prod(bonds.shape[1:]))).astype(np.int32)
    # res = sample_flip_mask(bonds, rand_bits)
    _end2a = time.time()
    _start2b = time.time()
    bonds = bonds.astype(np.int32)
    res2 = sample_flip_mask_cext(bonds, rand_bits)
    res2 = {'mask': res2[0] == 1, 'labels': res2[1]}
    # assert np.allclose(res['labels'], res2['labels']), 'labels mismatch'
    # assert np.allclose(res['mask'], res2['mask']), 'flip_mask mismatch'
    _end2b = time.time()
    # flip_mask = res['mask']
    flip_mask = res2['mask']
    _start3 = time.time()
    # print('-'*len(flip_mask))
    # for row in flip_mask[0]:
    #     print(''.join(map(lambda x: '#' if x else ' ', row)))
    _end3 = time.time()
    _start4 = time.time()
    cfg[flip_mask] = 2*cfg_star - cfg[flip_mask]
    assert cfg[x] == cfg_star
    mean = 2*int(np.round(np.mean(cfg/2)))
    cfg -= mean
    _end4 = time.time()
    print(f'sample_bonds time = {_end1-_start1:.4f}s')
    print(f'sample_flip_mask time = {_end2a-_start2a:.4f}s')
    print(f'sample_flip_mask_cext time = {_end2b-_start2b:.4f}s')
    print(f'print_clusters time = {_end3-_start3:.4f}s')
    print(f'do_update time = {_end4-_start4:.4f}s')
    return cfg

def metropolis_update(cfg, *, e2):
    assert len(cfg.shape) == 3, 'specialized for 3d'
    shape = cfg.shape
    acc = 0
    hits = 0
    for x in itertools.product(*[range(Li) for Li in shape]):
        hits += 1
        delta = 4*np.random.randint(2) - 2
        cfg_x = cfg[x]
        new_cfg_x = cfg[x] + delta
        delta_S = (
            local_action(cfg, x, new_cfg_x, e2=e2) -
            local_action(cfg, x, cfg_x, e2=e2)
        )
        # TMP
        # new_cfg = np.copy(cfg)
        # new_cfg[x] = new_cfg_x
        # assert np.isclose(
        #     delta_S,
        #     action(new_cfg, e2=e2) - action(cfg, e2=e2))
        if np.random.random() < np.exp(-delta_S):
            cfg[x] = new_cfg_x
            acc += 1
    print(f'Acc rate {100*acc/hits:.2f}%')
    return cfg

def hb_update(cfg, *, e2):
    mask = get_checkerboard_mask(0, shape=cfg.shape)
    dh = 2*np.random.randint(2, size=cfg.shape) - 1
    dcfg = 2*dh
    r = np.random.random(size=cfg.shape)
    # even sites
    S = resolved_action(cfg, e2=e2)
    new_cfg = np.copy(cfg)
    new_cfg[mask] += dcfg[mask]
    new_S = resolved_action(new_cfg, e2=e2)
    acc = r[mask] < np.exp(-new_S[mask] + S[mask])
    ind_acc = tuple(np.transpose(np.argwhere(mask)[acc]))
    cfg[ind_acc] = new_cfg[ind_acc]
    # odd sites
    S = resolved_action(cfg, e2=e2)
    mask = ~mask
    new_cfg = np.copy(cfg)
    new_cfg[mask] += dcfg[mask]
    new_S = resolved_action(new_cfg, e2=e2)
    acc = r[mask] < np.exp(-new_S[mask] + S[mask])
    ind_acc = tuple(np.transpose(np.argwhere(mask)[acc]))
    cfg[ind_acc] = new_cfg[ind_acc]
    return cfg

# mixed_update = lambda cfg, *, e2: hb_update(cluster_update(cfg, e2=e2), e2=e2)
### FORNOW - skip heatbath
mixed_update = lambda cfg, *, e2: cluster_update(cfg, e2=e2)

def run_exploratory_sweep():
    L = 8
    shape = (L,)*3
    cfg0 = make_init_cfg(shape)

    all_e2s = [0.8, 1.0, 1.2, 1.4] # np.arange(0.25, 2.0+1e-6, 0.25)
    all_actions = []
    est_actions = []
    all_Ms = []
    all_MTs = []
    est_Ms = []
    est_MTs = []
    n_step = 110
    binsize = 5
    th = 10
    for e2 in all_e2s:
        res_metr = mcmc(cfg0, e2=e2, n_step=n_step, n_skip=10, update=hb_update)
        # mcmc(cfg0, e2=e2, n_step=n_step, n_skip=10, update=metropolis_update)
        res_clust = mcmc(cfg0, e2=e2, n_step=n_step, n_skip=10, update=cluster_update)
        res_hb = mcmc(cfg0, e2=e2, n_step=n_step, n_skip=10, update=hb_update)
        all_actions.append((res_metr['Ss'], res_clust['Ss'],  res_hb['Ss']))
        est_actions.append((
            al.bootstrap(al.bin_data(res_metr['Ss'][th:], binsize=binsize)[1], Nboot=100, f=al.rmean),
            al.bootstrap(al.bin_data(res_metr['Ss'][th:], binsize=binsize)[1], Nboot=100, f=al.rmean),
            al.bootstrap(al.bin_data(res_hb['Ss'][th:], binsize=binsize)[1], Nboot=100, f=al.rmean)
        ))
        est_Ms.append((
            al.bootstrap(al.bin_data(res_metr['Ms'][th:], binsize=binsize)[1], Nboot=100, f=al.rmean),
            al.bootstrap(al.bin_data(res_clust['Ms'][th:], binsize=binsize)[1], Nboot=100, f=al.rmean),
            al.bootstrap(al.bin_data(res_hb['Ms'][th:], binsize=binsize)[1], Nboot=100, f=al.rmean)
        ))
        est_MTs.append((
            al.bootstrap(al.bin_data(res_metr['MTs'][th:], binsize=binsize)[1], Nboot=100, f=al.rmean),
            al.bootstrap(al.bin_data(res_clust['MTs'][th:], binsize=binsize)[1], Nboot=100, f=al.rmean),
            al.bootstrap(al.bin_data(res_hb['MTs'][th:], binsize=binsize)[1], Nboot=100, f=al.rmean)
        ))
        all_Ms.append((res_metr['Ms'], res_clust['Ms'], res_hb['Ms']))
        all_MTs.append((res_metr['MTs'], res_clust['MTs'], res_hb['MTs']))
        # np.save(f'ens_{e2:.2f}_metr.npy', res_metr['cfgs'][th:])
        # np.save(f'ens_{e2:.2f}_clust.npy', res_clust['cfgs'][th:])
    est_Ms = np.swapaxes(est_Ms, axis1=0, axis2=1)
    est_MTs = np.swapaxes(est_MTs, axis1=0, axis2=1)
    est_actions = np.swapaxes(est_actions, axis1=0, axis2=1)

    fig, axes = plt.subplots(3,1)
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
    for (
            e2, (actions_metr, actions_clust, actions_hb),
            (M_metr, M_clust, M_hb), (MT_metr, MT_clust, MT_hb), color
    ) in zip(all_e2s, all_actions, all_Ms, all_MTs, colors):
        axes[0].plot(actions_metr, label=f'e2={e2} (Metr)', color=color, marker='o')
        axes[0].plot(actions_clust, label=f'e2={e2} (Clust)', color=color, marker='^')
        axes[0].plot(actions_hb, label=f'e2={e2} (HB)', color=color, marker='^')
        axes[1].plot(M_metr, color=color, marker='o')
        axes[1].plot(M_clust, color=color, marker='^')
        axes[1].plot(M_hb, color=color, marker='^')
        axes[2].plot(MT_metr, color=color, marker='o')
        axes[2].plot(MT_clust, color=color, marker='^')
        axes[2].plot(MT_hb, color=color, marker='^')
    axes[0].legend()
    fig.set_tight_layout(True)
    fig.savefig('figs/mcmc.pdf')

    fig, ax = plt.subplots(1,1)
    al.add_errorbar(
        np.transpose(est_Ms[0]), xs=all_e2s, ax=ax, label='M (Metr)',
        color='k', marker='o')
    al.add_errorbar(
        np.transpose(est_Ms[1]), xs=all_e2s, ax=ax, label='M (Clust)',
        color='k', marker='^')
    al.add_errorbar(
        np.transpose(est_Ms[2]), xs=all_e2s, ax=ax, label='M (HB)',
        color='k', marker='^')
    al.add_errorbar(
        np.transpose(est_MTs[0]), xs=all_e2s, ax=ax, label='MT (Metr)',
        color='xkcd:purple', marker='o')
    al.add_errorbar(
        np.transpose(est_MTs[1]), xs=all_e2s, ax=ax, label='MT (Clust)',
        color='xkcd:purple', marker='^')
    al.add_errorbar(
        np.transpose(est_MTs[2]), xs=all_e2s, ax=ax, label='MT (HB)',
        color='xkcd:purple', marker='^')
    ax.legend()
    fig.set_tight_layout(True)
    fig.savefig('figs/magnetization.pdf')

    fig, ax = plt.subplots(1,1)
    al.add_errorbar(
        np.transpose(est_actions[0]), xs=all_e2s, ax=ax, label='S (Metr)',
        color='k', marker='o')
    al.add_errorbar(
        np.transpose(est_actions[1]), xs=all_e2s, ax=ax, label='S (Clust)',
        color='k', marker='^')
    al.add_errorbar(
        np.transpose(est_actions[2]), xs=all_e2s, ax=ax, label='S (HB)',
        color='k', marker='^')
    ax.legend()
    fig.set_tight_layout(True)
    fig.savefig('figs/action.pdf')
    # plt.show()

def run_exploratory_sweep2():
    L = 8
    shape = (L,)*3
    cfg0 = make_init_cfg(shape)

    all_e2s = np.arange(0.8, 1.4+1e-6, 0.04)
    n_step = 110
    th = 10
    for e2 in all_e2s:
        res_clust = mcmc(cfg0, e2=e2, n_step=n_step, n_skip=10, update=cluster_update)
        np.save(f'ens_{e2:.2f}_clust2.npy', res_clust['cfgs'][th:])

def run_exploratory_sweep3(rank):
    L = 8
    shape = (L,)*3
    cfg0 = make_init_cfg(shape)

    all_e2s = np.arange(0.40, 1.10+1e-6, 0.02)
    n_step = 1010
    th = 10
    if rank >= len(all_e2s): return
    e2 = all_e2s[rank]
    res_clust = mcmc(cfg0, e2=e2, n_step=n_step, n_skip=500, update=mixed_update)
    np.save(f'ens_{e2:0.2f}_L{L}_mixed.npy', res_clust['cfgs'][th:])

def run_exploratory_sweep4(rank):
    L = 16
    shape = (L,)*3
    cfg0 = make_init_cfg(shape)

    all_e2s = np.arange(0.40, 1.10+1e-6, 0.02)
    n_step = 260
    th = 10
    if rank >= len(all_e2s): return
    e2 = all_e2s[rank]
    res_clust = mcmc(cfg0, e2=e2, n_step=n_step, n_skip=5000, update=mixed_update)
    np.save(f'ens_{e2:0.2f}_L{L}_mixed.npy', res_clust['cfgs'][th:])

def run_exploratory_sweep5(rank):
    L = 32
    shape = (L,)*3
    cfg0 = make_init_cfg(shape)

    all_e2s = np.arange(0.50, 1.10+1e-6, 0.02)
    n_step = 110
    th = 10
    if rank >= len(all_e2s): return
    e2 = all_e2s[rank]
    # for e2 in all_e2s:
    res_clust = mcmc(cfg0, e2=e2, n_step=n_step, n_skip=100, update=mixed_update)
    np.save(f'ens_{e2:0.2f}_L{L}_mixed.npy', res_clust['cfgs'][th:])

def run_exploratory_sweep6(rank):
    L = 64
    shape = (L,)*3
    cfg0 = make_init_cfg(shape)

    all_e2s = np.arange(0.92, 1.10+1e-6, 0.02)
    # all_e2s = np.arange(0.2, 0.70+1e-6, 0.10)
    n_step = 110
    th = 10
    if rank >= len(all_e2s): return
    e2 = all_e2s[rank]
    # for e2 in all_e2s:
    res_clust = mcmc(cfg0, e2=e2, n_step=n_step, n_skip=5000, update=mixed_update)
    np.save(f'ens_{e2:0.2f}_L{L}_mixed.npy', res_clust['cfgs'][th:])

def run_exploratory_sweep7(rank):
    L = 96
    shape = (L,)*3
    cfg0 = make_init_cfg(shape)

    # all_e2s = np.arange(0.50, 1.00+1e-6, 0.10)
    all_e2s = np.arange(0.20, 0.50+1e-6, 0.10)
    n_step = 110
    th = 10
    if rank >= len(all_e2s): return
    e2 = all_e2s[rank]
    # for e2 in all_e2s:
    res_clust = mcmc(cfg0, e2=e2, n_step=n_step, n_skip=2500, update=mixed_update)
    np.save(f'ens_{e2:0.2f}_L{L}_mixed.npy', res_clust['cfgs'][th:])

def run_exploratory_sweep8(rank):
    all_Ls = np.array([8, 16, 32, 64], dtype=int)
    all_e2s = np.arange(1.20, 3.00+1e-6, 0.2)
    all_Ls, all_e2s = np.meshgrid(all_Ls, all_e2s, indexing='ij')
    all_Ls = all_Ls.flatten()
    all_e2s = all_e2s.flatten()
    n_step = 110
    th = 10
    if rank >= len(all_e2s): return
    L = all_Ls[rank]
    e2 = all_e2s[rank]
    shape = (L,)*3
    cfg0 = make_init_cfg(shape)
    res_clust = mcmc(cfg0, e2=e2, n_step=n_step, n_skip=1000, update=mixed_update)
    np.save(f'ens_{e2:0.2f}_L{L}_mixed.npy', res_clust['cfgs'][th:])

def run_histogram_sweep(rank):
    all_Ls = np.array([8, 16, 32, 64, 96, 128, 192, 256], dtype=int)
    all_e2s = np.arange(0.40, 2.60+1e-6, 0.2)
    all_Ls, all_e2s = np.meshgrid(all_Ls, all_e2s, indexing='ij')
    all_Ls = all_Ls.flatten()
    all_e2s = all_e2s.flatten()

    n_step = 110
    th = 10
    if rank >= len(all_e2s): return
    
    L = all_Ls[rank]
    e2 = all_e2s[rank]
    shape = (L,)*3
    cfg0 = make_init_cfg(shape)

    res_clust = mcmc(
        cfg0, e2=e2, n_step=n_step, n_mc_skip=1000, n_obs_skip=10,
        save_cfg=False,
        update=mixed_update, obs={'M': obs_hist_M, 'MT': obs_hist_MT})
    np.save(f'raw_obs/obs_{e2:0.2f}_L{L}_mixed.npy', res_clust['Os'])

def run_trace_sweep(rank):
    # all_Ls = np.array([8, 16, 32, 48, 64, 80], dtype=int)
    all_Ls = np.array([96, 128], dtype=int)
    all_e2s = np.arange(0.30, 1.80+1e-6, 0.05)
    all_Ls, all_e2s = np.meshgrid(all_Ls, all_e2s, indexing='ij')
    all_Ls = all_Ls.flatten()
    all_e2s = all_e2s.flatten()

    n_step = 100
    th = 10
    if rank >= len(all_e2s): return
    
    L = all_Ls[rank]
    e2 = all_e2s[rank]
    shape = (L,)*3
    cfg0 = make_init_cfg(shape)

    obs_trace_E = make_obs_trace(energy)
    obs_trace_S = make_obs_trace(functools.partial(action, e2=e2))
    res_clust = mcmc(
        cfg0, e2=e2, n_step=n_step, n_therm=th, n_mc_skip=100, n_obs_skip=10,
        save_cfg=False, update=mixed_update,
        obs={
            'M': obs_trace_M,
            'MT': obs_trace_MT,
            'MC': obs_trace_MC,
            'S': obs_trace_S,
            'E': obs_trace_E
        })
    obs_trace = {k: np.array(v) for k,v in res_clust['Os'].items()}
    obs_trace['version'] = 5
    np.save(f'raw_obs/obs_trace_{e2:0.2f}_L{L}_mixed.npy', obs_trace)

def run_big_ensembles():
    L = 64
    shape = (L,)*3
    cfg0 = make_init_cfg(shape)
    e2 = 0.2

    n_step = 110
    th = 10

    res = mcmc(cfg0, e2=e2, n_step=n_step, n_skip=100, update=cluster_update)
    fig, ax = plt.subplots(1,1)
    ax.plot(res['Ss'])
    fig.savefig('figs/actions.pdf')

    np.save(f'ens_{e2:0.2f}_L{L}.npy', res['cfgs'][th:])

def main():
    rank = int(sys.argv[1]) if len(sys.argv) == 2 else 0
    run_trace_sweep(rank)

if __name__ == '__main__': main()
