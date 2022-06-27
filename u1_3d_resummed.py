### Simulations of 3D U(1) height model, with theta = pi, after resumming the
### odd sites of the lattice.

import analysis as al
import functools
import itertools
import matplotlib.pyplot as plt
import numpy as np
import operator
import paper_plt
paper_plt.load_latex_config()
import tqdm

from u1_lib import *

# We need sparser masks for resummed HB, since the cosh interaction acts between
# 7 sites at once. The region of dependence is two sites away for the even
# sites. We can therefore use the following partition of sites:
#  - Odd sites
#  - 2^Nd offsets of checkerboarded subset of even sites
@functools.cache
def get_hb_masks(shape):
    Nd = len(shape)
    V = int(np.prod(shape))
    odd_mask = get_checkerboard_mask(1, shape=shape)
    even_mask = ~odd_mask
    arrs = np.ix_(*[np.arange(Lx) for Lx in shape])
    all_even_sublatt = functools.reduce(operator.and_, (arr % 2 == 0 for arr in arrs))
    assert np.sum(all_even_sublatt) == V//(2**Nd)
    sum_coords = sum(arr//2 for arr in arrs)
    even_chx_mask = all_even_sublatt & (sum_coords % 2 == 0)
    assert np.sum(even_chx_mask) == V//(2**(Nd+1))
    all_masks = [odd_mask]
    axes = tuple(range(Nd))
    basis_vecs = np.stack([np.array((2,) + (0,)*(Nd-1))] + [
        np.array((0,)*i + (-1,1) + (0,)*(Nd-i-2))
        for i in range(Nd-1)
    ], axis=1)
    assert len(basis_vecs) == Nd
    for offs in itertools.product([0,1], repeat=Nd):
        tot_off = basis_vecs @ np.array(offs)
        assert np.sum(tot_off) % 2 == 0
        all_masks.append(np.roll(even_chx_mask, tot_off, axis=axes))
    assert np.all(sum(all_masks) == 1)
    return all_masks

def make_init_cfg(shape):
    cfg0 = np.zeros(shape, dtype=int)
    return cfg0

def action_resum(cfg, *, e2):
    mask = get_checkerboard_mask(1, shape=cfg.shape) # odd sites
    S0 = energy(cfg) * (e2/2)
    local_deltas = 0
    h = cfg/2
    for i in range(len(cfg.shape)):
        local_deltas = local_deltas + (
            2*h - np.roll(h, -1, axis=i) - np.roll(h, 1, axis=i)
        )
    S1 = -np.sum(np.log(np.cosh((e2/2) * local_deltas[mask])))
    return S0 + S1

def local_action_resum(cfg, x, cfg_x, *, e2):
    Nd = len(cfg.shape)
    S = local_action(cfg, x, cfg_x, e2=e2)
    if np.sum(x) % 2 == 1: # odd sites
        hx = cfg_x / 2
        local_deltas = 0
        for i in range(Nd):
            x_fwd = shift_x(x, 1, axis=i, shape=shape)
            x_bwd = shift_x(x, -1, axis=i, shape=shape)
            h_fwd = cfg[x_fwd]/2
            h_bwd = cfg[x_bwd]/2
            local_deltas += 2*hx - h_fwd - h_bwd
        S -= np.log(np.cosh((e2/2) * local_deltas))
    else: # even sites
        for i in range(Nd):
            for s in [-1,1]:
                y = shift_x(x, s, axis=i, shape=shape)
                hy = cfg[y]/2
                local_deltas = 0
                for j in range(Nd):
                    y_fwd = shift_x(y, 1, axis=j, shape=shape)
                    y_bwd = shift_x(y, -1, axis=j, shape=shape)
                    h_fwd = cfg_x/2 if y_fwd == x else cfg[y_fwd]/2
                    h_bwd = cfg_x/2 if y_bwd == x else cfg[y_bwd]/2
                    local_deltas += 2*hy - h_fwd - h_bwd
                S -= np.log(np.cosh((e2/2) * local_deltas))
    return S

def resolved_action_resum(cfg, *, e2):
    shape = cfg.shape
    Nd = len(shape)
    h = cfg / 2
    S = np.zeros(shape, dtype=np.float64)
    for i in range(Nd):
        S += (e2/2)*((np.roll(h, -1, axis=i) - h)**2 + (np.roll(h, 1, axis=i) - h)**2)
    local_deltas = 0
    for i in range(Nd):
        local_deltas = local_deltas + (
            2*h - np.roll(h, -1, axis=i) - np.roll(h, 1, axis=i)
        )
    mask = get_checkerboard_mask(1, shape=cfg.shape)
    # odd sites
    S[mask] -= np.log(np.cosh((e2/2) * local_deltas[mask]))
    # even sites get contribs from all odd neighbors
    for i in range(Nd):
        S[~mask] -= np.log(np.cosh((e2/2) * np.roll(local_deltas, -1, axis=i)[~mask]))
        S[~mask] -= np.log(np.cosh((e2/2) * np.roll(local_deltas, 1, axis=i)[~mask]))
    assert S.shape == cfg.shape
    return S


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
            local_action_resum(cfg, x, new_cfg_x, e2=e2) -
            local_action_resum(cfg, x, cfg_x, e2=e2)
        )
        # TMP
        # new_cfg = np.copy(cfg)
        # new_cfg[x] = new_cfg_x
        # delta_S_test = action_resum(new_cfg, e2=e2) - action_resum(cfg, e2=e2)
        # assert np.isclose(delta_S, delta_S_test), (delta_S, delta_S_test)
        if np.random.random() < np.exp(-delta_S):
            cfg[x] = new_cfg_x
            acc += 1
    # print(f'Acc rate {100*acc/hits:.2f}%')
    return cfg

def hb_update(cfg, *, e2):
    masks = get_hb_masks(shape=cfg.shape)
    dh = 2*np.random.randint(2, size=cfg.shape) - 1
    dcfg = 2*dh
    r = np.random.random(size=cfg.shape)
    tot_acc = 0
    indiv_accs = []
    V = np.prod(cfg.shape)
    for i, mask in enumerate(masks):
        # _tmp_S = action_resum(cfg, e2=e2)
        S = resolved_action_resum(cfg, e2=e2)
        new_cfg = np.copy(cfg)
        new_cfg[mask] += dcfg[mask]
        new_S = resolved_action_resum(new_cfg, e2=e2)
        acc = r[mask] < np.exp(-new_S[mask] + S[mask])
        ind_acc = tuple(np.transpose(np.argwhere(mask)[acc]))
        # print(f'mask {i}', cfg[ind_acc], new_cfg[ind_acc])
        cfg[ind_acc] = new_cfg[ind_acc]
        # _tmp_Sp = action_resum(cfg, e2=e2)
        # assert np.isclose(_tmp_Sp - _tmp_S, np.sum((new_S - S)[ind_acc])), \
        #     (_tmp_Sp - _tmp_S, np.sum((new_S - S)[ind_acc]))
        tot_acc += np.sum(acc) / V
        indiv_accs.append(np.mean(acc))
    # print(f'Acc rate {100*tot_acc:.2f}% {tuple(100*acc for acc in indiv_accs)}')
    return cfg

# (tildeh_x - h_y)^2 ==> wp_x (h_x - h_y + 1/2)^2 + wm_x (h_x - h_y - 1/2)^2
# where wp_x and wm_x are the relative probabilities of +/- 1/2 for b_x
def energy_resum(cfg, *, e2):
    mask = get_checkerboard_mask(1, shape=cfg.shape) # odd sites
    Nd = len(cfg.shape)
    h = cfg/2
    local_deltas = 0
    for i in range(Nd):
        local_deltas = local_deltas + (
            2*h - np.roll(h, -1, axis=i) - np.roll(h, 1, axis=i)
        )
    wp = np.exp(-(e2/2) * local_deltas) / (2*np.cosh((e2/2) * local_deltas))
    wm = np.exp((e2/2) * local_deltas) / (2*np.cosh((e2/2) * local_deltas))
    E = 0.0
    for i in range(Nd):
        E += np.sum((
            wp * ((h - np.roll(h, -1, axis=i) + 1/2)**2 +
                  (h - np.roll(h, 1, axis=i) + 1/2)**2) +
            wm * ((h - np.roll(h, -1, axis=i) - 1/2)**2 +
                  (h - np.roll(h, 1, axis=i) - 1/2)**2)
        )[mask])
    return E

# O_x ==> prod_{odd y in cube(x)}
#     (wp_y delta(h_y, tildeh_y + 1/2) + wm_y delta(h_y, tildeh_y - 1/2)) O_x
def compute_OC_resum(cfg, *, e2):
    mask = get_checkerboard_mask(1, shape=cfg.shape) # odd sites
    Nd = len(cfg.shape)
    h = cfg/2
    local_deltas = 0
    for i in range(Nd):
        local_deltas = local_deltas + (
            2*h - np.roll(h, -1, axis=i) - np.roll(h, 1, axis=i)
        )
    wp = np.exp(-(e2/2) * local_deltas) / (2*np.cosh((e2/2) * local_deltas))
    wm = np.exp((e2/2) * local_deltas) / (2*np.cosh((e2/2) * local_deltas))
    h1 = np.roll(h, -1, axis=0)
    h2 = np.roll(h, -1, axis=1)
    h3 = np.roll(h, -1, axis=2)
    h12 = np.roll(h1, -1, axis=1)
    h23 = np.roll(h2, -1, axis=2)
    h13 = np.roll(h1, -1, axis=2)
    h123 = np.roll(h12, -1, axis=2)
    O_cube = np.zeros(cfg.shape)
    for p,p12,p23,p13 in itertools.product([-1,1], repeat=4):
        cube_h = h + p/2
        cube_h1 = h1
        cube_h2 = h2
        cube_h3 = h3
        cube_h12 = h12 + p12/2
        cube_h23 = h23 + p23/2
        cube_h13 = h13 + p13/2
        cube_h123 = h123
        w = wp if p > 0 else wm
        w12 = np.roll(wp if p12 > 0 else wm, (-1,-1), axis=(0,1))
        w23 = np.roll(wp if p23 > 0 else wm, (-1,-1), axis=(1,2))
        w13 = np.roll(wp if p13 > 0 else wm, (-1,-1), axis=(0,2))
        tot_w = w*w12*w23*w13
        cube_h_bar = (cube_h + cube_h1 + cube_h2 + cube_h3 +
                      cube_h12 + cube_h23 + cube_h13 + cube_h123) / 8
        O_cube[mask] += tot_w[mask] * (
            (cube_h - cube_h_bar)**2 + (cube_h12 - cube_h_bar)**2 +
            (cube_h23 - cube_h_bar)**2 + (cube_h13 - cube_h_bar)**2
            - (cube_h1 - cube_h_bar)**2 - (cube_h2 - cube_h_bar)**2
            - (cube_h3 - cube_h_bar)**2 - (cube_h123 - cube_h_bar)**2
        )[mask]
    for p1,p2,p3,p123 in itertools.product([-1,1], repeat=4):
        cube_h = h
        cube_h1 = h1 + p1/2
        cube_h2 = h2 + p2/2
        cube_h3 = h3 + p3/2
        cube_h12 = h12
        cube_h23 = h23
        cube_h13 = h13
        cube_h123 = h123 + p123/2
        w1 = np.roll(wp if p1 > 0 else wm, -1, axis=0)
        w2 = np.roll(wp if p2 > 0 else wm, -1, axis=1)
        w3 = np.roll(wp if p3 > 0 else wm, -1, axis=2)
        w123 = np.roll(wp if p123 > 0 else wm, (-1,-1,-1), axis=(0,1,2))
        tot_w = w1*w2*w3*w123
        cube_h_bar = (cube_h + cube_h1 + cube_h2 + cube_h3 +
                      cube_h12 + cube_h23 + cube_h13 + cube_h123) / 8
        O_cube[~mask] += tot_w[~mask] * (
            (cube_h - cube_h_bar)**2 + (cube_h12 - cube_h_bar)**2 +
            (cube_h23 - cube_h_bar)**2 + (cube_h13 - cube_h_bar)**2
            - (cube_h1 - cube_h_bar)**2 - (cube_h2 - cube_h_bar)**2
            - (cube_h3 - cube_h_bar)**2 - (cube_h123 - cube_h_bar)**2
        )[~mask]

    return O_cube

def compute_MC_resum(cfg, *, e2):
    mask = get_checkerboard_mask(1, shape=cfg.shape) # odd sites
    O_cube = compute_OC_resum(cfg, e2=e2)
    return np.sum(O_cube[mask]) - np.sum(O_cube[~mask])

def _test_metr_vs_hb():
    L = 8
    e2 = 1.0
    shape = (L,)*3
    cfg = make_init_cfg(shape)
    metr_Es = []
    for i in tqdm.tqdm(range(200)):
        cfg = metropolis_update(cfg, e2=e2)
        metr_Es.append(energy_resum(cfg, e2=e2))
    cfg = make_init_cfg(shape)
    hb_Es = []
    for i in tqdm.tqdm(range(200)):
        cfg = hb_update(cfg, e2=e2)
        hb_Es.append(energy_resum(cfg, e2=e2))
        print(cfg)
    V = L**3
    fig, ax = plt.subplots(1,1)
    ax.plot(np.array(metr_Es)/V, label='metr')
    ax.plot(np.array(hb_Es)/V, label='hb')
    ax.legend()
    fig.savefig('tmp.pdf')

def _test_E_sweep():
    L = 8
    V = L**3
    shape = (L,)*3
    e2s = np.arange(0.50, 1.80+1e-6, 0.05)
    E_ests = []
    MC_ests = []
    MC2_ests = []
    MC2p_ests = []
    for e2 in e2s:
        cfg = make_init_cfg(shape)
        hb_Es = []
        hb_MCs = []
        for i in tqdm.tqdm(range(-100, 5000)):
            cfg = hb_update(cfg, e2=e2)
            if i >= 0:
                hb_Es.append(energy_resum(cfg, e2=e2))
                hb_MCs.append(compute_MC_resum(cfg, e2=e2))
        E_ests.append(al.bootstrap(np.array(hb_Es)/V, Nboot=1000, f=al.rmean))
        MC_ests.append(al.bootstrap(np.array(hb_MCs), Nboot=1000, f=al.rmean))
        MC2_ests.append(al.bootstrap(np.array(hb_MCs)**2 / V, Nboot=1000, f=al.rmean))
        MC2p_ests.append(al.bootstrap(np.array(hb_MCs)**2 / V**2, Nboot=1000, f=al.rmean))
    E_ests = np.transpose(E_ests)
    MC_ests = np.transpose(MC_ests)
    MC2_ests = np.transpose(MC2_ests)
    MC2p_ests = np.transpose(MC2p_ests)
    fig, axes = plt.subplots(1,4, figsize=(10,2.5))
    al.add_errorbar(E_ests, xs=e2s, ax=axes[0], marker='o', fillstyle='none', markersize=6)
    axes[0].set_ylabel(r'$\left< E/V \right>$')
    axes[0].set_yscale('log')
    al.add_errorbar(MC_ests, xs=e2s, ax=axes[1], marker='o', fillstyle='none', markersize=6)
    axes[1].set_ylabel(r'$\left< M_C \right>$')
    al.add_errorbar(MC2_ests, xs=e2s, ax=axes[2], marker='o', fillstyle='none', markersize=6)
    axes[2].set_ylabel(r'$\left< M_C^2 \right> / V$')
    axes[2].set_yscale('log')
    al.add_errorbar(MC2p_ests, xs=e2s, ax=axes[3], marker='o', fillstyle='none', markersize=6)
    axes[3].set_ylabel(r'$\left< M_C^2 \right> / V^2$')
    axes[3].set_yscale('log')
    for ax in axes:
        ax.set_xlabel(r'$e^2$')
    fig.set_tight_layout(True)
    fig.savefig('tmp.pdf')

    fig, ax = plt.subplots(1,1)
    ax.plot(hb_MCs)
    fig.savefig('tmp2.pdf')

if __name__ == '__main__': _test_E_sweep()
