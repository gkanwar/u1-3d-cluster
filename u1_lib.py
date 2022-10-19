import numpy as np
import tqdm

def shift_x(x, d, axis, *, shape):
    xp = tuple(x[i] if i != axis else ((x[i] + d) %  shape[axis])
               for i in range(len(x)))
    return xp

def energy(cfg):
    assert len(cfg.shape) == 3, 'specialized for 3d'
    Nd = 3
    h = cfg/2
    E = 0.0
    for i in range(Nd):
        mp = h - np.roll(h, -1, axis=i)
        E += np.sum(mp**2)
    return E

def action(cfg, *, e2):
    return energy(cfg) * (e2/2)

def local_action(cfg, x, cfg_x, *, e2):
    shape = cfg.shape
    Nd = len(shape)
    hx = cfg_x / 2
    S = 0.0
    for i in range(Nd):
        x_fwd = shift_x(x, 1, axis=i, shape=shape)
        x_bwd = shift_x(x, -1, axis=i, shape=shape)
        h_fwd = cfg[x_fwd]/2
        h_bwd = cfg[x_bwd]/2
        S += (e2/2)*((h_fwd - hx)**2 + (h_bwd - hx)**2)
    return S

def resolved_action(cfg, *, e2):
    shape = cfg.shape
    Nd = len(shape)
    h = cfg / 2
    S = np.zeros(shape, dtype=np.float64)
    for i in range(Nd):
        S += (e2/2)*((np.roll(h, -1, axis=i) - h)**2 + (np.roll(h, 1, axis=i) - h)**2)
    assert S.shape == cfg.shape
    return S

def randomly_translate(cfg):
    Nd = len(cfg.shape)
    origin = (0,)*Nd
    parity = cfg[origin] % 2
    roll = tuple(np.random.randint(Li) for Li in cfg.shape)
    cfg[:] = np.roll(cfg, roll, axis=tuple(range(Nd)))
    if cfg[origin] % 2 != parity:
        cfg -= 1

def compute_M(cfg):
    h = cfg/2
    mask = get_checkerboard_mask(0, shape=cfg.shape)
    return np.sum(h[mask]) - np.sum(h[~mask])
def compute_MT(cfg):
    h = cfg/2
    mask = get_checkerboard_mask(0, shape=cfg.shape)
    h_bar = np.mean(h)
    return np.sum((h[mask]-h_bar)**2) - np.sum((h[~mask]-h_bar)**2)

def compute_MC(cfg): # cube magnetization
    h = cfg/2
    mask = get_checkerboard_mask(0, shape=cfg.shape)
    h1 = np.roll(h, -1, axis=0)
    h2 = np.roll(h, -1, axis=1)
    h3 = np.roll(h, -1, axis=2)
    h12 = np.roll(h1, -1, axis=1)
    h23 = np.roll(h2, -1, axis=2)
    h13 = np.roll(h1, -1, axis=2)
    h123 = np.roll(h12, -1, axis=2)
    cube_h_bar = (h + h1 + h2 + h3 + h12 + h23 + h13 + h123) / 8
    O_cube = (
        (h - cube_h_bar)**2 + (h12 - cube_h_bar)**2 +
        (h23 - cube_h_bar)**2 + (h13 - cube_h_bar)**2
        - (h1 - cube_h_bar)**2 - (h2 - cube_h_bar)**2
        - (h3 - cube_h_bar)**2 - (h123 - cube_h_bar)**2
    )
    return np.sum(O_cube[mask]) - np.sum(O_cube[~mask])

def compute_MS(cfg): # star magnetization
    h = cfg/2
    mask = get_checkerboard_mask(0, shape=cfg.shape)
    Nd = len(cfg.shape)
    hs = []
    for mu in range(Nd):
        hs.append(np.roll(h, -1, axis=mu))
        hs.append(np.roll(h, 1, axis=mu))
    hbar = np.mean(hs, axis=0)
    O_star = np.sum((np.stack(hs, axis=0) - hbar)**2, axis=0)
    return np.sum(O_star[~mask]) - np.sum(O_star[mask])

def make_obs_hist(f, bins):
    def obs_hist(cfg, state, _res):
        f_cfg = f(cfg)
        if state is None:
            state = (bins, np.zeros(len(bins)-1, dtype=int))
        i = np.argmax(f_cfg < bins)
        if i > 0 and f_cfg < bins[i]:
            state[1][i-1] += 1
        return state
    return obs_hist

obs_hist_M = make_obs_hist(compute_M, bins=np.linspace(-0.1, 0.1, num=50, endpoint=True))
obs_hist_MT = make_obs_hist(compute_MT, bins=np.linspace(-0.1, 0.1, num=50, endpoint=True))

def make_obs_hist_clust_sizes(V):
    def obs_hist(cfg, state, res):
        assert 'labels' in res
        _, counts = np.unique(res['labels'], return_counts=True)
        if state is None:
            state = np.zeros(V+1, dtype=int)
        for c in counts:
            state[c] += 1
        return state
    return obs_hist

def make_obs_trace(f):
    def obs_trace(cfg, state, _res):
        if state is None: state = []
        state.append(f(cfg))
        return state
    return obs_trace

obs_trace_M = make_obs_trace(compute_M)
obs_trace_MT = make_obs_trace(compute_MT)
obs_trace_MC = make_obs_trace(compute_MC)
obs_trace_MS = make_obs_trace(compute_MS)

def mcmc(cfg0, *, e2, n_step, n_mc_skip, n_obs_skip, update, n_therm=0, save_cfg=True, obs={}):
    cfg = np.copy(cfg0)
    ens = []
    actions = []
    obs_vals = {k: None for k in obs}
    for i in tqdm.tqdm(range(-n_therm*n_mc_skip, n_step*n_mc_skip)):
        _, res = update(cfg, e2=e2)
        # if (i+1) % 100 == 0:
        #     randomly_translate(cfg)
        if i >= 0 and (i+1) % n_mc_skip == 0:
            if save_cfg: ens.append(np.copy(cfg))
            actions.append(action(cfg, e2=e2))
        if i >= 0 and (i+1) % n_obs_skip == 0:
            for k,O in obs.items():
                obs_vals[k] = O(cfg, obs_vals[k], res)
    return {
        'cfgs': ens,
        'Ss': np.array(actions),
        'Os': obs_vals
    }

def get_checkerboard_mask(p, *, shape):
    arrs = [np.arange(Lx) for Lx in shape]
    return sum(np.ix_(*arrs)) % 2 == p

def make_init_cfg(shape):
    cfg0 = np.zeros(shape, dtype=int)
    mask = get_checkerboard_mask(0, shape=shape)
    cfg0[mask] = 1
    return cfg0
