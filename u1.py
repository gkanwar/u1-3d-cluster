import numpy as np

# Representation format:
#   cfg[i,x] = 2*m[i,x], where -j <= m <= j
# To get to index [0, ..., d-1 = 2j] we can compute
#   ind[i,x] = (cfg[i,x] + d - 1)//2
#
# Hamiltonian:
#   H = (e^2/2) \sum_xi E_xi^2 + \sum_p (1/2e^2) B_p^2
#   E_xi^2 = m[i,x]^2
#   B_p^2 = -(U1*U2*U3*U4 + (U1*U2*U3*U4)^dag) / 2
#
# Trotterized steps:
#  (1) all E_xi
#  (2) B_p^2 on a non-overlapping subset of plaqs
# Evaluating (1) gives a contribution to the action within a timeslice.
# Evaluating (2) requires exponentiating the raising/lowering operators,
# which gives
#   \sum_\Delta f_\Delta(e^2, dt) delta(m1', m1+Delta) delta(m2', m2+Delta) ...
# on each included plaquette, where m1, m2, etc are the spin values of the
# links included in that plaquette. In other words, we have some positive
# function which determines the action associated with raising/lowering each
# plaquette.
#
# Determining f_\Delta(e^2, dt) is not too difficult:
#   exp(A (Up + Up^dag)) = \sum_nm A^(n+m)*Up^n*Up^dag^m / n!m!
# NOTE: Below is incorrect! Must handle tensor product structure.
# Probably best to diagonalize. The eigenvalues are -2j, ..., 2j-2, 2j,
# i.e. 2m for each possible -j <= m <= j.
# Each eigenvalue is unique, so solving for the associated eigenvectors
# is also straightforward:
#   2 m' \sum_m c_m |m> = (Up + Up^\dag) \sum_m c_m |m>
#   2 m' \sum_m c_m |m> = \sum_m c_m (
#      sqrt(j(j+1) - m(m+1)) |m+1> + sqrt(j(j+1) - m(m-1)) |m-1> )
# We choose the top state to have coeff 1, so
#   2 m' = c_{j-1} sqrt(j(j+1) - (j-1)j)
#   c_{j-1} = (2 m') / sqrt(2j)
# Then all lower states can be recursively determined
#   2 m' c_m = sqrt(j(j+1) - (m-1)m) c_{m-1} + sqrt(j(j+1) - m(m+1)) c_{m+1}
#   c_{m-1} = (2 m' c_m - sqrt(j(j+1) - m(m+1)) c_{m+1})
#      / sqrt(j(j+1) - (m-1)m)
# Finally, the bottom state is given by
#   2 m' c_{-j} = sqrt(j(j+1) - (-j+1)(-j)) c_{-j+1}
#   c_{-j} = sqrt(2 j) c_{-j+1} / (2 m')


def shift_x(x, d, axis, *, shape):
    xp = tuple(x[i] if i != axis else ((x[i] + d) %  shape[axis])
               for i in range(len(x)))
    return xp

cache_U_Udag = {}
cache_Up_Updag_diag = {}

def make_U_Udag(d):
    if d in cache_U_Udag:
        return cache_U_Udag[d]
    U = np.zeros((d,d), dtype=np.float64)
    Udag = np.zeros((d,d), dtype=np.float64)
    j = (d-1)/2
    for i,m in enumerate(np.arange(-j, j, 1)):
        U[i+1,i] = Udag[i,i+1] = np.sqrt(j*(j+1) - m*(m+1))
    cache_U_Udag[d] = (U, Udag)
    return U, Udag

# Up = U*U*Udag*Udag, assuming CCW ordering, last link runs fastest
def get_Up_Updag_diag(d):
    if d in cache_Up_Updag_diag:
        return cache_Up_Updag_diag[d]
    U, Udag = make_U_Udag(d)
    Up = np.einsum('ai,bj,ck,dl->abcdijkl', U, U, Udag, Udag).reshape(d**4, d**4)
    Updag = np.einsum('ai,bj,ck,dl->abcdijkl', Udag, Udag, U, U).reshape(d**4, d**4)
    evs, evecs = np.linalg.eig(Up + Updag)
    cache_Up_Updag_diag[d] = (evs, evecs)
    return evs, evecs # evs[i] goes with evecs[:,i]

def get_plaq_matrix_elt(a, b, *, A, d):
    evs, evecs = get_Up_Updag_diag(d)
    exp_evs = np.exp(A*evs)
    if not np.isscalar(a):
        a = np.array(a)
    if not np.isscalar(b):
        b = np.array(b)
    return np.einsum('...i,i,...i->...', evecs[a], exp_evs, np.conj(evecs[b]))

def _test_get_plaq_matrix_elt():
    assert np.isclose(get_plaq_matrix_elt(5, 80, A=0.15, d=4), 1.799092635)
    assert np.isclose(get_plaq_matrix_elt(26, 101, A=0.15, d=4), 5.345377978)
    assert np.allclose(
        get_plaq_matrix_elt([9,10], [84,85], A=0.15, d=4),
        np.array([2.271476334, 4.480317080]))
    print('[PASSED test_get_plaq_matrix_elt]')
# if __name__ == '__main__': _test_get_plaq_matrix_elt()

def action_term1(cfg_slice):
    return np.sum((cfg_slice/2)**2)

def get_checkerboard_mask(p, *, shape):
    arrs = [np.arange(Lx) for Lx in shape]
    return sum(np.ix_(*arrs)) % 2 == p

def _test_checkerboard_mask():
    assert np.all(
        get_checkerboard_mask(0, shape=(3,3)) ==
        np.array([
            [True, False, True],
            [False, True, False],
            [True, False, True]]))
    assert np.all(
        get_checkerboard_mask(1, shape=(3,3)) ==
        np.array([
            [False, True, False],
            [True, False, True],
            [False, True, False]]))
    assert np.all(
        get_checkerboard_mask(0, shape=(2,3,3)) ==
        np.array([
            [[True, False, True],
             [False, True, False],
             [True, False, True]],
            [[False, True, False],
             [True, False, True],
             [False, True, False]]
        ]))
    print('[PASSED test_checkeboard_mask]')
# if __name__ == '__main__': _test_checkerboard_mask()

def make_plaq_inds(cfg, i, j, *, d):
    Nd = len(cfg.shape[1:])
    assert cfg.shape[0] == Nd-1
    assert i >= 0 and i < cfg.shape[0] and j >= 0 and j < cfg.shape[0]
    assert np.all((cfg + d - 1) % 2 == 0)
    ind = (cfg + d - 1) // 2
    U0 = ind[i]
    U1 = np.roll(ind[j], -1, axis=i)
    U2 = np.roll(ind[i], -1, axis=j)
    U3 = ind[j]
    assert np.all(U0 < d) and np.all(U1 < d) and np.all(U2 < d) and np.all(U3 < d), \
        (U0, U1, U2, U3)
    return U0*d**3 + U1*d**2 + U2*d + U3

def action_term2(cfg_slice1, cfg_slice2, *, coeffB, p, d):
    assert p in [0,1]
    Nd = len(cfg_slice1.shape)-1
    spatial_shape = cfg_slice1.shape[1:-1]
    Lt = cfg_slice1.shape[-1]
    mask = get_checkerboard_mask(p, shape=spatial_shape)
    count = np.sum(mask)
    S = 0.0
    A = coeffB / 2 # coeff of (Up + Up^dag) in Trotterized (-dt H)
    for i in range(Nd-1):
        for j in range(i+1, Nd-1):
            plaq_ij_slice1 = make_plaq_inds(cfg_slice1, i, j, d=d)[mask]
            plaq_ij_slice2 = make_plaq_inds(cfg_slice2, i, j, d=d)[mask]
            assert plaq_ij_slice1.shape == (count, Lt)
            assert plaq_ij_slice2.shape == (count, Lt)
            S -= np.sum(np.log(get_plaq_matrix_elt(plaq_ij_slice1, plaq_ij_slice2, A=A, d=d)))
    return S
            

def action_3d(cfg, *, d, e2, dt):
    # plaquettes can be checkerboarded ==> (1) + (2a) + (2b)
    assert cfg.shape[0] == 2, 'cfg must be 2+1D'
    assert len(cfg.shape) == 1+2+1, 'cfg must be 2+1D'
    assert cfg.shape[-1] % 2 == 0, 'cfg must be even length in time'
    coeffE = dt * (e2 / 2) # coeff of E_xi^2
    coeffB = dt / (2*e2) # coeff of B_p^2
    S1 = coeffE * action_term1(cfg[...,::2])
    S2a = action_term2(cfg[...,::2], cfg[...,1::2], coeffB=coeffB, p=0, d=d)
    S2b = action_term2(cfg[...,1::2], np.roll(cfg[...,::2], -1, axis=-1), coeffB=coeffB, p=1, d=d)
    # print(f'S1 = {S1}, S2a = {S2a}, S2b = {S2b}')
    return S1 + S2a + S2b

def metropolis_sweep_3d(cfg, *, d, e2, dt):
    assert cfg.shape[0] == 2, 'cfg must be 2+1D'
    assert len(cfg.shape) == 1+2+1, 'cfg must be 2+1D'
    shape = cfg.shape[1:]
    max_cfg = d-1
    S = action_3d(cfg, d=d, e2=e2, dt=dt)
    it = np.nditer(cfg[0], flags=['multi_index'])
    acc = 0
    hits = 0
    with it:
        while not it.finished:
            x = it.multi_index
            if sum(x) % 2 != 1:
                it.iternext()
                continue
            hits += 1
            x_fwd0 = shift_x(x, 1, axis=0, shape=shape)
            x_fwd1 = shift_x(x, 1, axis=1, shape=shape)
            x_fwd2 = shift_x(x, 1, axis=2, shape=shape)
            x_fwd02 = shift_x(x_fwd0, 1, axis=2, shape=shape)
            x_fwd12 = shift_x(x_fwd1, 1, axis=2, shape=shape)
            # print(f'{x=} {x_fwd0=} {x_fwd1=} {x_fwd2=} {x_fwd02=} {x_fwd12=}')
            ind0 = (0,) + x
            ind1 = (1,) + x_fwd0
            ind2 = (0,) + x_fwd1
            ind3 = (1,) + x
            ind4 = (0,) + x_fwd2
            ind5 = (1,) + x_fwd02
            ind6 = (0,) + x_fwd12
            ind7 = (1,) + x_fwd2
            U0, U1, U2, U3 = cfg[ind0], cfg[ind1], cfg[ind2], cfg[ind3]
            U4, U5, U6, U7 = cfg[ind4], cfg[ind5], cfg[ind6], cfg[ind7]
            # print(U0, U1, U2, U3)
            # print(U4, U5, U6, U7)
            Delta = 4*np.random.randint(2) - 2
            if Delta > 0 and (
                    U0 >= max_cfg or U1 >= max_cfg or U2 <= -max_cfg or U3 <= -max_cfg or
                    U4 >= max_cfg or U5 >= max_cfg or U6 <= -max_cfg or U7 <= -max_cfg):
                it.iternext()
                continue
            if Delta < 0 and (
                    U0 <= -max_cfg or U1 <= -max_cfg or U2 >= max_cfg or U3 >= max_cfg or
                    U4 <= -max_cfg or U5 <= -max_cfg or U6 >= max_cfg or U7 >= max_cfg):
                it.iternext()
                continue
            prop_U0 = U0 + Delta
            prop_U1 = U1 + Delta
            prop_U2 = U2 - Delta
            prop_U3 = U3 - Delta
            prop_U4 = U4 + Delta
            prop_U5 = U5 + Delta
            prop_U6 = U6 - Delta
            prop_U7 = U7 - Delta
            new_cfg = np.copy(cfg)
            new_cfg[ind0] = prop_U0
            new_cfg[ind1] = prop_U1
            new_cfg[ind2] = prop_U2
            new_cfg[ind3] = prop_U3
            new_cfg[ind4] = prop_U4
            new_cfg[ind5] = prop_U5
            new_cfg[ind6] = prop_U6
            new_cfg[ind7] = prop_U7
            # print('prop t', prop_U0, prop_U1, prop_U2, prop_U3)
            # print('prop t+1', prop_U4, prop_U5, prop_U6, prop_U7)
            new_S = action_3d(new_cfg, d=d, e2=e2, dt=dt)
            # print(f'Delta S = {new_S - S}')
            if np.random.random() < np.exp(-new_S + S):
                cfg = new_cfg
                S = new_S
                acc += 1
            it.iternext()
    print(f'Acc rate = {100.0*acc/hits:.4f}%')
    check_gauss_law(cfg)
    return cfg, S

def check_gauss_law(cfg):
    out = 0
    Nd = len(cfg.shape) - 1
    for i in range(Nd-1):
        out = out + cfg[i] - np.roll(cfg[i], 1, axis=i)
    assert np.all(out == 0)

def main():
    L = 8
    d = 4
    e2 = 1.0
    dt = 0.25
    mask = get_checkerboard_mask(0, shape=(L,L))
    cfg = np.zeros((2, L, L, L), dtype=int)
    cfg[0][~mask] = -1
    cfg[1][~mask] = 1
    cfg[0][mask] = 1
    cfg[1][mask] = -1
    # cfg = np.ones((2, L, L, L), dtype=int)
    check_gauss_law(cfg)
    S = action_3d(cfg, d=d, e2=e2, dt=dt)
    Ss = [S]
    for i in range(100):
        cfg, S = metropolis_sweep_3d(cfg, d=d, e2=e2, dt=dt)
        Ss.append(S)
    print(Ss)

if __name__ == '__main__': main()
