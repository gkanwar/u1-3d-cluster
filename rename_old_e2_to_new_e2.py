import numpy as np
import os
import shutil

def rename(e2, L, *, old_prefix, new_prefix):
    new_e2 = 2 * e2
    old_fname = f'{old_prefix}_{e2:0.2f}_L{L:d}_mixed.npy'
    new_fname = f'{new_prefix}_{new_e2:0.2f}_L{L:d}_mixed.npy'
    if not os.path.exists(old_fname):
        return # origin file does not exist
    print(f'mv {old_fname} {new_fname}')
    if os.path.exists(new_fname):
        print('Target file exists... skipping')
        return
    shutil.copyfile(old_fname, new_fname)

def rename_obs_files():
    all_e2 = np.arange(0.20, 1.30+1e-6, 0.05)
    for e2 in all_e2:
        for L in [8, 16, 32, 64]:
            rename(e2, L, old_prefix='raw_obs_old_e2/obs', new_prefix='raw_obs/obs')
            rename(e2, L, old_prefix='raw_obs_old_e2/obs_trace', new_prefix='raw_obs/obs_trace')

def rename_cfg_files():
    all_e2 = np.arange(0.25, 1.50+1e-6, 0.01)
    for e2 in all_e2:
        for L in [8, 16, 32, 64]:
            rename(e2, L, old_prefix='./ens', new_prefix='cfgs/ens')
        
if __name__ == '__main__':
    rename_obs_files()
    rename_cfg_files()
