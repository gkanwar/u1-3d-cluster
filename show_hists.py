import analysis as al
import itertools
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
import pyvista.examples
import scipy as sp
import scipy.interpolate
import scipy.optimize
import scipy.ndimage
import tqdm

def plot_hists(Ls, e2s, M_hists):
    keys = ['M', 'MT']
    for k in keys:
        with PdfPages(f'figs/all_{k}_hists.pdf') as pdf:
            for i,e2 in enumerate(e2s):
                fig, ax = plt.subplots(1,1)
                bins = np.linspace(-0.20, 0.20, num=101, endpoint=True)
                for j,L in enumerate(Ls):
                    obs = M_hists[j][i]
                    if obs is None: continue
                    ax.hist(
                        obs[k], bins=bins, histtype='stepfilled', alpha=0.5,
                        label=f'L = {L}')
                ax.legend()
                fig.suptitle(f'e2 = {e2:0.2f}')
                pdf.savefig(fig)
                plt.close(fig)
        with PdfPages(f'figs/all_{k}_traces.pdf') as pdf:
            for i,e2 in enumerate(e2s):
                count = 0
                for j in range(len(Ls)):
                    if M_hists[j][i] is not None: count += 1
                fig, axes = plt.subplots(count, 1)
                ax_ind = 0
                for j,L in enumerate(Ls):
                    obs = M_hists[j][i]
                    if obs is None: continue
                    ax = axes[ax_ind]
                    ax.plot(obs[k], label=f'L = {L}', color=f'C{ax_ind}')
                    ax.legend()
                    ax_ind += 1
                fig.suptitle(f'e2 = {e2:0.2f}')
                pdf.savefig(fig)
                plt.close(fig)
                    

def main():
    e2s = np.arange(0.20, 1.30+1e-6, 0.1)
    Ls = np.array([8, 16, 32, 64, 96, 128, 192, 256])
    M_hists = [[None for _ in e2s] for _ in Ls]
    for i,e2 in enumerate(e2s):
        for j,L in enumerate(Ls):
            try:
                obs = np.load(f'raw_obs/obs_trace_{e2:0.2f}_L{L}_mixed.npy', allow_pickle=True).item()
                M_hists[j][i] = obs
            except Exception as e:
                print(f'Skipping ({e2:0.2f},{L}) ({e})')
    plot_hists(Ls, e2s, M_hists)

if __name__ == '__main__': main()
