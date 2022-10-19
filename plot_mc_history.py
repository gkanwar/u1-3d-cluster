import analysis as al
import argparse
import matplotlib.pyplot as plt
import numpy as np

def compute_rho(ys, *, tmax=10000):
    rho = []
    for t in range(tmax):
        if t > 0:
            rho.append(np.mean(ys[t:] * ys[:-t]))
        else:
            rho.append(np.mean(ys[t:]**2))
    rho = np.array(rho)
    rho /= rho[0]
    return rho

def compute_tint(ys):
    tint = np.cumsum(np.insert(compute_rho(ys)[1:], 0, 0.5))
    # fig, ax = plt.subplots(1,1)
    # ax.plot(tint)
    # fig.set_tight_layout(True)
    # fig.savefig('tmp_tint.pdf')
    return tint

def do_plot(*, fname, label, binsize, out_fname):
    ys = np.fromfile(fname)
    # print('tint', compute_tint(ys))
    xs, ys = al.bin_data(ys, binsize=binsize)
    if len(ys) < 1000:
        marker = '.'
    else:
        marker = ''
    fig, ax = plt.subplots(1,1, figsize=(6,2))
    ax.plot(xs, ys, marker=marker)
    ax.set_xlabel('MC step')
    ax.set_ylabel(label)
    fig.set_tight_layout(True)
    fig.savefig(out_fname)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fname', type=str, required=True)
    parser.add_argument('--out_fname', type=str, required=True)
    parser.add_argument('--label', type=str, required=True)
    parser.add_argument('--binsize', type=int, default=1)
    args = parser.parse_args()
    do_plot(**vars(args))

if __name__ == '__main__': main()
