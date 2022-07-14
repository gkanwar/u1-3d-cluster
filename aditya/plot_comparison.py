import analysis as al
import matplotlib.pyplot as plt
import numpy as np
import os
import paper_plt
paper_plt.load_latex_config()

def load_aditya_file(fname):
    out_vals = []
    with open(fname, 'r') as f:
        next(f) # skip header line
        for line in f:
            row = list(map(float, line.strip().split()))
            out_vals.append(row)
    return np.transpose(out_vals)

def load_sweep(sweep):
    Ls, e2s = sweep
    all_traces = []
    do_bin = lambda x: al.bin_data(x, binsize=20)[1]
    for L in Ls:
        V = L**3
        trace_e2s = []
        trace_M = []
        trace_M2 = []
        trace_MT = []
        trace_MT2 = []
        trace_MC = []
        trace_MC2 = []
        trace_MCsus = []
        trace_E = []
        for e2 in e2s:
            prefix = f'data_tej/obs_trace_{e2:0.2f}_L{L}_cluster'
            if not os.path.exists(prefix + '_E.dat'):
                print(f'Skipping prefix {prefix} (does not exist)')
                continue
            print(f'Loading prefix {prefix}...')
            d = {}
            d['E'] = np.fromfile(prefix + '_E.dat')
            d['M'] = np.fromfile(prefix + '_M.dat')
            d['MT'] = np.fromfile(prefix + '_MT.dat')
            d['MC'] = np.fromfile(prefix + '_MC.dat')
                    
            print(f'... loaded timeseries with {len(d["M"])} pts')
            trace_e2s.append(e2)
            M = al.bootstrap(do_bin(d['M']), Nboot=1000, f=al.rmean)
            M2 = al.bootstrap(do_bin(d['M']**2), Nboot=1000, f=al.rmean)
            MT = al.bootstrap(do_bin(d['MT']), Nboot=1000, f=al.rmean)
            MT2 = al.bootstrap(do_bin(d['MT']**2), Nboot=1000, f=al.rmean)
            MC = al.bootstrap(do_bin(d['MC']), Nboot=1000, f=al.rmean)
            MC2 = al.bootstrap(do_bin(d['MC']**2), Nboot=1000, f=al.rmean)
            
            MCsus = al.bootstrap(
                d['MC'], Nboot=1000,
                f=lambda MC: al.rmean(MC**2) - al.rmean(np.abs(MC))**2)
            E = al.bootstrap(do_bin(d['E']), Nboot=1000, f=al.rmean)
            trace_M.append(M)
            trace_M2.append(M2)
            trace_MT.append(MT)
            trace_MT2.append(MT2)
            trace_MC.append(MC)
            trace_MC2.append(MC2)
            trace_MCsus.append(MCsus)
            trace_E.append(E)
        trace_e2s = np.array(trace_e2s)
        trace_M = np.transpose(trace_M)
        trace_M2 = np.transpose(trace_M2)
        trace_MT = np.transpose(trace_MT)
        trace_MT2 = np.transpose(trace_MT2)
        trace_MC = np.transpose(trace_MC)
        trace_MC2 = np.transpose(trace_MC2)
        trace_MCsus = np.transpose(trace_MCsus)
        trace_E = np.transpose(trace_E)
        all_traces.append({
            'e2': trace_e2s,
            'M': trace_M,
            'M2': trace_M2,
            'MT': trace_MT,
            'MT2': trace_MT2,
            'MC': trace_MC,
            'MC2': trace_MC2,
            'MCsus': trace_MCsus,
            'E': trace_E
        })
    return all_traces


L = 30
V = L**3
Ls = np.array([L])
e2s = np.arange(0.1, 2.0+1e-6, 0.1)
e2s_aditya = np.arange(0.4, 8.0+1e-6, 0.4)
tej_sweep = load_sweep((Ls, e2s))[0]

aditya_all_actions = load_aditya_file('data_aditya/meas_action_L30.txt')
aditya_all_Ms = load_aditya_file('data_aditya/meas_OCT_OT_L30.txt')
aditya_all_M2s = load_aditya_file('data_aditya/meas_OCT2_OT2_L30.txt')
aditya_all_MCs = load_aditya_file('data_aditya/meas_cubeOT_cubeOT2_L30.txt')

assert np.allclose(aditya_all_actions[0], e2s_aditya/8)
assert np.allclose(aditya_all_Ms[0], e2s_aditya/8)
assert np.allclose(aditya_all_M2s[0], e2s_aditya/8)
assert np.allclose(aditya_all_MCs[0], e2s_aditya/8)

aditya_S = aditya_all_actions[1:3]
aditya_M = aditya_all_Ms[1:3]
aditya_MT = aditya_all_Ms[3:5]
aditya_M2 = aditya_all_M2s[1:3]
aditya_MT2 = aditya_all_M2s[3:5]
aditya_MC = aditya_all_MCs[1:3]
aditya_MC2 = aditya_all_MCs[3:5]

style1=dict(marker='o', markersize=6, fillstyle='none', color='xkcd:emerald green')
style2=dict(marker='s', markersize=6, fillstyle='none', color='k')
fig, axes = plt.subplots(2,3, figsize=(8,3.5), squeeze=False)

al.add_errorbar(aditya_M2, xs=e2s_aditya, ax=axes[0,0], **style1,
                label=rf'$L={L}$ (aditya)')
al.add_errorbar(aditya_MT2, xs=e2s_aditya, ax=axes[0,1], **style1)
al.add_errorbar(aditya_S, xs=e2s_aditya, ax=axes[0,2], **style1)
al.add_errorbar(aditya_MC2, xs=e2s_aditya, ax=axes[1,0], **style1)
al.add_errorbar(aditya_MC, xs=e2s_aditya, ax=axes[1,1], **style1)

al.add_errorbar(4*tej_sweep['M2']/V**2, xs=e2s, ax=axes[0,0], **style2,
                label=rf'$L={L}$ (tej)')
al.add_errorbar(16*tej_sweep['MT2']/V**2, xs=e2s, ax=axes[0,1], **style2)
al.add_errorbar(4*tej_sweep['E']/(3*V), xs=e2s, ax=axes[0,2], **style2)
al.add_errorbar(16*tej_sweep['MC2']/V**2, xs=e2s, ax=axes[1,0], **style2)
al.add_errorbar(4*tej_sweep['MC']/V, xs=e2s, ax=axes[1,1], **style2)

axes[0,2].set_ylabel(r'$\left< 4E/3V \right>$')
axes[0,0].set_ylabel(r'$\left< 4 M^2 / V^2 \right>$')
axes[0,1].set_ylabel(r'$\left< 16 M_T^2 / V^2 \right>$')
axes[1,0].set_ylabel(r'$\left< 16 M_C^2 / V^2 \right>$')
axes[1,1].set_ylabel(r'$\left< 4 M_C / V \right>$')

axes[0,0].set_yscale('log')
axes[0,1].set_yscale('log')
axes[0,2].set_yscale('log')
axes[1,0].set_yscale('log')

for ax in axes[1]:
    ax.set_xlabel(r'$e^2$')

handles, labels = axes[0,0].get_legend_handles_labels()
axes[0,2].legend(handles, labels)

fig.set_tight_layout(True)
fig.savefig('figs/comparison_fully_staggered.pdf')
