import pickle
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.patches as mpatches
plt.rcParams['font.family'] = 'serif'

rs = [1]
markers = ['o', 'v', 'D', 'p', 's', '>']
loc = 'speed_training_data.pkl'
data = pickle.load(open(loc,'rb'))

colors = ['red', 'orange', 'green', 'blue']

speedups_fft = data['speedup_fft']
speedups_butterfly = data['speedup_butterfly']
sizes = data['sizes']
lw = 3
msize = 6

print('data: ', data)

start_idx = 0

print('fft speedup: ', speedups_fft[start_idx:])
print('butterfly speedup: ', speedups_butterfly[start_idx:])

print('sizes, speedups: ', sizes.size, speedups_fft.shape)
plt.plot(sizes[start_idx:],speedups_fft[start_idx:], linewidth=lw, label='FFT',marker=markers[0],color=colors[0],
    markeredgecolor=colors[0],markersize=msize)
plt.plot(sizes[start_idx:],speedups_butterfly[start_idx:], linewidth=lw, label='Butterfly',marker=markers[0],color=colors[3],
    markeredgecolor=colors[3],markersize=msize)

plt.axhline(y=1.0, color='black',linewidth=3)
plt.xscale('log', basex=2)
plt.yscale('log')
plt.xlabel(r'$N$',fontsize=14)
plt.ylabel("Speedup over GEMM", fontsize=14)

classes = [mpatches.Patch(color=colors[0], label='FFT'),
           mpatches.Patch(color=colors[3], label='Butterfly')]

plt.legend(handles=classes, ncol=4, bbox_to_anchor=(0.75, -0.15))#, loc='upper left')

plt.savefig('speed_training_plot.pdf', bbox_inches='tight')
