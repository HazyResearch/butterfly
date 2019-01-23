import pickle
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.patches as mpatches
plt.rcParams['font.family'] = 'serif'

rs = [1]
markers = ['o', 'v', 'D', 'p', 's', '>']
loc = 'speed_data.pkl'
data = pickle.load(open(loc,'rb'))

colors = ['red', 'orange', 'green', 'blue']

speedups_fft = data['speedup_fft']
speedups_dct = data['speedup_dct']
speedups_dst = data['speedup_dst']
speedups_bp = data['speedup_bp']
sizes = data['sizes']
lw = 3
msize = 6

print('data: ', data)

start_idx = 0

print('fft speedup: ', speedups_fft[start_idx:])
print('dct speedup: ', speedups_dct[start_idx:])
print('dst speedup: ', speedups_dst[start_idx:])
print('bp speedup: ', speedups_bp[start_idx:])

print('sizes, speedups: ', sizes.size, speedups_fft.shape)
plt.plot(sizes[start_idx:],speedups_fft[start_idx:], linewidth=lw, label='FFT',marker=markers[0],color=colors[0],
    markeredgecolor=colors[0],markersize=msize)
plt.plot(sizes[start_idx:],speedups_dct[start_idx:], linewidth=lw, label='DCT',marker=markers[0],color=colors[1],
    markeredgecolor=colors[1],markersize=msize)
plt.plot(sizes[start_idx:],speedups_dst[start_idx:], linewidth=lw, label='DST',marker=markers[0],color=colors[2],
    markeredgecolor=colors[2],markersize=msize)
plt.plot(sizes[start_idx:],speedups_bp[start_idx:], linewidth=lw, label='BP',marker=markers[0],color=colors[3],
    markeredgecolor=colors[3],markersize=msize)

plt.axhline(y=1.0, color='black',linewidth=3)
plt.xscale('log', basex=2)
plt.yscale('log')
plt.xlabel(r'$N$',fontsize=14)
plt.ylabel("Speedup over GEMV", fontsize=14)

classes = [mpatches.Patch(color=colors[0], label='FFT'),
           mpatches.Patch(color=colors[1], label='DCT'),
           mpatches.Patch(color=colors[2], label='DST'),
           mpatches.Patch(color=colors[3], label='BP')]

#ranks_row1 = []
rank_entries = []
#marker_entries = {}

# for idx_r, r in enumerate(rs):
#     this_marker = plt.scatter([], [], marker=markers[idx_r], edgecolor='black', facecolor='black',label='Rank ' + str(r))
#     rank_entries.append(this_marker)
#     #marker_entries[idx_r] = this_marker

"""
fake_marker = plt.scatter([], [], marker=markers[idx_r], edgecolor='white', facecolor='white',label="       ")

ranks_row1.append(marker_entries[0])
ranks_row1.append(fake_marker)
ranks_row1.append(marker_entries[1])
ranks_row1.append(fake_marker)
ranks_row1.append(marker_entries[2])

ranks_row2 = [marker_entries[3],marker_entries[4]]
"""

plt.legend(handles=classes, ncol=4, bbox_to_anchor=(0.85, -0.15))#, loc='upper left')
#plt.legend(handles=rank_entries, ncol=4, bbox_to_anchor=(1.02, -0.2))#, loc='upper left')
#plt.legend(handles=ranks_row1, ncol=3, bbox_to_anchor=(0.85, -0.25))#, loc='upper left')
#l2 = plt.legend(handles=ranks_row1, ncol=3, bbox_to_anchor=(0.85, -0.25))
#plt.legend(handles=ranks_row2, ncol=2, bbox_to_anchor=(0.75, -0.25),frameon=False)#, loc='upper left')
#fig.axes.add_artist(l1)
#plt.gca().add_artist(l1)
#plt.gca().add_artist(l2)
# Legend: Show ranks
# Legend: Show Toeplitz-like, LDR-SD

plt.savefig('speed_plot.pdf',bbox_inches='tight')
