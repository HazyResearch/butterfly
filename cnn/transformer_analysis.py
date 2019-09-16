import pickle
import json
from pathlib import Path
import numpy as np

butterfly_bleu = [32.99, 33.8, 34.32, 34.3, 34.23, 34.1]
lr_bleu = [30.05, 32.71, 33.6, 33.08, 34.15, 34.3]
param_fraction = np.array([9, 18, 36, 72, 108, 128]) / 128

import matplotlib.pyplot as plt
plt.switch_backend('agg')

markers = ['o', 'v', 'D', 'p', 's', '>']
colors = ['red', 'orange', 'green', 'blue', 'black']

plt.plot(param_fraction, butterfly_bleu, marker=markers[0], color=colors[0], label='Butterfly')
plt.plot(param_fraction, lr_bleu, marker=markers[2], color=colors[2], label='Low-rank')
# plt.plot(mobilenet_numparams / toeplitzlike_numparams[1:], all_accuracy[2, 1:], marker=markers[2], color=colors[2], label='Toeplitz-like')
# plt.plot(mobilenet_numparams / resnet_numparams, resnet_accuracy, marker=markers[3], color=colors[3], label='Resnet18')
# plt.plot(1, butterfly_bleu, marker=markers[4], color=colors[4], label='MobileNet')
# plt.axhline(butterfly_bleu, color='black', linewidth=3)
ax = plt.gca()
# ax.text(0.55, butterfly_bleu + 0.005, 'MobileNet 3.2M', color=colors[4])
# ax.text(0.55 * mobilenet_numparams / resnet_numparams, resnet_accuracy - 0.0075, 'ResNet18 11M', color=colors[3])
# ax.text(0.55 * mobilenet_numparams / all_params[6], all_accuracy[0, 6] + 0.005, 'Structured 0.6M', color=colors[0])
# plt.xscale('log')
plt.xlabel('Fraction query/key projection params vs standard attention', fontsize=14)
plt.ylabel('BLEU', fontsize=14)
# plt.xticks([0.3, 1, 2, 5, 10, 25], [0.3, 1, 2, 5, 10, 25])
# plt.xticks([1, 2, 5, 10, 25], [1, 2, 5, 10, 25])

# plt.legend(['Butterfly', 'Circulant', 'Toeplitz-like'])
plt.legend(['Butterfly', 'Low-rank'])
plt.savefig('transformer_compression.pdf', bbox_inches='tight')
