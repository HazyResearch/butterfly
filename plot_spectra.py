import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import sys

dataset = 'cifar10' if len(sys.argv) < 2 else sys.argv[1]
svds_file = 'svds_{}.pkl'.format(dataset)

if os.path.exists(svds_file):
    with open(svds_file, 'rb') as pkl: spectra = pickle.load(pkl)
else:
    names = []
    tensors = []
    name_patterns = ['conv', 'shortcut']
    if dataset == 'cifar10':
        model = torch.load('model_optimizer.pth')['model']
        for n in model.keys():
            if 'weight' in n and any([x in n for x in name_patterns]) and not 'shortcut.1.weight' in n:
                # shortcut.1.* corresponds to the batch norm in the shortcut
                names.append(n)
                tensors.append(model[n].detach().cpu().numpy())
    elif dataset == 'imagenet':
        import torchvision.models as models
        model = models.resnet50(pretrained=True)
        for n, p in model.named_parameters():
            if 'weight' in n and any([x in n for x in name_patterns]):
                names.append(n)
                tensors.append(p.detach().cpu().numpy())
    spectra = {}
    for n, t in zip(names, tensors):
        t_spectra = []
        print('Processing {}...'.format(n))
        for a in range(t.shape[-2]):
            for b in range(t.shape[-1]):
                f = t[:,:,a,b]
                s = np.linalg.svd(f, compute_uv=False)
                t_spectra.append(s)
        spectra[n] = t_spectra
    with open(svds_file, 'wb') as pkl: pickle.dump(spectra, pkl)

plot_names = sorted(list(spectra.keys()))
print(plot_names)
subplot_grid_dim = int(np.ceil(np.sqrt(len(plot_names))))

fig, ax = plt.subplots(int(np.ceil(len(plot_names) / subplot_grid_dim)), subplot_grid_dim, figsize=(20, 20))

for i, n in enumerate(plot_names):
    for j, s in enumerate(spectra[n]):
        xvals = np.arange(len(s)) / len(s)
        subplot = ax[i // subplot_grid_dim, i % subplot_grid_dim]
        subplot.plot(xvals, s, label=n + '_filter{}'.format(j))
        subplot.tick_params(axis='y', labelsize=6)
        subplot.axes.get_xaxis().set_visible(False)
        subplot.set_title(n, fontsize=10)
fig.suptitle(dataset.capitalize())
fig.savefig('spectra_{}.png'.format(dataset), dpi=250)
