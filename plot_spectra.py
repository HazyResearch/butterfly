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
    name_patterns = ['conv', 'shortcut', 'downsample']
    if dataset == 'cifar10':
        model = torch.load('model_optimizer.pth')['model']
        for n in model.keys():
            if 'weight' in n and any([x in n for x in name_patterns]) and not 'shortcut.1.weight' in n:
                # shortcut.1.* corresponds to the batch norm in the shortcut
                names.append(n)
                tensors.append(model[n].detach().cpu().numpy())
    elif dataset == 'imagenet':
        import torchvision.models as models
        resnet50 = False
        if resnet50:
            model = models.resnet50(pretrained=True)
        else:
            model = models.resnet18(pretrained=True)
        for n, p in model.named_parameters():
            if 'weight' in n and any([x in n for x in name_patterns]) and not 'downsample.1.weight' in n:
                names.append(n)
                tensors.append(p.detach().cpu().numpy())
    elif dataset == 'timit':
        model = torch.load('final_architecture1.pkl')['model_par']
        for n in model.keys():
            if 'ln' not in n and 'bn' not in n and 'bias' not in n:
                names.append(n)
                # Add two dummy dimensions at the end just to simplify later code
                tensors.append(model[n].detach().cpu().unsqueeze_(-1).unsqueeze_(-1).numpy())
    else:
        raise ValueError('Unknown dataset')
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
if dataset == 'timit':
    plot_names.sort(key = lambda s: int("".join(filter(str.isdigit, s))))
print(plot_names)
subplot_grid_dim = int(np.ceil(np.sqrt(len(plot_names))))

figsize = 30 if dataset == 'imagenet' else 20
fig, ax = plt.subplots(int(np.ceil(len(plot_names) / subplot_grid_dim)), subplot_grid_dim, figsize=(28, 28))

for i, n in enumerate(plot_names):
    for j, s in enumerate(spectra[n]):
        xvals = np.arange(len(s)) + 1
        subplot = ax[i // subplot_grid_dim, i % subplot_grid_dim]
        subplot.plot(xvals, s, label=n + '_filter{}'.format(j))
        if len(s) <= 5:
            subplot.set_xticks(xvals)
        else:
            subplot.set_xticks([1, len(s)//2, len(s)])
        subplot.tick_params(axis='x', labelsize=6)
        subplot.tick_params(axis='y', labelsize=6)
        subplot.set_xlabel('Index', fontsize=7)
        subplot.set_ylabel('Singular value', fontsize=7)
        subplot.set_title(n, fontsize=10)
fig.suptitle(dataset.capitalize(), fontsize=24)
fig.savefig('spectra_{}.png'.format(dataset), dpi=250)
