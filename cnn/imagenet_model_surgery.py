import os, sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
# Add to $PYTHONPATH in addition to sys.path so that ray workers can see
os.environ['PYTHONPATH'] = project_root + ":" + os.environ.get('PYTHONPATH', '')

import argparse
import torchvision.models as torch_models
import models.resnet_imagenet as models # only use imagenet models
import torch
import logging
from collections import OrderedDict
import torchvision.datasets as datasets
import time

from train_utils import AverageMeter, data_prefetcher
import train_utils
def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--num_structured_layers', default=0, type=int,
                        help='Number of layers starting from the back that are structured')
    parser.add_argument('--structure_type', default='B', type=str, choices=['B', 'BBT', 'BBTBBT'],
                        help='Structure of butterflies')
    parser.add_argument('--nblocks', default=1, type=int, help='Number of blocks for each butterfly')
    parser.add_argument('--param', default='regular', type=str, help='Parametrization of butterfly factors')
    parser.add_argument('--input-dir', default='/distillation/resnet18/butterflies', type=str, help='Input directory for distilled butterflies')
    parser.add_argument('--output-dir', default='.', help='Output directory for initialized resnets with butterflies.')
    parser.add_argument('--random', action='store_true', help='Use randomly initialized butterflies so don\'t load weights for butterflies')
    return parser

# from imagenet_experiment import get_loaders, validate
args = get_parser().parse_args()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.info(args)
os.makedirs(args.output_dir, exist_ok=True)

# initialize new model
student_model = models.__dict__['resnet18'](num_structured_layers=args.num_structured_layers,
    structure_type=args.structure_type, nblocks=args.nblocks, param=args.param)
print(student_model.state_dict().keys())

checkpoint = torch.load("resnet18.pth.tar", map_location = lambda storage, loc: storage.cuda())
best_acc1 = checkpoint['best_prec1']
loaded_state_dict = train_utils.strip_prefix_if_present(checkpoint['state_dict'], prefix="module.")

if not args.random:
    # add butterflies
    if args.structure_type == "B":
        loaded_state_dict['features.7.0.conv1.twiddle'] = torch.load(
            f"{args.input_dir}/butterfly_features.7.0.conv1_B_1_regular.pt"
            ).twiddle
        loaded_state_dict['features.7.0.conv2.twiddle'] = torch.load(
            f"{args.input_dir}/butterfly_features.7.0.conv2_B_1_regular.pt"
            ).twiddle
        loaded_state_dict['features.7.0.downsample.0.twiddle'] = torch.load(
            f"{args.input_dir}/butterfly_features.7.0.downsample.0_B_1_regular.pt"
            ).twiddle
        loaded_state_dict['features.7.1.conv1.twiddle'] = torch.load(
            f"{args.input_dir}/butterfly_features.7.1.conv1_B_1_regular.pt"
        ).twiddle
        loaded_state_dict['features.7.1.conv2.twiddle'] = torch.load(
        f"{args.input_dir}/butterfly_features.7.1.conv2_B_1_regular.pt"
        ).twiddle

        # also load features.6
        if args.num_structured_layers == 2:
            loaded_state_dict['features.6.0.conv1.twiddle'] = torch.load(
                f"{args.input_dir}/butterfly_features.6.0.conv1_B_1_regular.pt"
                ).twiddle
            loaded_state_dict['features.6.0.conv2.twiddle'] = torch.load(
                f"{args.input_dir}/butterfly_features.6.0.conv2_B_1_regular.pt"
                ).twiddle
            loaded_state_dict['features.6.0.downsample.0.twiddle'] = torch.load(
                f"{args.input_dir}/butterfly_features.6.0.downsample.0_B_1_regular.pt"
                ).twiddle
            loaded_state_dict['features.6.1.conv1.twiddle'] = torch.load(
                f"{args.input_dir}/butterfly_features.6.1.conv1_B_1_regular.pt"
            ).twiddle
            loaded_state_dict['features.6.1.conv2.twiddle'] = torch.load(
            f"{args.input_dir}/butterfly_features.6.1.conv2_B_1_regular.pt"
            ).twiddle

    elif args.structure_type == 'BBT':
        features_7_0_conv1 = torch.load(
                    f"{args.input_dir}/butterfly_features.7.0.conv1_BBT_{args.nblocks}_regular.pt"
                    )
        features_7_0_conv2 = torch.load(
            f"{args.input_dir}/butterfly_features.7.0.conv2_BBT_{args.nblocks}_regular.pt")
        features_7_0_downsample_0 = torch.load(
            f"{args.input_dir}/butterfly_features.7.0.downsample.0_BBT_{args.nblocks}_regular.pt")
        features_7_1_conv1 = torch.load(
            f"{args.input_dir}/butterfly_features.7.1.conv1_BBT_{args.nblocks}_regular.pt")
        features_7_1_conv2 = torch.load(
            f"{args.input_dir}/butterfly_features.7.1.conv2_BBT_{args.nblocks}_regular.pt")
        # nblocks doubled since BBT
        for i in range(args.nblocks*2):
            loaded_state_dict[f'features.7.0.conv1.layers.{i}.twiddle'] = features_7_0_conv1.layers[i].twiddle
            loaded_state_dict[f'features.7.0.conv2.layers.{i}.twiddle'] = features_7_0_conv2.layers[i].twiddle
            loaded_state_dict[f'features.7.0.downsample.0.layers.{i}.twiddle'] = features_7_0_downsample_0.layers[i].twiddle
            loaded_state_dict[f'features.7.1.conv1.layers.{i}.twiddle'] = features_7_1_conv1.layers[i].twiddle
            loaded_state_dict[f'features.7.1.conv2.layers.{i}.twiddle'] = features_7_1_conv2.layers[i].twiddle

    else:
        raise ValueError("Invalid structure type")

# delete replaced keys so we can use strict loading
del loaded_state_dict["features.7.0.conv1.weight"]
del loaded_state_dict["features.7.0.conv2.weight"]
del loaded_state_dict["features.7.1.conv1.weight"]
del loaded_state_dict["features.7.1.conv2.weight"]
del loaded_state_dict["features.7.0.downsample.0.weight"]
if args.num_structured_layers == 2:
    del loaded_state_dict["features.6.0.conv1.weight"]
    del loaded_state_dict["features.6.0.conv2.weight"]
    del loaded_state_dict["features.6.1.conv1.weight"]
    del loaded_state_dict["features.6.1.conv2.weight"]
    del loaded_state_dict["features.6.0.downsample.0.weight"]

if not args.random:
    # keep strict=True to make sure keys aren't messed up in naming and weights are actually loaded
    student_model.load_state_dict(loaded_state_dict, strict=True)
else:
    # reuse randomly initialized butterfly weights (so they aren't keys)
    student_model.load_state_dict(loaded_state_dict, strict=False)

# add module prefix bc imagenet training code will expect it for loading model weights
student_model_dict = train_utils.add_prefix(student_model.state_dict(), prefix="module.")
state = {'state_dict': student_model_dict,
         'best_acc1': 0,
         'epoch': 0}

# save model state for loading into code for fine-tuning
if args.random:
    torch.save(state, f"{args.output_dir}/resnet18_butterfly_random_{args.structure_type}_{args.nblocks}_{args.param}.pth.tar")
else:
    torch.save(state, f"{args.output_dir}/resnet18_butterfly_distilled_{args.structure_type}_{args.nblocks}_{args.param}.pth.tar")
