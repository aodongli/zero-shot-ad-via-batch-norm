import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np

from mvtec import CLASS_NAMES, MVTecDataset

import random
from random import sample
import argparse
from tqdm import tqdm
from collections import OrderedDict
from torch.utils.data import DataLoader
from torchvision.models import wide_resnet50_2, resnet18

# (train_size, num_class, num_image, w, h, num_dim)
# (test_size, num_class, num_image, w, h, num_dim)

use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')

def parse_args():
    parser = argparse.ArgumentParser('ACR_feature_extractor')
    parser.add_argument('--data_path', type=str, default='../data/')
    parser.add_argument('--save_path', type=str, default='../data/mvtec_feature_layer3/')
    parser.add_argument('--arch', type=str, choices=['resnet18', 'wide_resnet50_2'], default='wide_resnet50_2')
    parser.add_argument('--subsample_channel', action='store_true')
    return parser.parse_args()

def main():

    args = parse_args()

    # load model
    if args.arch == 'resnet18':
        model = resnet18(pretrained=True, progress=True)
        t_d = 448
        d = 100
    elif args.arch == 'wide_resnet50_2':
        model = wide_resnet50_2(pretrained=True, progress=True)
        t_d = 1792
        d = 550
    model.to(device)
    model.eval()
    random.seed(1024)
    torch.manual_seed(1024)
    if use_cuda:
        torch.cuda.manual_seed_all(1024)

    idx = torch.tensor(sample(range(0, t_d), d))

    # set model's intermediate outputs
    outputs = []

    def hook(module, input, output):
        outputs.append(output)

    model.layer1[-1].register_forward_hook(hook)
    model.layer2[-1].register_forward_hook(hook)
    model.layer3[-1].register_forward_hook(hook)

    subsample_str = '_subsample' if args.subsample_channel else ''
    folder_path = os.path.join(args.save_path, '%s%s' % (args.arch, subsample_str))
    os.makedirs(folder_path, exist_ok=True)
    # fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    # fig_img_rocauc = ax[0]
    # fig_pixel_rocauc = ax[1]

    # total_roc_auc = []
    # total_pixel_roc_auc = []

    for class_name in CLASS_NAMES:

        train_dataset = MVTecDataset(args.data_path, class_name=class_name, is_train=True)
        train_dataloader = DataLoader(train_dataset, batch_size=32, pin_memory=True, shuffle=False)
        test_dataset = MVTecDataset(args.data_path, class_name=class_name, is_train=False)
        test_dataloader = DataLoader(test_dataset, batch_size=32, pin_memory=True, shuffle=False)

        train_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', [])])
        test_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', [])])

        # extract train set features
        train_feature_filepath = os.path.join(folder_path, 'train_%s.pt' % class_name)
        if not os.path.exists(train_feature_filepath):
            for (x, _, _) in tqdm(train_dataloader, '| feature extraction | train | %s |' % class_name):
                # model prediction
                with torch.no_grad():
                    _ = model(x.to(device))
                # get intermediate layer outputs
                for k, v in zip(train_outputs.keys(), outputs):
                    train_outputs[k].append(v.cpu().detach())
                # initialize hook outputs
                outputs = []
            for k, v in train_outputs.items():
                train_outputs[k] = torch.cat(v, 0)

            # Embedding concat
            # embedding_vectors = train_outputs['layer1']
            # for layer_name in ['layer2', 'layer3']:
            #     embedding_vectors = embedding_concat(embedding_vectors, train_outputs[layer_name])
            # embedding_vectors = embedding_concat(train_outputs['layer2'], train_outputs['layer3'])
            embedding_vectors = train_outputs['layer3']

            # randomly select d dimension
            if args.subsample_channel:
                embedding_vectors = torch.index_select(embedding_vectors, 1, idx)
            # calculate multivariate Gaussian distribution
            B, C, H, W = embedding_vectors.size()
            embedding_vectors = embedding_vectors.view(B, C, H * W)

            with open(train_feature_filepath, 'wb') as f:
                torch.save(embedding_vectors, f)
        else:
            print('load train set feature from: %s' % train_feature_filepath)
            with open(train_feature_filepath, 'rb') as f:
                train_embedding = torch.load(f)


        # extract test set features
        test_feature_filepath = os.path.join(folder_path, 'test_%s.pt' % class_name)
        gt_list = []
        gt_mask_list = []
        if not os.path.exists(test_feature_filepath):
            for (x, y, mask) in tqdm(test_dataloader, '| feature extraction | test | %s |' % class_name):
                gt_list.extend(y.cpu().detach().numpy())
                gt_mask_list.extend(mask.cpu().detach().numpy())

                # model prediction
                with torch.no_grad():
                    _ = model(x.to(device))
                # get intermediate layer outputs
                for k, v in zip(test_outputs.keys(), outputs):
                    test_outputs[k].append(v.cpu().detach())
                # initialize hook outputs
                outputs = []
            for k, v in test_outputs.items():
                test_outputs[k] = torch.cat(v, 0)

            # Embedding concat
            # embedding_vectors = test_outputs['layer1']
            # for layer_name in ['layer2', 'layer3']:
            #     embedding_vectors = embedding_concat(embedding_vectors, test_outputs[layer_name])
            # embedding_vectors = embedding_concat(test_outputs['layer2'], test_outputs['layer3'])
            embedding_vectors = test_outputs['layer3']

            # randomly select d dimension
            if args.subsample_channel:
                embedding_vectors = torch.index_select(embedding_vectors, 1, idx)
            # calculate multivariate Gaussian distribution
            B, C, H, W = embedding_vectors.size()
            embedding_vectors = embedding_vectors.view(B, C, H * W)

            with open(test_feature_filepath, 'wb') as f:
                torch.save(embedding_vectors, f)

            gt_list = np.asarray(gt_list)
            gt_mask_list = np.asarray(gt_mask_list)
            torch.save(gt_list, os.path.join(folder_path, 'test_%s_gt.pt' % class_name))
            torch.save(gt_mask_list, os.path.join(folder_path, 'test_%s_gt_mask.pt' % class_name))
        else:
            print('load train set feature from: %s' % test_feature_filepath)
            with open(test_feature_filepath, 'rb') as f:
                test_embedding = torch.load(f)


def denormalization(x):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    x = (((x.transpose(1, 2, 0) * std) + mean) * 255.).astype(np.uint8)
    return x


def embedding_concat(x, y):
    B, C1, H1, W1 = x.size()
    _, C2, H2, W2 = y.size()
    s = int(H1 / H2)
    x = F.unfold(x, kernel_size=s, dilation=1, stride=s)
    x = x.view(B, C1, -1, H2, W2)
    z = torch.zeros(B, C1 + C2, x.size(2), H2, W2)
    for i in range(x.size(2)):
        z[:, :, i, :, :] = torch.cat((x[:, :, i, :, :], y), 1)
    z = z.view(B, -1, H2 * W2)
    z = F.fold(z, kernel_size=s, output_size=(H1, W1), stride=s)
    return z


if __name__ == '__main__':
    main()