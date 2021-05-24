#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 Takuma Yagi <tyagi@iis.u-tokyo.ac.jp>
#
# Distributed under terms of the MIT license.

# from models.resnet_simclr import ResNetSimCLR
from torchvision.models import resnet18, resnet50
import warnings
import argparse
import os
import os.path as osp
import time
import datetime
import json

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader

import torchvision.transforms as transforms
from torchvision.datasets import DatasetFolder
from torchvision import models

from PIL import Image
import matplotlib.pyplot as plt
import sklearn.metrics as metrics

from data_loader import EvalDataset

from mllogger import MLLogger
logger = MLLogger(init=False)


warnings.simplefilter('ignore', UserWarning)

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))


def loader_func(impath):
    """
    Returns RGB normalized image
    """
    img = Image.open(impath).convert("RGB")

    trans_func = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return {
        "imgs": trans_func(img)
    }


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', type=str, default="outputs")
    parser.add_argument('--dir_name', type=str, default="tmp")
    parser.add_argument('--nb_workers', type=int, default=8)

    parser.add_argument('--model', type=str, default="resnet18")

    # parser.add_argument('--pred_file', required=True, type=str)
    parser.add_argument('--gt_file', required=True, type=str)
    parser.add_argument('--k_vals', default=1, nargs='+', type=int)
    parser.add_argument('--score_th', default=0.5, type=float)

    parser.add_argument('--nb_iters', type=int, default=481416 // 64 * 30)  # About 50000 iteratio
    parser.add_argument('--iter_evaluation', type=int, default=481416 // 64)
    parser.add_argument('--iter_snapshot', type=int, default=481416 // 64)
    parser.add_argument('--iter_display', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--optimizer', type=str, default="sgd")
    parser.add_argument('--min_lr', type=float, default=3e-8)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--lr_step_list', type=float, nargs="*", default=[30000, 60000])

    parser.add_argument('--resume', type=str, default="/home/ubuntu/SimCLR/runs/May11_13-32-51_ip-172-31-26-32/checkpoint_0200.pth.tar")
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--vis', action='store_true')
    parser.add_argument('--save_model', action='store_true')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--seed', type=int, default=1701)  # XXX
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                        choices=model_names,
                        help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')
    parser.add_argument('--out_dim', default=128, type=int,
                        help='feature dimension (default: 128)')
    args = parser.parse_args()

    return args


def eval_net(args, device, save_dir, model, valid_dataset):

    predictions = []

    if args.vis:
        vis_dir = osp.join(save_dir, args.vis_dir_name)
        os.makedirs(vis_dir, exist_ok=True)

    model.eval()
    with torch.no_grad():
        valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, num_workers=args.nb_workers)

        loss, total_acc, total_cnt = 0.0, 0.0, 0.0
        criterion = nn.CrossEntropyLoss()
        y_trues, y_preds = [], []

        thresholds = np.arange(0.0, 1.1, 0.1)
        acc_list = [0.0 for x in range(len(thresholds))]
        y_pred_list = [[] for x in range(len(thresholds))]

        for valid_cnt, (index, batch, labels) in enumerate(valid_loader):

            # import pdb;pdb.set_trace()
            batch, labels = batch.to(device), labels.to(device)
            out = model(batch)
            out_score = F.softmax(out, dim=1)

            for idx, th in enumerate(thresholds):
                out_openset = torch.zeros((out.shape[0], out.shape[1] + 1), dtype=torch.float32).to(device)
                mask = out_score.max(dim=1)[0] > th
                out_openset[mask, 1:] = out[mask]
                out_openset[~mask, 0] = torch.ones(len(batch), dtype=torch.float32).to(device)[~mask]
                y_pred = torch.argmax(out_openset, dim=1)
                acc_list[idx] += torch.sum(y_pred == labels).item()
                y_pred_list[idx].extend(y_pred.tolist())

            out_openset = torch.zeros((out.shape[0], out.shape[1] + 1), dtype=torch.float32).to(device)
            mask = out_score.max(dim=1)[0] > args.score_th
            out_openset[mask, 1:] = out[mask]
            out_openset[~mask, 0] = torch.ones(len(batch), dtype=torch.float32).to(device)[~mask]

            loss += criterion(out_openset, labels).item() * len(batch)
            y_pred = torch.argmax(out_openset, dim=1)
            acc = torch.sum(y_pred == labels).item() / len(batch)
            total_acc += acc * len(batch)
            total_cnt += len(batch)
            y_trues.extend(labels.tolist())
            y_preds.extend(y_pred.tolist())

            pred_dict = {
                "y_logit": out_openset.tolist(),
                "y_pred": y_pred.tolist(),
                "y_true": labels.tolist(),
                "acc": acc,
            }
            predictions.append(pred_dict)

            # if args.vis:
            #   offset = valid_cnt * args.batch_size
            #   vis_predictions(vis_dir, valid_dataset, offset, pred_dict, args.fps, args.ratio)

    acc_list = np.array(acc_list) / total_cnt
    f_score_list = np.array([metrics.f1_score(y_trues, x, average='macro') for x in y_pred_list])

    logger.info("Loss: {}".format(loss / total_cnt))
    logger.info("Accuracy: {}".format(total_acc / total_cnt))
    logger.info("F-score: {}".format(metrics.f1_score(y_trues, y_preds, average='macro')))
    logger.info(np.round(total_acc / total_cnt, 3))
    logger.info(acc_list.tolist())
    model.train()

    plt.figure()
    plt.plot(thresholds, acc_list, label="Accuracy")
    plt.plot(thresholds, f_score_list, label="F-score")
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.title("{} {}".format(args.model, osp.basename(args.gt_file)))
    plt.xlabel("Threhold")
    plt.legend(bbox_to_anchor=(0, 0), loc='lower left', borderaxespad=1)
    # plt.ylabel("Accuracy")
    for idx, x in enumerate(thresholds):
        plt.text(x, acc_list[idx], np.round(acc_list[idx], 3), ha="center", va="bottom")
        plt.text(x, f_score_list[idx], np.round(f_score_list[idx], 3), ha="center", va="bottom")
    plt.savefig(osp.join(save_dir, "acc_{}.jpg".format(osp.basename(args.gt_file))))
    plt.close()

    return predictions


def main():
    args = get_args()
    if args.dir_name == "tmp":
        dir_name = "{}_{}_{}".format(args.model, args.dir_name, datetime.datetime.now().strftime('%y%m%d'))
    else:
        dir_name = "{}_{}_{}".format(args.model, args.dir_name, datetime.datetime.now().strftime('%y%m%d_%H%M%S'))
    if args.eval and args.out_dir == "outputs":
        args.out_dir = "predictions"
    logger.initialize(args.out_dir, dir_name)
    logger.info(vars(args))
    save_dir = logger.get_savedir()
    logger.info("Written to {}".format(save_dir))

    device = torch.device("cuda:{}".format(0)) if torch.cuda.is_available() else torch.device("cpu")

    root_train_dir = "data/dataset-open-world-vision-challenge/train/known_classes/"
    root_valid_dir = "data/dataset-open-world-vision-challenge/"
    train_dataset = DatasetFolder(root_train_dir, loader_func, extensions=IMG_EXTENSIONS)
    valid_dataset = EvalDataset(args.gt_file, root_valid_dir, header=True)

    logger.info(f"Train set size: {len(train_dataset)}")
    logger.info(f"Valid set size: {len(valid_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.nb_workers, drop_last=True)

    if args.model == "resnet18":
        base_model = resnet18(pretrained=False, num_classes=413)
    elif args.model == "resnet50":
        base_model = resnet50(pretrained=False, num_classes=413)

    # base_model = ResNetSimCLR(base_model=args.arch, out_dim=args.out_dim)

    logger.info("Model: {}".format(base_model.__class__.__name__))
    logger.info("Output dir: {}".format(save_dir))
    if not args.eval and not args.save_model:
        logger.info("NOT saving model! Add \"--save_model\" if you want to save your model")

    if args.resume != "":
        base_model.load_state_dict(torch.load(args.resume), strict=False)
        print("Load pretrained weight.")
    # base_model.fc = nn.Linear(512, 413)
    base_model.to(device)
    model = base_model  # XXX No data parallel so far

    start_time = time.time()

    if not args.eval:
        criterion = nn.CrossEntropyLoss()

        if args.optimizer == "adam":
            optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4, betas=(0.9, 0.999))
        elif args.optimizer == "sgd":
            optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.99, weight_decay=1e-4)
        else:
            raise NotImplementedError()

        logger.info("Optimizer: {}".format(args.optimizer))
        scheduler = MultiStepLR(optimizer, args.lr_step_list, 0.1)

        epoch_cnt, iter_cnt = 1, 0
        loss_elapsed = []

        st = time.time()
        while iter_cnt != args.nb_iters and optimizer.param_groups[0]['lr'] > args.min_lr:
            print("")
            logger.info("Epoch {}".format(epoch_cnt))

            for cnt, (batch, labels) in enumerate(train_loader):
                if iter_cnt == args.nb_iters:
                    break

                batch = dict([(k, v.to(device)) if type(v) != list else (k, v) for k, v in batch.items()])
                labels = labels.to(device)

                optimizer.zero_grad()

                out = model(batch["imgs"])
                loss = criterion(out, labels)

                if args.debug and iter_cnt % 10 == 0:
                    print(np.round(loss.item(), 3))
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, norm_type=2)
                optimizer.step()
                scheduler.step()
                loss_elapsed.append(loss.item())

                if (iter_cnt + 1) % args.iter_display == 0:
                    logger.info("Iter {}: {}, {} iter / s".format(iter_cnt + 1, np.mean(loss_elapsed), args.iter_display / (time.time() - st)))
                    loss_elapsed = []
                    st = time.time()

                if (iter_cnt + 1) % args.iter_snapshot == 0:
                    model_path = osp.join(save_dir, "model_{:06d}.pth".format(iter_cnt + 1))
                    logger.info("Checkpoint: {}".format(model_path))
                    torch.save(base_model.state_dict(), model_path)

                if args.iter_evaluation != -1 and (iter_cnt + 1) % args.iter_evaluation == 0:
                    logger.info("Validation...")
                    eval_net(args, device, save_dir, model, valid_dataset)
                    st = time.time()

                iter_cnt += 1

            epoch_cnt += 1

    else:  # Evaluation
        predictions = eval_net(args, device, save_dir, model, valid_dataset)
        pred_path = osp.join(save_dir, "predictions.json")
        with open(pred_path, "w") as f:
            json.dump(predictions, f)
    logger.info("Done. Elapsed time: {} (s)".format(time.time() - start_time))


if __name__ == "__main__":
    main()
