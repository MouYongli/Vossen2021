import datetime
import math
import os
import os.path as osp
import shutil

import numpy as np
import pytz
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import tqdm


def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) +
        label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)
    return hist


def label_accuracy_score(label_trues, label_preds, n_class=8):
    hist = np.zeros((n_class, n_class))
    hist += _fast_hist(label_trues, label_preds, n_class)
    acc = np.diag(hist).sum() / hist.sum()
    with np.errstate(divide='ignore', invalid='ignore'):
        precision = np.diag(hist) / hist.sum(axis=1)
    mean_precision = np.nanmean(precision)
    with np.errstate(divide='ignore', invalid='ignore'):
        recall = np.diag(hist) / hist.sum(axis=0)
    mean_recall = np.nanmean(recall)
    with np.errstate(divide='ignore', invalid='ignore'):
        iou = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    mean_iou = np.nanmean(iou)
    with np.errstate(divide='ignore', invalid='ignore'):
        f1 = (2 * np.diag(hist))/ (hist.sum(axis=1) + hist.sum(axis=0) + 2 * np.diag(hist))
    mean_f1 = np.nanmean(f1)
    return acc, mean_precision, mean_recall, mean_iou, mean_f1



class Trainer(object):
    def __init__(self, cuda, model, optim, loss_fn, train_loader, val_loader, experiment_dir, epochs):
        self.cuda = cuda
        self.model = model
        self.optim = optim
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.timestamp_start = datetime.datetime.now(pytz.timezone('Asia/Tokyo'))
        self.experiment_dir = experiment_dir
        if not osp.exists(osp.join(self.experiment_dir, 'train.csv')):
            with open(osp.join(self.experiment_dir, 'train.csv'), 'w') as f:
                train_log_headers = ['epoch', 'iteration', 'train/loss', 'train/acc', 'train/precision', 'train/recall', 'train/iu', 'train/f1', 'elapsed_time',]
                f.write(','.join(train_log_headers) + '\n')
        if not osp.exists(osp.join(self.experiment_dir, 'valid.csv')):
            with open(osp.join(self.experiment_dir, 'valid.csv'), 'w') as f:
                valid_log_headers = ['epoch', 'valid/loss', 'valid/acc', 'valid/precision', 'valid/recall', 'valid/iu',
                                     'valid/f1', 'elapsed_time']
                f.write(','.join(valid_log_headers) + '\n')
        if not osp.exists(osp.join(self.experiment_dir, 'test.csv')):
            with open(osp.join(self.experiment_dir, 'test.csv'), 'w') as f:
                test_log_headers = ['loss', 'acc', 'precision', 'recall', 'iu', 'f1', 'elapsed_time']
                f.write(','.join(test_log_headers) + '\n')
        self.epoch = 0
        self.iteration = 0
        self.epochs = epochs
        self.best_mean_iu = 0

    def validate(self):
        self.model.eval()
        n_class = len(self.val_loader.dataset.class_names)
        val_loss = 0
        label_trues, label_preds = [], []
        for batch_idx, (data, target) in tqdm.tqdm(enumerate(self.val_loader), total=len(self.val_loader), desc='Valid iteration=%d' % self.iteration, ncols=80, leave=False):
            if self.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            with torch.no_grad():
                score = self.model(data)
            loss = self.loss_fn(score, target, size_average=self.size_average)
            loss_data = loss.data.item()
            if np.isnan(loss_data):
                raise ValueError('loss is nan while validating')
            val_loss += loss_data / len(data)
            imgs = data.data.cpu()
            lbl_pred = score.data.max(1)[1].cpu().numpy()[:, :, :]
            lbl_true = target.data.cpu()
            for img, lt, lp in zip(imgs, lbl_true, lbl_pred):
                img, lt = self.val_loader.dataset.untransforms(img, lt)
                label_trues.append(lt)
                label_preds.append(lp)
        label_trues = np.array(label_trues)
        label_preds = np.array(label_preds)
        acc, mean_precision, mean_recall, mean_iou, mean_f1 = label_accuracy_score(label_trues, label_preds, n_class)
        metrics = np.array([acc, mean_precision, mean_recall, mean_iou, mean_f1])
        val_loss /= len(self.val_loader)
        with open(osp.join(self.out, 'valid_log.csv'), 'a') as f:
            elapsed_time = (
                datetime.datetime.now(pytz.timezone('Asia/Tokyo')) -
                self.timestamp_start).total_seconds()
            log = [self.epoch, val_loss] + list(metrics) + [elapsed_time]
            log = map(str, log)
            f.write(','.join(log) + '\n')

        mean_iu = metrics[3]
        is_best = mean_iu > self.best_mean_iu
        if is_best:
            self.best_mean_iu = mean_iu
        torch.save({
            'epoch': self.epoch,
            'iteration': self.iteration,
            'arch': self.model.__class__.__name__,
            'optim_state_dict': self.optim.state_dict(),
            'model_state_dict': self.model.state_dict(),
            'best_mean_iu': self.best_mean_iu,
        }, osp.join(self.out, 'checkpoint.pth.tar'))
        if is_best:
            shutil.copy(osp.join(self.out, 'checkpoint.pth.tar'),
                        osp.join(self.out, 'model_best.pth.tar'))

    def train_epoch(self):
        self.model.train()
        n_class = len(self.train_loader.dataset.class_names)
        for batch_idx, (data, target) in tqdm.tqdm(
                enumerate(self.train_loader), total=len(self.train_loader),
                desc='Train epoch=%d' % self.epoch, ncols=80, leave=False):
            iteration = batch_idx + self.epoch * len(self.train_loader)
            if self.iteration != 0 and (iteration - 1) != self.iteration:
                continue  # for resuming
            self.iteration = iteration
            assert self.model.training
            if self.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            self.optim.zero_grad()
            score = self.model(data)
            loss = cross_entropy2d(score, target, size_average=self.size_average)
            loss /= len(data)
            loss_data = loss.data.item()
            if np.isnan(loss_data):
                raise ValueError('loss is nan while training')
            loss.backward()
            self.optim.step()
            metrics = []
            lbl_pred = score.data.max(1)[1].cpu().numpy()[:, :, :]
            lbl_true = target.data.cpu().numpy()
            acc, mean_precision, mean_recall, mean_iou, mean_f1 = label_accuracy_score(lbl_true, lbl_pred, n_class=n_class)
            metrics.append((acc, mean_precision, mean_recall, mean_iou, mean_f1))
            metrics = np.mean(metrics, axis=0)
            with open(osp.join(self.out, 'train_log.csv'), 'a') as f:
                elapsed_time = (
                    datetime.datetime.now(pytz.timezone('Asia/Tokyo')) -
                    self.timestamp_start).total_seconds()
                log = [self.epoch, self.iteration] + [loss_data] + metrics.tolist() + [elapsed_time]
                log = map(str, log)
                f.write(','.join(log) + '\n')

    def train(self):
        for epoch in tqdm.trange(self.epochs, desc='Train', ncols=80):
            self.epoch = epoch
            self.validate()
            self.train_epoch()