from ast import Module
import warnings

import numpy as np
import os
import pandas as pd
import torch
from torch.nn import MSELoss, CrossEntropyLoss
from torch import optim
from torch.utils.tensorboard import SummaryWriter

from sklearn.metrics import accuracy_score

from tqdm import tqdm
from queue import Queue

from .batch_generator import Dataset
from .nn_logging import Logger
from .sgdr_restarts_warmup import CosineAnnealingWarmupRestarts
from .LRscheduler_constant import ConstantLR
from .batch_factory import BatchFactory
from .gpu_augmenter import Augmenter
from .weighted_mse import WeightedMse, get_weights_and_bounds, BNILoss, BMCLoss, WeightedMSESavedWeights
from .resnet_regressor import ResnetRegressor


def train_single_epoch(model: torch.nn.Module,
                       optimizer: torch.optim.Optimizer,
                       loss_function,
                       cuda_batches_queue: Queue,
                       per_step_epoch: int,
                       current_epoch: int,
                       tb_writer: SummaryWriter,
                       lr_scheduler):
    model.train()
    loss_values = []
    loss_tb = []
    pbar = tqdm(total=per_step_epoch, dynamic_ncols=True)
    pbar.set_description(desc='train')

    warning_elapsed = False

    true_labels = []
    model_labels = []

    for batch_idx in range(per_step_epoch):
        batch = cuda_batches_queue.get(block=True)

        optimizer.zero_grad()
        data_out = model(batch.images)

        loss = loss_function(data_out, batch.targets)
        loss_values.append(loss.item())

        curr_preds = np.squeeze(data_out.detach().cpu().numpy())
        curr_preds = curr_preds>=0.5
        model_labels.append(curr_preds)
        curr_targets = np.squeeze(batch.targets.detach().cpu().numpy())
        true_labels.append(curr_targets)
        curr_batch_accuracy = accuracy_score(curr_targets, curr_preds)

        loss_tb.append(loss.item())
        tb_writer.add_scalar('train_loss_per_step', np.mean(loss_tb), current_epoch * per_step_epoch + batch_idx)
        loss_tb = []

        loss.backward()
        optimizer.step()
        pbar.update()
        pbar.set_postfix({'loss': loss.item(), 'batch_accuracy': curr_batch_accuracy, 'cuda_queue_len': cuda_batches_queue.qsize()})

        # if lr_scheduler is not None:
        #     lr_scheduler.step()
        # else:
        #     if not warning_elapsed:
        #         warnings.warn('lr_scheduler is None')
        #         warning_elapsed = True

    pbar.close()

    model_labels = np.concatenate(model_labels)
    true_labels = np.concatenate(true_labels)
    epoch_accuracy = accuracy_score(true_labels, model_labels)
    print('epoch train loss: %f' % np.mean(loss_values))
    print('epoch train accuracy: %f' % epoch_accuracy)


    return np.mean(loss_values), epoch_accuracy


def validate_single_epoch(model: torch.nn.Module,
                          loss_function,
                          cuda_batches_queue: Queue,
                          per_step_epoch: int,
                          current_epoch: int,
                          tb_writer: SummaryWriter):
    model.eval()
    loss_values = []

    true_labels = []
    model_labels = []

    pbar = tqdm(total=per_step_epoch, dynamic_ncols=True)
    pbar.set_description(desc='validation')
    for batch_idx in range(per_step_epoch):
        batch = cuda_batches_queue.get(block=True)

        with torch.no_grad():
            data_out = model(batch.images)

        loss = loss_function(data_out, batch.targets)

        curr_preds = np.squeeze(data_out.detach().cpu().numpy())
        curr_preds = curr_preds>=0.5
        model_labels.append(curr_preds)
        curr_targets = np.squeeze(batch.targets.detach().cpu().numpy())
        true_labels.append(curr_targets)
        curr_batch_accuracy = accuracy_score(curr_targets, curr_preds)

        loss_values.append(loss.item())

        pbar.update()
        pbar.set_postfix({'loss': loss.item(),
                          'batch_accuracy': curr_batch_accuracy,
                          'cuda_queue_len': cuda_batches_queue.qsize()})
    pbar.close()

    model_labels = np.concatenate(model_labels)
    true_labels = np.concatenate(true_labels)
    epoch_accuracy = accuracy_score(true_labels, model_labels)
    print('epoch test loss: %f' % np.mean(loss_values))
    print('epoch test accuracy: %f' % epoch_accuracy)


    return np.mean(loss_values), epoch_accuracy
