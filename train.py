import os
import sys
from os.path import join, isfile, isdir
from tqdm import tqdm
import numpy as np
import warnings

from libs.sgdr_restarts_warmup import CosineAnnealingWarmupRestarts

sys.path.append(os.path.join(sys.path[0], '..'))

import torch
import torch.optim as optim
import pandas as pd
from queue import Empty, Queue
from threading import Thread
from torch.utils.tensorboard import SummaryWriter
from torch.nn import MSELoss, CrossEntropyLoss, BCELoss

from libs.service_defs import find_files, find_directories, EnsureDirectoryExists, DoesPathExistAndIsFile
from libs.copytree import copytree_multi, copy2
from libs.batch_generator import Dataset
from libs.train_common import train_single_epoch, validate_single_epoch
from libs.cnn_model import CNNmodel
from libs.parse_args import parse_args
from libs.threaded_preprocessing import thread_killer, threaded_batches_feeder, threaded_cuda_batches
from libs.LRscheduler_constant import ConstantLR


def main(args=None):
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)
    
    DEVICE = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')

    # region logs_basepath
    existing_logs_directories = find_directories('./logs', '%s_run*' % args.run_prefix, maxdepth=2)
    prev_runs = [os.path.basename(os.path.split(d)[0]) for d in existing_logs_directories]
    prev_runs = [int(s.replace('%s_run' % args.run_prefix, '')) for s in prev_runs]
    if len(prev_runs) > 0:
        curr_run = np.max(prev_runs) + 1
    else:
        curr_run = 1
    curr_run = '%s_run%04d' % (args.run_prefix, curr_run)
    logs_basepath = os.path.join('./logs', curr_run)
    EnsureDirectoryExists(logs_basepath)

    tb_basepath = os.path.join('./TBoard', curr_run)
    checkpoints_basepath = os.path.join('./checkpoints', curr_run)
    EnsureDirectoryExists(checkpoints_basepath)

    vis_path = os.path.join('./logs', curr_run, 'plots')
    EnsureDirectoryExists(vis_path)
    # endregion

    # region backing up the scripts configuration
    EnsureDirectoryExists('./scripts_backup')
    print('backing up the scripts')
    ignore_func = lambda dir, files: [f for f in files if (isfile(join(dir, f)) and f[-3:] != '.py')] + [d for d in files if ((isdir(d)) & (('scripts_backup' in d) |
                                                                                                                                            ('__pycache__' in d)    |
                                                                                                                                            ('.pytest_cache' in d)  |
                                                                                                                                            d.endswith('.ipynb_checkpoints') |
                                                                                                                                            d.endswith('logs.bak')           |
                                                                                                                                            d.endswith('outputs')            |
                                                                                                                                            d.endswith('processed_data')     |
                                                                                                                                            d.endswith('build')              |
                                                                                                                                            d.endswith('logs')               |
                                                                                                                                            d.endswith('TBoard')             |
                                                                                                                                            d.endswith('checkpoints')        |
                                                                                                                                            d.endswith('snapshots')))]
    scripts_backup_dir = os.path.join('./scripts_backup', curr_run)
    copytree_multi('./', scripts_backup_dir, ignore=ignore_func)
    with open(os.path.join(scripts_backup_dir, 'launch_parameters.txt'), 'w+') as f:
        f.writelines([f'{s}\n' for s in sys.argv])
    if DoesPathExistAndIsFile(args.csv_train):
        copy2(args.csv_train, os.path.join(scripts_backup_dir, os.path.basename(args.csv_train)))
    if DoesPathExistAndIsFile(args.csv_test):
        copy2(args.csv_test, os.path.join(scripts_backup_dir, os.path.basename(args.csv_test)))
    # endregion backing up the scripts configuration

    orig_size = (550, 720)
    
    assert DoesPathExistAndIsFile(args.csv_train), 'csv_train file should exist'
    assert DoesPathExistAndIsFile(args.csv_test), 'csv_test file should exist'

    train_dataset = Dataset(index_fname=args.csv_train,
                            batch_size = args.batch_size,
                            do_shuffle=True,
                            debug=args.debug,
                            caching=args.caching)
    
    steps_per_epoch_train = len(train_dataset)//args.batch_size + (0 if len(train_dataset)%args.batch_size == 0 else 1)

    print('train dataset contains %d files; with batch_size=%d it is %d batches per epoch' % (len(train_dataset),
                                                                                              train_dataset.batch_size,
                                                                                              steps_per_epoch_train))
    if args.csv_test is None:
        test_dataset = None
        print('No test annotations provided.')
    else:
        test_dataset = Dataset(index_fname=args.csv_test,
                               batch_size = args.batch_size,
                               do_shuffle=False,
                               debug=args.debug,
                               caching=args.caching)
        steps_per_epoch_test = len(test_dataset)//args.batch_size + (0 if len(test_dataset)%args.batch_size == 0 else 1)
    
    
    batches_queue_length = 4
    train_preprocess_workers = 1
    test_preprocess_workers = 1

    train_batches_queue = Queue(maxsize=batches_queue_length)
    train_cuda_batches_queue = Queue(maxsize=8)
    train_thread_killer = thread_killer()
    train_thread_killer.set_tokill(False)

    for _ in range(train_preprocess_workers):
        thr = Thread(target=threaded_batches_feeder,
                     args=(train_thread_killer,
                           train_batches_queue,
                           train_dataset))
        thr.start()

    train_cuda_transfers_thread_killer = thread_killer()
    train_cuda_transfers_thread_killer.set_tokill(False)
    train_cudathread = Thread(target=threaded_cuda_batches,
                              args=(train_cuda_transfers_thread_killer,
                                    train_cuda_batches_queue,
                                    train_batches_queue,
                                    DEVICE))
    train_cudathread.start()

    if test_dataset is not None:
        test_batches_queue = Queue(maxsize=batches_queue_length)
        test_cuda_batches_queue = Queue(maxsize=4)
        test_thread_killer = thread_killer()
        test_thread_killer.set_tokill(False)
        
        for _ in range(test_preprocess_workers):
            thr = Thread(target=threaded_batches_feeder,
                        args=(test_thread_killer,
                              test_batches_queue,
                              test_dataset))
            thr.start()
        
        test_cuda_transfers_thread_killer = thread_killer()
        test_cuda_transfers_thread_killer.set_tokill(False)
        test_cudathread = Thread(target=threaded_cuda_batches,
                                 args=(test_cuda_transfers_thread_killer,
                                       test_cuda_batches_queue,
                                       test_batches_queue,
                                       DEVICE))
        test_cudathread.start()

    model = CNNmodel()
    model = model.to(DEVICE)
    # model = torch.nn.DataParallel(model, DEVICE)
    model.training = True

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    # scheduler = ConstantLR(optimizer=optimizer, lr=1e-4)
    scheduler = CosineAnnealingWarmupRestarts(optimizer=optimizer,
                                              first_cycle_steps = 16,
                                              cycle_mult = 1.5,
                                              max_lr = 1e-4,
                                              min_lr = 1e-8,
                                              warmup_steps = 8,
                                              gamma = 0.7,
                                              last_epoch = -1)
    model.train()

    tb_writer = SummaryWriter(log_dir=tb_basepath)
    loss_fn = BCELoss()
    
    
    best_test_accuracy = 0.0
    best_test_epoch = -1
    best_test_accuracies = []
    best_test_epoches = []

    try:
        for epoch in range(args.epochs):
            print(f'Epoch {epoch} / {args.epochs}')

            train_loss, train_accuracy = train_single_epoch(model,
                                                            optimizer,
                                                            loss_fn,
                                                            train_cuda_batches_queue,
                                                            steps_per_epoch_train,
                                                            current_epoch=epoch,
                                                            tb_writer=tb_writer,
                                                            lr_scheduler=scheduler)
            

            
            tb_writer.add_scalar('train_loss_per_epoch', train_loss, epoch)
            tb_writer.add_scalar('train_accuracy_per_epoch', train_accuracy, epoch)

            if scheduler is not None:
                scheduler.step()
            else:
                if not warning_elapsed:
                    warnings.warn('learning rate scheduler is None')
                    warning_elapsed = True

            with open(os.path.join(logs_basepath, 'train_measures_evolution.txt'), 'a') as f:
                f.write(str(epoch) + ';' + str(train_loss) + ';' + str(train_accuracy) + '\n')

            test_loss, test_accuracy = validate_single_epoch(model,
                                                             loss_fn,
                                                             test_cuda_batches_queue,
                                                             steps_per_epoch_test,
                                                             current_epoch=epoch,
                                                             tb_writer=tb_writer)
            tb_writer.add_scalar('val_loss_per_epoch', test_loss, epoch)
            tb_writer.add_scalar('val_accuracy_per_epoch', test_accuracy, epoch)

            with open(os.path.join(logs_basepath, 'test_measures_evolution.txt'), 'a') as f:
                f.write(str(epoch) + ';' + str(test_loss) + ';' + str(test_accuracy) + '\n')
            print(f'Test loss: {test_loss}')

            if best_test_accuracy > test_accuracy:
                # save new model
                torch.save(model, os.path.join(checkpoints_basepath, f'model_ep{epoch}.pt'))
                # delete old model
                old_model_path = os.path.join(checkpoints_basepath, f'model_ep{best_test_epoch}.pt')
                if os.path.exists(old_model_path):
                    os.remove(old_model_path)
                best_test_accuracy = test_accuracy
                best_test_epoch = epoch

            if epoch == args.epochs-1:
                torch.save(model, os.path.join(checkpoints_basepath, f'model_ep{epoch}.pt'))


    except KeyboardInterrupt:
        pass

    finally:
        train_thread_killer.set_tokill(True)
        train_cuda_transfers_thread_killer.set_tokill(True)
        for _ in range(train_preprocess_workers):
            try:
                # Enforcing thread shutdown
                train_batches_queue.get(block=True, timeout=1)
                train_cuda_batches_queue.get(block=True, timeout=1)
            except Empty:
                pass
        if test_dataset is not None:
            test_thread_killer.set_tokill(True)
            test_cuda_transfers_thread_killer.set_tokill(True)
            for _ in range(test_preprocess_workers):
                try:
                    # Enforcing thread shutdown
                    test_batches_queue.get(block=True, timeout=1)
                    test_cuda_batches_queue.get(block=True, timeout=1)
                except Empty:
                    pass

    print('done training model')
    return best_test_accuracy
    
    





if __name__ == '__main__':
    main()
