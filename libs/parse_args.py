import argparse, warnings, datetime, os
import numpy as np


def parse_args(args):
    parser = argparse.ArgumentParser(prog='CNN training script for reactability classification',
                                     description='Simple training script for training a CNN network.')
    

    parser.add_argument('-d', '--debug', dest='debug', help='Debug flag', action="store_true")
    parser.add_argument('--run-prefix', dest='run_prefix', default='devel')
    parser.add_argument('--csv-train', dest='csv_train', help='Path to file containing training annotations', required=True)
    parser.add_argument('--csv-test', dest='csv_test', help='Path to file containing testing annotations (optional)')

    # parser.add_argument('-e', '--epochs', help='Number of epochs', type=int, default=25)
    parser.add_argument('-s', '--scheduler', help='type of LR scheduler for ADAM optimization algorithm',
                        type=str,
                        choices=['constant', 'sgdr'])
    lr_parameters_group = parser.add_mutually_exclusive_group()
    lr_parameters_group.add_argument('--epochs', help='Number of epochs', type=int)
    lr_parameters_group.add_argument('--cycles', help='number of cycles in SGDR LR schedule', type=int)

    sgdr_parameters_group = parser.add_argument_group('SGDR scheduler parameters')
    sgdr_parameters_group.add_argument('--first-cycle-steps', dest='first_cycle_steps', type=int, default=16)
    sgdr_parameters_group.add_argument('--cycle-mult', dest='cycle_mult', type=float, default=2.0)
    sgdr_parameters_group.add_argument('--max-lr', dest='max_lr', type=float, default=1e-4)
    sgdr_parameters_group.add_argument('--min-lr', dest='min_lr', type=float, default=1e-8)
    sgdr_parameters_group.add_argument('--warmup-steps', dest='warmup_steps', type=float, default=8)
    sgdr_parameters_group.add_argument('--gamma', dest='gamma', type=float, default=0.7)
    sgdr_parameters_group.add_argument('--last-epoch', dest='last_epoch', type=int, default=-1)


    parser.add_argument('-b', '--batch-size', dest='batch_size', help='batch size', type=int, default=32)
    parser.add_argument('-c', '--caching', dest='caching', help='whether use caching', action="store_true")

    parser.add_argument('--no-training', dest='no_training', help='Disables training (for debug purpose only)', action='store_true')
    parser.add_argument('--no-evaluation', dest='no_evaluation', help='Disables evaluation (for debug purpose only)', action='store_true')

    parser.add_argument('--evaluate-every', dest='evaluate_every', help='period for evaluation (in epochs)', type=int, default=1)
    parser.add_argument('--tblog-batch-loss-every', dest='tblog_batch_loss_every', help='period for batch loss log to Tensorboard (in batches)', type=int, default=16)

    return preprocess_args(parser.parse_args(args))


def preprocess_args(parsed_args):
    if parsed_args.scheduler == 'sgdr':
        assert parsed_args.first_cycle_steps
        assert parsed_args.cycle_mult
        assert parsed_args.max_lr
        assert parsed_args.min_lr
        assert parsed_args.warmup_steps
        assert parsed_args.gamma
        assert parsed_args.last_epoch
        assert parsed_args.cycles
    return parsed_args