import argparse, warnings, datetime, os
import numpy as np


def parse_args(args):
    parser = argparse.ArgumentParser(prog='CNN training script for reactability classification',
                                     description='Simple training script for training a CNN network.')
    

    parser.add_argument('-d', '--debug', dest='debug', help='Debug flag', action="store_true")
    parser.add_argument('--run-prefix', dest='run_prefix', default='devel')
    parser.add_argument('--csv-train', dest='csv_train', help='Path to file containing training annotations', required=True)
    parser.add_argument('--csv-test', dest='csv_test', help='Path to file containing testing annotations (optional)')
    parser.add_argument('-e', '--epochs', help='Number of epochs', type=int, default=25)
    parser.add_argument('-b', '--batch-size', dest='batch_size', help='batch size', type=int, default=32)
    parser.add_argument('-c', '--caching', dest='caching', help='whether use caching', action="store_true")

    parser.add_argument('--no-training', dest='no_training', help='Disables training (for debug purpose only)', action='store_true')
    parser.add_argument('--no-evaluation', dest='no_evaluation', help='Disables evaluation (for debug purpose only)', action='store_true')

    parser.add_argument('--evaluate-every', dest='evaluate_every', help='period for evaluation (in epochs)', type=int, default=1)
    parser.add_argument('--tblog-batch-loss-every', dest='tblog_batch_loss_every', help='period for batch loss log to Tensorboard (in batches)', type=int, default=16)

    return preprocess_args(parser.parse_args(args))


def preprocess_args(parsed_args):
    return parsed_args