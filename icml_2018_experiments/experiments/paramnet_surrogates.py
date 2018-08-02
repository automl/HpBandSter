import os
import argparse

import numpy as np

from workers.paramnet_surrogates import ParamNetSurrogateWorker as Worker
import util

# deactivate debug output for faster experiments
import logging
logging.basicConfig(level=logging.ERROR)


################################################################################
#                    Benchmark specific stuff
################################################################################
parser = argparse.ArgumentParser(description='Run different optimizers on the CountingOnes problem.', conflict_handler='resolve')
parser = util.standard_parser_args(parser)


# add benchmark specific arguments
parser.add_argument('--dataset', choices=['adult', 'higgs', 'letter', 'mnist', 'optdigits', 'poker'], help="name of the dataset used", default='mnist')
parser.add_argument('--surrogate_path', type=str, help='path to the pickled surrogate models', default=None)
parser.add_argument('--min_budget', type=int, help='Not used! Dataset specific value given in the source.', default=0)
parser.add_argument('--max_budget', type=int, help='Not used! Dataset specific value given in the source.', default=0)

args = parser.parse_args()

# this is a synthetic benchmark, so we will use the run_id to separate the independent runs
worker = Worker(dataset=args.dataset, surrogate_path=args.surrogate_path, measure_test_loss=False, run_id=args.run_id)

args.min_budget, args.max_budget = worker.budgets[args.dataset]

# directory where the results are stored
dest_dir = os.path.join(args.dest_dir, "ParamNetSurrogates", "%s"%(args.dataset))

# SMAC can be informed whether the objective is deterministic or not
smac_deterministic = True

# run experiment
result = util.run_experiment(args, worker, dest_dir, smac_deterministic)
print(result.get_all_runs())
