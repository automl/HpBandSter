import os
import argparse

import numpy as np

from workers.bnn import BNNWorker as Worker
import util

# deactivate debug output for faster experiments
import logging
logging.basicConfig(level=logging.DEBUG)


################################################################################
#                    Benchmark specific stuff
################################################################################
parser = argparse.ArgumentParser(description='Run different optimizers to optimize BNNs on different datasets.', conflict_handler='resolve')
parser = util.standard_parser_args(parser)


# add benchmark specific arguments
parser.add_argument('--dataset', choices=['toyfunction', 'bostonhousing', 'proteinstructure', 'yearprediction'], help="name of the dataset used", default='bostonhousing')
parser.add_argument('--min_budget', type=int, help='Minimum number of MCMC steps used to draw samples for the BNN.', default=300)
parser.add_argument('--max_budget', type=int, help='Maximum number of MCMC steps used to draw samples for the BNN.', default=10000)

args = parser.parse_args()

args = parser.parse_args('--method bohb'.split())

# this is a synthetic benchmark, so we will use the run_id to separate the independent runs
worker = Worker(dataset=args.dataset, measure_test_loss=False, run_id=args.run_id, max_budget=args.max_budget)

# directory where the results are stored
dest_dir = os.path.join(args.dest_dir, "BNNs", "%s"%(args.dataset))

# SMAC can be informed whether the objective is deterministic or not
smac_deterministic = True

# run experiment
result = util.run_experiment(args, worker, dest_dir, smac_deterministic, store_all_runs=True)
print(result.get_all_runs())
