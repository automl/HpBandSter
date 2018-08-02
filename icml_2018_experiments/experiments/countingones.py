import os
import argparse

from workers.countingones import CountingOnesWorker as Worker
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
parser.add_argument('--min_budget', type=int, help='Minimum number of draws from each Bernoulli distribution.', default=9)
parser.add_argument('--max_budget', type=int, help='Minimum number of draws from each Bernoulli distribution.', default=729)
parser.add_argument('--num_categoricals', type=int, help='Number of categorical parameters in the search space.', default=4)
parser.add_argument('--num_continuous', type=int, help='Number of continuous parameters in the search space.', default=4)

args = parser.parse_args()


# this is a synthetic benchmark, so we will use the run_id to separate the independent runs
worker = Worker(num_continuous=args.num_continuous, num_categorical=args.num_categoricals, max_budget=args.max_budget, measure_test_loss=True, run_id=args.run_id)

# directory where the results are stored
dest_dir = os.path.join(args.dest_dir, "CountingOnes", "%i_%i"%(args.num_continuous, args.num_categoricals))

# SMAC can be informed whether the objective is deterministic or not
smac_deterministic = True

# run experiment
util.run_experiment(args, worker, dest_dir, smac_deterministic)
