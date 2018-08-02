import os
import argparse

from workers.svm_surrogate import SVMSurrogateWorker as Worker
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
parser.add_argument('--surrogate_path', type=str, help='Path to the pickled surrogate models. If None, HPOlib2 will automatically download the surrogates to the .hpolib directory in your home directory.', default=None)
parser.add_argument('--min_budget', type=float, help='Smallest fraction of the full dataset that is used.', default=1/512)
parser.add_argument('--max_budget', type=float, help='Largest fraction of the full dataset that is used.', default=1)
args = parser.parse_args()

#args = parser.parse_args('--method bohb --num_iterations 32 --run_id 2'.split())

# this is a synthetic benchmark, so we will use the run_id to separate the independent runs
worker = Worker(surrogate_path=args.surrogate_path, measure_test_loss=True, run_id=args.run_id)

# directory where the results are stored
dest_dir = os.path.join(args.dest_dir, "SVMSurrogate")

# SMAC can be informed whether the objective is deterministic or not
smac_deterministic = True

# run experiment
result = util.run_experiment(args, worker, dest_dir, smac_deterministic)

print(result.get_incumbent_trajectory())
