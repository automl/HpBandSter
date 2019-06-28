"""
Example 9 - Live monitoring with Neptune callback
================================

"""
import logging

logging.basicConfig(level=logging.WARNING)

import argparse

import hpbandster.core.nameserver as hpns
import hpbandster.core.result as hpres

from hpbandster.optimizers import BOHB as BOHB
from hpbandster.examples.commons import MyWorker

import neptune


parser = argparse.ArgumentParser(description='Example 1 - sequential and local execution.')
parser.add_argument('--min_budget', type=float, help='Minimum budget used during the optimization.', default=9)
parser.add_argument('--max_budget', type=float, help='Maximum budget used during the optimization.', default=243)
parser.add_argument('--n_iterations', type=int, help='Number of iterations performed by the optimizer', default=4)
args = parser.parse_args()

NS = hpns.NameServer(run_id='example1', host='127.0.0.1', port=None)
NS.start()

w = MyWorker(sleep_interval=0, nameserver='127.0.0.1', run_id='example1')
w.run(background=True)


# Step 1: Create a callback
# You need define a class with __call__ method which is your de-facto callback and a dummy new_config method.
# In the __call__ method you do whatever you want after every iteration with the job object.
# In this example we extract the loss and hyperparameters for every run and log them to an external server but
# you can do whatever you want to happen on iteration end.
class NeptuneCallback:
    def new_config(self, *args, **kwargs):
        pass

    def __call__(self, job):
        neptune.send_metric('run_score', job.result['loss'])
        neptune.send_text('run_parameters', str(job.kwargs['config']))

# Step 2: Pass callback to optimizer
# Every optimizer has results_logger argument. Just pass your callback instance there.
bohb = BOHB(configspace=w.get_configspace(),
            run_id='example1', nameserver='127.0.0.1',
            min_budget=args.min_budget, max_budget=args.max_budget,
            result_logger=NeptuneCallback(),
            )

# Step 3: Setup live logging
# You need specify your user token and project to log.
# By default you can use this open api token and an open project.
# Go to https://ui.neptune.ml/o/shared/org/HpBandSter-Callback/experiments to see your experiment.
# You can see an example run here https://ui.neptune.ml/o/shared/org/HpBandSter-Callback/e/HPBAN-2/charts
neptune.init(api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5tbCIsImFwaV9rZXkiOiJiNzA2YmM4Zi03NmY5LTRjMmUtOTM5ZC00YmEwMzZmOTMyZTQifQ==',
             project_qualified_name='shared/HpBandSter-Callback')
with neptune.create_experiment():
    res = bohb.run(n_iterations=args.n_iterations)

bohb.shutdown(shutdown_workers=True)
NS.shutdown()
