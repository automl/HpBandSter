import logging

import ConfigSpace
import numpy as np
import scipy.stats as sps
import scipy.optimize as spo
import statsmodels.api as sm

from hpbandster.config_generators.base import base_config_generator



class KDEEI(base_config_generator):
    
    def __init__(self, configspace, top_n_percent=10, update_after_n_points=1,
                 min_points_in_model = None, mode='DE',
                 num_samples = 1024, random_fraction=0.0,
                **kwargs):
        """
            Fits for each given budget a kernel density estimator on the best N percent of the
            evaluated configurations on this budget.


            Parameters:
            -----------
            configspace: ConfigSpace
                Configuration space object
            top_n_percent: int
                Determines the percentile of configurations that will be used as training data
                for the kernel density estimator, e.g if set to 10 the 10% best configurations will be considered
                for training.
            update_after_n_points: int
                Specifies after how many new observed points the kernel density will be retrained.
            min_points_in_model: int
                minimum number of datapoints needed to fit a model
            mode: str
                how EI is optimized:
                    -'sampling' corresponds to sampling from the KDE and evaluating EI
                    -'DE' corresponds to differential evolution (from scipy.optimize)
            num_samples: int
                number of samples drawn to optimize EI via sampling or number of function
                when using DE
            random_fraction: float
                fraction of random configurations returned

        """
        super(KDEEI, self).__init__(**kwargs)

        self.top_n_percent = top_n_percent
        self.update_after_n_points = update_after_n_points
        self.configspace = configspace
        
        self.min_points_in_model = min_points_in_model
        if min_points_in_model is None:
            self.min_points_in_model = len(self.configspace.get_hyperparameters())+1

        self.mode=mode
        self.num_samples = num_samples
        self.random_fraction = random_fraction


        # TODO: so far we only consider continuous configuration spaces
        self.var_type = "c" * len(self.configspace.get_hyperparameters())
        self.configs = dict()
        self.losses = dict()
        self.kde_models = dict()
        
    def get_config(self, budget):
        """
            Function to sample a new configuration

            This function is called inside Hyperband to query a new configuration


            Parameters:
            -----------
            budget: float
                the budget for which this configuration is scheduled

            returns: config
                should return a valid configuration

        """

        sample = None
        info_dict = {}
        
        # If no model is available, sample from prior
        if len(self.kde_models.keys()) == 0 or np.random.rand() < self.random_fraction:
            sample =  self.configspace.sample_configuration().get_dictionary()
            info_dict['model_based_pick'] = False



        if sample is None:
            # If we haven't seen anything with this budget, we sample from the kde trained on the highest budget
            #if budget not in self.kde_models.keys():
            #    budget = max(self.kde_models.keys())

            #sample from largest budget
            budget = max(self.kde_models.keys())

            l = self.kde_models[budget]['good'].pdf
            g = self.kde_models[budget]['bad' ].pdf
        
            minimize_me = lambda x: max(1e-8, g(x))/max(l(x), 1e-8)
            
            if self.mode == 'DE':
                
                dim = len(self.configspace._hyperparameters)
                # the 15*dim is the default population size of spo.differential_evolution
                maxiter = self.num_samples//(15*dim) + 1
            
                res = spo.differential_evolution(minimize_me, [(0,1)]*dim, maxiter=maxiter, init='random')
                
                sample = ConfigSpace.Configuration(self.configspace, vector=res.x)
                
                
            if self.mode == 'sampling':
                
                kde_good = self.kde_models[budget]['good']

                best = float('inf')
                best_vector = None


                for i in range(self.num_samples):
                    idx = np.random.randint(0, len(kde_good.data))
                    vector = [sps.truncnorm.rvs(-m/bw,(1-m)/bw, loc=m, scale=bw) for m,bw in zip(kde_good.data[idx], 2*kde_good.bw)]
                    
                    val = minimize_me(vector) 
                    if val < best:
                        best = val
                        best_vector = vector

                if best_vector is None:
                    self.logger.debug("Sampling based optimization with %i samples failed -> using random configuration"%self.num_samples)
                    sample = self.configspace.sample_configuration().get_dictionary()
                    info_dict['model_based_pick']  = False
                else:
                    sample = ConfigSpace.Configuration(self.configspace, vector=best_vector).get_dictionary()
                    info_dict['model_based_pick'] = True

        return sample, info_dict

    def new_result(self, job):
        """
            function to register finished runs

            Every time a run has finished, this function should be called
            to register it with the result logger. If overwritten, make
            sure to call this method from the base class to ensure proper
            logging.


            Parameters:
            -----------
            job: hpbandster.distributed.dispatcher.Job object
                contains all the info about the run
        """
        
        super(KDEEI, self).new_result(job)

        if job.result is None:
            # skip crashed results
            # alternatively, one could also assign a -inf loss and 
            # count them as bad configurations
            return

        budget = job.kwargs["budget"]
        loss = job.result["loss"]

        if budget not in self.configs.keys():
            self.configs[budget] = []
            self.losses[budget] = []

        # We want to get a numerical representation of the configuration in the original space

        conf = ConfigSpace.Configuration(self.configspace, job.kwargs['config'])
        self.configs[budget].append(conf.get_array())
        self.losses[budget].append(loss)

        if len(self.configs[budget]) <= self.min_points_in_model:
            return 

        if len(self.configs[budget]) % self.update_after_n_points == 0:
            train_configs = self.configs[budget]
            train_losses =  self.losses[budget]
        
            n_good = int(max(self.top_n_percent * len(train_configs) / 100., self.min_points_in_model))
            n_bad = int(max((100-self.top_n_percent) * len(train_configs) / 100., self.min_points_in_model))

            # Refit KDE for the current budget
            idx = np.argsort(train_losses)

            train_data_good = (np.array(train_configs)[idx])[:n_good]
            train_data_bad  = (np.array(train_configs)[idx])[-n_bad:]
            
            # quick rule of thumb
            bw_estimation = 'normal_reference'


            if train_data_good.shape[0] < train_data_good.shape[1]:
                return
            if train_data_bad.shape[0] < train_data_bad.shape[1]:
                return          
            
            #more expensive crossvalidation method
            #bw_estimation = 'cv_ls'

            self.kde_models[budget] = {
                    'good': sm.nonparametric.KDEMultivariate(data=train_data_good, var_type=self.var_type, bw=bw_estimation),
                    'bad' : sm.nonparametric.KDEMultivariate(data=train_data_bad,  var_type=self.var_type, bw=bw_estimation)
            }
            self.logger.debug('done building a new model for budget %f based on %i/%i split'%(budget, n_good, n_bad))
