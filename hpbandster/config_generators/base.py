import logging
from hpbandster.utils import  json_result_logger


class base_config_generator(object):
    def __init__(self, directory=None, result_logger=json_result_logger, overwrite=False, logger=None):
        """
        Parameters:
        -----------

        directory: string
            where the results are logged
        logger: hpbandster.utils.result_logger_v??
            the logger to store the data, defaults to v1
        overwrite: bool
            whether or not existing data will be overwritten
        logger: logging.logger
            for some debug output

        """
        if not directory is None:
            self.result_logger  = result_logger(directory, overwrite=overwrite)
        else:
            self.result_logger = None
            
        if logger is None:
            self.logger=logging.getLogger('hpbandster')
        else:
            self.logger=logger

    def get_config(self, budget):
        """
            function to sample a new configuration

            This function is called inside Hyperband to query a new configuration


            Parameters:
            -----------
            budget: float
                the budget for which this configuration is scheduled

            returns: (config, info_dict)
                must return a valid configuration and a (possibly empty) info dict
                
            
        """
        raise NotImplementedError('This function needs to be overwritten in %s.'%(self.__class__.__name__))

    def new_result(self, job):
        """
            function to register finished runs

            Every time a run has finished, this function should be called
            to register it with the result logger. If overwritten, make
            sure to call this method from the base class to ensure proper
            logging.


            Parameters:
            -----------
            job: instance of hpbandster.distributed.dispatcher.Job
                contains all necessary information about the job
        """
        if not self.result_logger is None:
            self.result_logger(job)
