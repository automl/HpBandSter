# ICML 2018 Experiments

This branch contains source code to reproduce most of the experiments from the paper

    Falkner, Stefan and Klein, Aaron and Hutter, Frank
    BOHB: Robust and Efficient Hyperparameter Optimization at Scale
    In: Proceedings of the 35th International Conference on Machine Learning

Everything you need to run is in the `icml_2018` folder, in particular:
  - a `requirements.txt` to install all necessary python dependencies.
  - python scripts to actually run the experiments under `experiments`.
  - scripts to go through the results and produce plots similar to the ones in the paper in `analysis`
  - some precomputed results of the more expensive methods (mostly based on Gaussian Processes) that are quite expensive to rerun in `data`

If you have any problems running the code or questions about it, please don't hesitate to contact us.

*NOTE:* The folder `icml_2018/experiments/workers/lib/cifar10_cutout_validation` contains modified code based on the work by X. Gastaldi see.
Please refer to https://github.com/xgastaldi/shake-shake for the requirements needed to run it.
