import numpy as np
import unittest

from hpbandster.learning_curve_models.lcnet import LCNetWrapper


class TestLCNetWrapper(unittest.TestCase):

    def test_fit_and_predict(self):
        def toy_example(t, a, b):
            return (10 + a * np.log(b * t)) / 10. + 10e-3 * np.random.rand()

        observed = 20
        N = 200
        n_epochs = 100
        observed_t = int(n_epochs * (observed / 100.))

        t_idx = np.arange(1, observed_t + 1) / n_epochs

        configs = np.random.rand(N, 2)
        learning_curves = [1 - toy_example(t_idx, configs[i, 0], configs[i, 1]) for i in range(N)]

        times = np.repeat(t_idx[None, :] * n_epochs, N, axis=0)

        lcnet = LCNetWrapper(max_num_epochs=n_epochs)
        lcnet.fit(times=times, losses=learning_curves, configs=configs)

        config = np.random.rand(2)
        t = np.arange(1, n_epochs)
        m, v = lcnet.predict_unseen(t, config)

if __name__ == "__main__":
    unittest.main()
