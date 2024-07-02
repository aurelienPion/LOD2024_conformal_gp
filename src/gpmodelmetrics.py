import gpmp as gp
import gpmp.num as gnp
import numpy as np

from src.data import Data
from src.metrics import iae_alpha, rmse
from src.utils import matern_p, constant_mean
from src.j_plus_gp import j_plus_gp


class GPExperiment:
    """Class to conduct experiments with Gaussian Process (GP) models
    and compute related metrics (IAE, REML).
    
    Attributes:
        - d (int): dimension of the design
        - p (int): regularity of the GP model
        - x_min (gnp.array): lower bound of the design
        - x_max (gnp.array): upper bound of the design
        - f (function): test function
        - n_train (int): number of points in the training set
        - n_test (int): number of points in the test set

    Methods:
        - j_plus_gp_point: compute (IAE, REML) when the prediction interval is built with J+GP
        - compute_metrics_set: compute a set of metrics (IAE, RMSE) for varying GP model parameters

    """

    def __init__(self, d, p, x_min, x_max, f, n_train=50, n_test=1500):
        self.d = d
        self.p = p
        self.x_min = x_min
        self.x_max = x_max
        self.n_train = n_train
        self.n_test = n_test

        # GP mean and covariance function
        self.mean = constant_mean
        self.kernel = matern_p
        self.meanparam_sp = None

        # setting the value of f generates the DoE
        self.f = f

    @property
    def f(self):
        return self._f

    @f.setter
    def f(self, f):
        """Set the value of f and build the Design of Experiment"""
        self._f = f

        # Generate data
        x_test = gnp.asarray(
            gp.misc.designs.randunif(self.d, self.n_test, [self.x_min, self.x_max])
        )
        x_train = gnp.asarray(
            gp.misc.designs.randunif(self.d, self.n_train, [self.x_min, self.x_max])
        )
        z = f(gnp.concatenate((x_train, x_test)))
        z_train = gnp.asarray(z[: self.n_train].flatten())
        z_test = gnp.asarray(z[self.n_train :].flatten())

        self.data = Data(x_train=x_train, z_train=z_train, x_test=x_test, z_test=z_test)
        self.model = gp.core.Model(self.mean, self.kernel(self.p))

        self.reml_model()

    def reml_model(self):
        """
        - Select the parameters of the GP model using REML.
        - Compute predictions on data.x_test.
        - Compute predictions by LOO on data.x_train.
        - Compute the IAE and RMSE metrics.
        """
        self.model, info = gp.kernel.select_parameters_with_reml(
            self.model, self.data.x_train, self.data.z_train, info=True
        )
        gp.misc.modeldiagnosis.diag(
            self.model, info, self.data.x_train, self.data.z_train
        )

        self.covparam_reml = np.copy(self.model.covparam)

        # Predictions on the test set
        self.zpm, self.zpv = self.model.predict(self.data.x_train, self.data.z_train, self.data.x_test, convert_out=False)

        # Predictions on the train set using LOO
        self.zpmloo, self.zpvloo, _ = self.model.loo(self.data.x_train, self.data.z_train, convert_out=False)

        # Compute metrics
        self.rmse_reml = rmse(self.zpm, self.data.z_test)
        self.iae_alpha_reml = iae_alpha(self.data.z_test, zpm=self.zpm, zpv=self.zpv)
        self.rmse_remlloo = rmse(self.zpmloo, self.data.z_train)
        self.iae_alpha_remlloo = iae_alpha(
            self.data.z_train, zpm=self.zpmloo, zpv=self.zpvloo
        )

    def j_plus_gp_point(self, covparam=None):
        """Compute IAE of prediction by J+GP"""
        if covparam is not None:
            self.model.covparam = covparam 
        quantiles_res_plus, quantiles_res_minus = j_plus_gp(self.model, self.data)
        self.iae_j_plus_gp = iae_alpha(
            self.data.z_test,
            quantiles_minus=quantiles_res_minus,
            quantiles_plus=quantiles_res_plus,
        )
        self.model.covparam = np.copy(self.covparam_reml)

    def evaluate_model_variation(self, lb, ub, set_size=500):
        """Compute a set of metrics (IAE, RMSE) for predictions with
        the GP model when the parameters vary around
        self.covparam_reml.

        Each parameter theta_i varies in [lb_i*u + ub_i], where u~U(0, 1).

        The results are stored in the attributes:
            - On the test set
                - rmse_res
                - iae_alpha_res
            - On the train set
                - rmse_resloo
                - iae_alpha_resloo

        Parameters:
            - lb (list): lower bounds for parameter variation
            - ub (list): upper bounds for parameter variation
            - set_size (int): number of points in the set

        """
        # Parameters exploration
        param = np.random.rand(set_size, lb.shape[0])

        # Results
        # Metrics computed on the test set
        self.rmse_res = np.zeros(set_size)
        self.iae_alpha_res = np.zeros(set_size)

        # Metrics computed on the train set by LOO
        self.rmse_resloo = np.zeros(set_size)
        self.iae_alpha_resloo = np.zeros(set_size)

        for i in range(set_size):
            # Modify the value of the covparam of the GP model
            self.model.covparam =  (ub - lb) * param[i] + lb

            # Metrics on the train set by LOO
            zpmloo, zpvloo, _ = self.model.loo(self.data.x_train, self.data.z_train, convert_out=False)
            zpvloo[zpvloo <= 0.0] = 1e-5
            self.rmse_resloo[i] = rmse(zpmloo, self.data.z_train)
            self.iae_alpha_resloo[i] = iae_alpha(self.data.z_train, zpmloo, zpvloo)

            # Metrics on the test set
            zpm, zpv = self.model.predict(
                    self.data.x_train, self.data.z_train, self.data.x_test, convert_out=False
                )
            
            zpv[zpv <= 0.0] = 1e-5            
            self.rmse_res[i] = rmse(zpm, self.data.z_test)
            self.iae_alpha_res[i] = iae_alpha(self.data.z_test, zpm, zpv)

        self.model.covparam = np.copy(self.covparam_reml)

        # return the random parameters
        return param