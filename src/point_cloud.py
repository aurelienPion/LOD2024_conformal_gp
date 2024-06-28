import gpmp as gp
import gpmp.num as gnp
import numpy as np

from src.data import Data
from src.metrics import iae_alpha, rmse
from src.utils import matern_p, constant_mean
from src.j_plus_gp import j_plus_gp

class Cloud:
    def __init__(self, d, p, x_min, x_max, f, n_train=50, n_test=1500):
        self.d = d
        self.p = p
        self.x_min = x_min
        self.x_max = x_max

        self.n_train = n_train
        self.n_test = n_test
        
        # GP mean and covaraince function
        self.mean = constant_mean
        self.kernel = matern_p
        self.meanparam_sp = None

        # setting the value of f generate the DoE
        self.f = f

    @property
    def f(self):
        return self._f

    @f.setter
    def f(self, f):
        """Set the value of f
        and build the Design of Experiment
        """
        # generate data
        x_test = gnp.asarray(gp.misc.designs.randunif(
            self.d, self.n_test, [self.x_min, self.x_max]))
        x_train = gnp.asarray(gp.misc.designs.randunif(
            self.d, self.n_train, [self.x_min, self.x_max]))

        # compute features
        y_tot = f(gnp.concatenate((x_train, x_test)))

        z_train = gnp.asarray(y_tot[:self.n_train].flatten())
        z_test = gnp.asarray(y_tot[self.n_train:].flatten())

        self.data = Data(x_train=x_train, z_train=
            z_train, x_test=x_test, z_test=z_test)
        
        # create init model
        covparam = np.zeros(self.d+1)  # np.zeros(d+1)
        covparam[0] = np.log(.3)
        covparam[1:] = np.repeat(np.log(0.5), self.d)
        self.model = gp.core.Model(self.mean, self.kernel(self.p), self.meanparam_sp, covparam)

        self.reml_model()
        
        self._f = f

    def reml_model(self):
        """
        - Select the parameter of the GP model
        - Compute the prediction on data.x_test
        - Compute the prediction by LOO on data.x_train
        - Compute the according IAE and RMSE metrics
        """
        self.model = gp.kernel.select_parameters_with_reml(self.model, self.data.x_train, self.data.z_train, info=False)

        self.covparam_reml = np.copy(self.model.covparam)
        
        # predictions on the test set
        zpm, zpv = self.model.predict(self.data.x_train, self.data.z_train, self.data.x_test)
        self.zpm = gnp.asarray(zpm)
        self.zpv = gnp.asarray(zpv)

        # prediction on the train set
        zpmloo, zpvloo, _ = self.model.loo(self.data.x_train, self.data.z_train, self.model.covparam)
        self.zpmloo = gnp.asarray(zpmloo)
        self.zpvloo = gnp.asarray(zpvloo)

        # compute metrics
        self.rmse_reml = rmse(self.zpm, self.data.z_test)
        self.iae_alpha_reml = iae_alpha(self.data.z_test, zpm=self.zpm, zpv=self.zpv)
        self.rmse_remlloo = rmse(zpmloo, self.data.z_train)
        self.iae_alpha_remlloo = iae_alpha(self.data.z_train, zpm=self.zpmloo, zpv=self.zpvloo)

    def j_plus_gp_point(self):
        """Compute IAE of prediction by J+GP
        """
        quantiles_res_plus, quantiles_res_minus = j_plus_gp(self.model, self.data)
        self.iae_j_plus_gp = iae_alpha(self.data.z_test, quantiles_minus=quantiles_res_minus, quantiles_plus=quantiles_res_plus)

    def compute_cloud_points(self, a, b, nb_p=2*10):
        """Compute the a points cloud of 
        metrics (IAE, RMSE) for prediction with
        the GP model self.model when the parameters
        vary around self.covparam_reml

        Each parameter theta_i vary in [a_i*u + b_i],
        where u~U(0, 1)

        Parameters:
            - a (list): parameters for variation
            - b (list): parameters for variation
            - n_rp (int): number of pointsa in the cloud
        """
        # parameters exploration
        param = np.random.rand(nb_p*nb_p*nb_p, 3)

        # results
        # metrics computed on the test set
        self.rmse_res = np.zeros(nb_p*nb_p*nb_p)
        self.iae_alpha_res = np.zeros(nb_p*nb_p*nb_p)

        # metrics computed on the train set by LOO
        self.rmse_resloo = np.zeros(nb_p*nb_p*nb_p)
        self.iae_alpha_resloo = np.zeros(nb_p*nb_p*nb_p)

        for i in range(nb_p*nb_p*nb_p):
            for k, (a_i, b_i) in enumerate(zip(a, b)):
                # modify the value of the covparam of the GP model
                self.model.covparam[k] = self.covparam_reml[k] + a_i*param[i, k] + b_i
                # metrics on train set by LOO
                zpmloo, zpvloo, _ = self.model.loo(self.data.x_train, self.data.z_train, self.model.covparam)
                zpmloo = gnp.asarray(zpmloo)
                zpvloo = gnp.asarray(zpvloo)
                zpvloo[zpvloo <= 0.] = 1e-5
                self.rmse_resloo[i] = rmse(zpmloo, self.data.z_train)
                self.iae_alpha_resloo[i] = iae_alpha(self.data.z_train, zpmloo, zpvloo)

                # metrics on test set
                zpm, zpv = self.model.predict(self.data.x_train, self.data.z_train, self.data.x_test)
                zpm = gnp.asarray(zpm)
                zpv = gnp.asarray(zpv)
                zpv[zpv <= 0.] = 1e-5
                self.rmse_res[i] = rmse(zpm, self.data.z_test)
                self.iae_alpha_res[i] = iae_alpha(self.data.z_test, zpm, zpv)

        self.model.covparam = np.copy(self.covparam_reml)