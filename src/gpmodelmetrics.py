import gpmp as gp
import gpmp.num as gnp
import numpy as np
import matplotlib.pyplot as plt

from src.data import Data
from src.metrics import iae_alpha, rmse
from src.utils import matern_p, constant_mean, compute_convex_lower_hull
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
        - plot: display the results

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

        # Restore the original covariance parameters
        self.model.covparam = np.copy(self.covparam_reml)

    def evaluate_model_variation(self, lb, ub, set_size=500):
        """Compute a set of metrics (IAE, RMSE) for predictions with
        the GP model when the parameters vary around
        self.covparam_reml.

        Each parameter theta_i varies in [lb_i, ub_i] with a random
        value in between.

        The results are stored in the attributes:
            - On the test set:
                - rmse_res
                - iae_alpha_res
            - On the train set:
                - rmse_resloo
                - iae_alpha_resloo

        Parameters:
            - lb (list): Lower bounds for parameter variation
            - ub (list): Upper bounds for parameter variation
            - set_size (int): Number of points in the set

        """
        # Parameters exploration
        param = np.random.uniform(low=lb, high=ub, size=(set_size, len(lb)))

        # Metrics computed on the test set
        self.rmse_res = np.zeros(set_size)
        self.iae_alpha_res = np.zeros(set_size)

        # Metrics computed on the train set by LOO
        self.rmse_resloo = np.zeros(set_size)
        self.iae_alpha_resloo = np.zeros(set_size)

        for i in range(set_size):
            # Modify the value of the covparam of the GP model
            self.model.covparam = param[i]

            # Metrics on the train set by LOO
            zpmloo, zpvloo, _ = self.model.loo(self.data.x_train, self.data.z_train, convert_out=False)
            zpvloo[zpvloo <= 0.0] = 1e-5
            self.rmse_resloo[i] = rmse(zpmloo, self.data.z_train)
            self.iae_alpha_resloo[i] = iae_alpha(self.data.z_train, zpmloo, zpvloo)

            # Metrics on the test set
            zpm, zpv = self.model.predict(self.data.x_train, self.data.z_train, self.data.x_test, convert_out=False)
            zpv[zpv <= 0.0] = 1e-5
            self.rmse_res[i] = rmse(zpm, self.data.z_test)
            self.iae_alpha_res[i] = iae_alpha(self.data.z_test, zpm, zpv)

        # Restore the original covariance parameters
        self.model.covparam = np.copy(self.covparam_reml)

        return param

    def plot(self, yliminf_loo=0.15, yliminf_test=0.1, xlimsup=7e4, ylimsup=0.25, iae_max=None, x_loo_max=None, path=None):
        """Plot the figure of points (RMSE, IAE) around the prediction
        made when the parameter is selected by REML.

        On the left the metrics are computed by LOO and on the right,
        the figure are computed on the TEST set.

        Parameters:
            yliminf_loo, yliminf_test, xlimsup, ylimsup (int): parameters to compute the inaccessible are by GP models (see compute_convex_lower_hull)
            iae_max, x_loo_max (float): parameters for zoom in around the REML point
            path (str): path top save the figure, if None the figure is not saved
        
        Return:
            The number of points plotted
        """
        # if iae_max and x_loo_max are not none, zoom in around the
        # REML point
        if iae_max is not None and x_loo_max is not None:
            ind_iae_loo = self.iae_alpha_resloo <= iae_max
            ind_rmse_loo = self.rmse_resloo <= x_loo_max
            ind_loo = np.logical_and(ind_iae_loo, ind_rmse_loo)

        # Compute the convex hull of the cloud to find the inaccessible area for prediction by GP.
        x_curve_loo, lower_curve_loo = compute_convex_lower_hull(
            gnp.asarray(self.rmse_resloo[ind_loo]),
            gnp.asarray(self.iae_alpha_resloo[ind_loo]),
            yliminf=yliminf_loo,
        )

        x_curve, lower_curve = compute_convex_lower_hull(
            gnp.asarray(self.rmse_res[ind_loo]),
            gnp.asarray(self.iae_alpha_res[ind_loo]),
            yliminf=yliminf_test,
            xlimsup=xlimsup,
            ylimsup=ylimsup,
        )

        # plot the results
        fig, axs = plt.subplots(1, 2, figsize=(17, 7), sharey=True)

        # loo points
        axs[0].plot(
            self.rmse_resloo[ind_loo],
            self.iae_alpha_resloo[ind_loo],
            "r*",
            alpha=0.5,
            zorder=-1,
        )
        axs[0].scatter(
            self.rmse_remlloo,
            self.iae_alpha_remlloo,
            s=150,
            c="b",
            marker="s",
            zorder=1,
            label="REML",
        )

        # inaccessible area
        axs[0].fill_between(
            np.concatenate((np.linspace(0, np.min(x_curve_loo), 2), x_curve_loo)),
            np.concatenate((iae_max*np.ones(2), lower_curve_loo)),
            np.zeros(2 + lower_curve_loo.shape[0]),
            hatch="/",
            alpha=0.5,
            color="white",
            edgecolor="black",
            zorder=-1,
            label="inaccessible for GP",
        )

        axs[0].set_xlabel(r"RMSE$(\theta)$")
        axs[0].set_ylabel(r"$J_{\rm IAE}(\theta)$")
        axs[0].set_title("Metrics computed by LOO on the train set")
        axs[0].legend(loc='upper right')

        # test points
        # inaccessible area
        axs[1].fill_between(
            np.concatenate((np.array([0, np.min(x_curve)]), x_curve)),
            np.concatenate((np.max(self.iae_alpha_res[ind_loo])*np.ones(2), lower_curve)),
            np.zeros(2 + lower_curve.shape[0]),
            hatch="/",
            alpha=0.5,
            color="white",
            edgecolor="black",
            zorder=-1,
            label="inaccessible for GP",
        )

        axs[1].scatter(
            self.rmse_reml,
            self.iae_alpha_reml,
            s=150,
            c="b",
            marker="s",
            zorder=1,
            label="REML",
        )
        axs[1].scatter(
            self.rmse_reml,
            self.iae_j_plus_gp,
            s=500,
            c="g",
            marker="*",
            label="J+GP method",
            zorder=1,
        )

        axs[1].plot(
            self.rmse_res[ind_loo], self.iae_alpha_res[ind_loo], "r*", alpha=0.5, zorder=-1
        )
        axs[1].set_xlabel(r"RMSE$(\theta)$")

        axs[1].set_title("Metrics computed on the test set")

        dy = self.iae_j_plus_gp - self.iae_alpha_reml
        axs[1].arrow(
            self.rmse_reml,
            self.iae_alpha_reml - 0.006,
            0,
            dy + 0.025,
            head_width=50,
            head_length=0.01,
            fc="k",
            ec="k",
        )

        axs[1].legend(loc='upper right')

        plt.tight_layout()
        plt.show()

        if path is not None:
            fig.savefig(f'{path}/pareto_cp.pdf', bbox_inches='tight')

        return self.rmse_res[ind_loo].shape[0]