import gpmp as gp
import gpmp.num as gnp
import numpy as np

from src.data import Data

def j_plus_gp(model: gp.core.Model, data: Data, normalized=True):
    """Compute sequences to build the prediction interval (PI) of J+ [1] and
        J+GP [2] algorithm

    [1] R. F. Barber, E. J. Candes, A. Ramdas, and R. J. Tibshirani, “Predictive
    inference with the jackknife+.” arXiv, May 29, 2020. Accessed: May 18, 2024.
    [Online]. Available: http://arxiv.org/abs/1905.02928

    [2] E. Jaber et al., “Conformal Approach To Gaussian Process Surrogate
    Evaluation With Coverage Guarantees.” arXiv, Jan. 15, 2024. Accessed: Apr.
    15, 2024. [Online]. Available: http://arxiv.org/abs/2401.07733

    Parameters:
        - model (gp.core.Model): GP model
        - data (Data): data, train + test set
        - normalized (bool): If true return J+ sequences, if false return J+GP
          sequences

    Return (tuple):
        sequences to build the PIs /!\ to compute the prediction interval at a
        given confidence level alpha you need to compute the quantiles of the
        resutls 
    """
    zpm_loo, zpv_loo, _ = model.loo(data.x_train, data.z_train)

    n_train = data.z_train.size()[0]
    n_test = data.x_test.size()[0]
    quantiles_plus = np.zeros((n_test, n_train))
    quantiles_minus = np.zeros((n_test, n_train))

    # compute the non-conformity scores
    if normalized:
        # J+GP
        R_i = np.abs((data.z_train - zpm_loo))/np.sqrt(zpv_loo)
    else:
        # J+
        R_i = np.abs(data.z_train - zpm_loo)
    
    if gnp._gpmp_backend_ == "torch":
        R_i = R_i.numpy()

    for i in range(n_train):
        # Remove the i-th point from the training set
        x_train = np.delete(data.x_train, i, axis=0)
        z_train = np.delete(data.z_train, i, axis=0)

        # compute the prediction zith the LOO model on the test set
        zpmi, zpvi = model.predict(x_train, z_train, data.x_test)

        # sequences to build the PI
        if normalized:
            quantiles_plus[:, i] = zpmi + np.sqrt(zpvi) * R_i[i] 
            quantiles_minus[:, i] = zpmi - np.sqrt(zpvi) * R_i[i]
        else:
            quantiles_plus[:, i] = zpmi + R_i[i]
            quantiles_minus[:, i] = zpmi - R_i[i]
    
    quantiles_res_plus = np.sort(quantiles_plus)
    quantiles_res_minus = np.sort(quantiles_minus)

    return quantiles_res_plus, quantiles_res_minus
