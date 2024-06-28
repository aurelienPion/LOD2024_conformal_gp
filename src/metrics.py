import gpmp.num as gnp
import numpy as np 
from scipy.stats import norm

def empirical_coverage(z, ci_alpha_inf, ci_alpha_sup):
    """empirical coverage defined by
        \delta_alpha = \sum 1_{z_i \in [ci_alpha_inf[i], ci_alpha_sup[i]]}
    
    Parameters:
        - z (n gnp.array): test features
        - ci_alpha_inf (n*n_alpha gnp.array): lower bound of prediction interval
          for several confidence level alpha
        - ci_alpha_sup (n*n_alpha gnp.array): upper bound of prediction interval
          for several confidence level alpha

    Return (n_alpha gnp.array):
        a list of empirical coverage \delta_alpha
    """
    n = z.shape[0]
    return gnp.sum(gnp.logical_and(ci_alpha_inf <= z, z <= ci_alpha_sup), 0)/n

def rmse(z, z_hat):
    """RMSE between z and z_hat

    Parameters:
        - z (gnp.array): test features
        - z_hat (gnp.array): predicted features
    
    Return (float):
        RMSE
    """
    return gnp.sqrt(gnp.mean((z - z_hat)**2))

def iae_alpha(z, zpm=None, zpv=None, quantiles_minus=None, quantiles_plus=None, n_alpha=20):
    """Compute the IAE for the prediction, using zpm
    and zpv when a GP model is used to compute the prediction interval (PI) or
    quantiles_minus/quantiles_plus when J+ or J+GP is used

    [1] A. Marrel and B. Iooss, “Probabilistic surrogate modeling by Gaussian
    process: A new estimation algorithm for more reliable prediction”.


    Parameters:
        - z (gnp.array): test features
        - zpm (gnp.array): (Optional) predicted mean
        - zpv (gnp.array): (Optional) predicted variance
        - quantiles_minus (gnp.array): (Optional) lower quantiles
        - quantiles_plus (gnp.array): (Optional) upper quantiles
        - n_alpha (int): discretization to compute the integral in IAE formula
    
    Return (float):
        IAE of predictions
    """
    alphas = gnp.linspace(0.02, .98, n_alpha)
    n_test = z.shape[0]

    # iae when J+GP or J+ is used for PI 
    if quantiles_minus is not None:
        ci_alpha_inf = gnp.zeros((n_test, n_alpha))
        ci_alpha_sup = gnp.zeros((n_test, n_alpha))
        for i, alpha_ in enumerate(alphas):
            alpha = alpha_
            n = quantiles_minus.shape[1]
            sup = int(np.ceil((1-alpha)*(n+1)))-1
            if sup > n - 1:
                sup = n - 1
            inf = int(np.floor(alpha*(n+1)))-1
            if inf < 0:
                inf = 0
            ci_alpha_inf[:, i] = gnp.asarray(quantiles_minus[:, inf])
            ci_alpha_sup[:, i] = gnp.asarray(quantiles_plus[:, sup])
    # iae when GP model is used for PI
    else:
        norm_ppf = norm.ppf(1 - alphas/2)
        norm_ppf[alphas == 0.] = -np.inf
        norm_ppf[alphas == 1.] = +np.inf
        quantiles_alphas = gnp.sqrt(zpv[:, None])*norm_ppf[None, :]
        
        ci_alpha_inf = zpm[:, None] - quantiles_alphas
        ci_alpha_sup = zpm[:, None] + quantiles_alphas
    
    # list of several empirical coverage 
    # at several confidence level
    deltas = empirical_coverage(
            z[np.newaxis].T, ci_alpha_inf, ci_alpha_sup)
    
    return gnp.mean(gnp.abs(deltas - (1 - alphas)))