import gpmp as gp
import gpmp.num as gnp
import numpy as np

from scipy.spatial import ConvexHull


def matern_p(p):
    """wrapper for kernel Matern
    to change the regularity parameter
    """
    def k(x, z, covparam, pairwise=False):
        K = gp.kernel.maternp_covariance(x, z, p, covparam, pairwise)
        return K
    return k

def constant_mean(x, param):
    return gnp.ones((x.shape[0], 1))


def compute_convex_lower_hull(rmse_res, iae_alpha_res, yliminf=0.23, ylimsup=0, xlim=np.inf, nb_p=2*10):
    """Compute the convex hull of a 2D cloud defined by
    (rmse_res, iae_alpha_res)

    Return only the part below the cloud.
    """
    two_d_arrays = gnp.zeros((nb_p*nb_p*nb_p, 2))
    two_d_arrays[:, 0] = rmse_res
    two_d_arrays[:, 1] = iae_alpha_res
    hull = ConvexHull(two_d_arrays)

    lower_curve = []
    x_curve = []
    for vertex in hull.vertices:
        x, y = two_d_arrays[vertex, 0], two_d_arrays[vertex, 1]
        # only keep the part below the cloud
        if y < yliminf and x < xlim:
            lower_curve.append(y)
            x_curve.append(x)
        elif x > xlim and y < ylimsup:
            lower_curve.append(y)
            x_curve.append(x)

    x_curve = np.array(x_curve)
    lower_curve = np.array(lower_curve)

    ind = np.argsort(x_curve)
    x_curve = x_curve[ind]
    lower_curve = lower_curve[ind]
    return x_curve, lower_curve


