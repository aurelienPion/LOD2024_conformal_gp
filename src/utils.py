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


def compute_convex_lower_hull(
    rmse_res, iae_alpha_res, yliminf=0.25, ylimsup=0.0, xlimsup=np.inf
):
    """
    Compute the convex hull of a 2D set of points defined by (rmse_res, iae_alpha_res)
    and return only the part below the set of points.

    Parameters:
    ----------
    rmse_res : np.ndarray
        Array of RMSE (Root Mean Square Error) results.
    iae_alpha_res : np.ndarray
        Array of IAE (Integrated Absolute Error) results.
    yliminf : float, optional
        Lower bound for the y-values (IAE) to include in the convex hull (default is 0.25).
    ylimsup : float, optional
        Upper bound for the y-values (IAE) to include in the convex hull (default is 0).
    xlimsup : float, optional
        Upper bound for the x-values (RMSE) to include in the convex hull (default is np.inf).

    Returns:
    -------
    x_curve : np.ndarray
        Sorted x-values (RMSE) of the convex hull points that lie below the set of points.
    lower_curve : np.ndarray
        Corresponding y-values (IAE) of the convex hull points that lie below the set of points.

    Notes:
    -----
    The function computes the convex hull of the given 2D points and filters out the points
    that do not meet the specified y-value (IAE) and x-value (RMSE) limits. The resulting points
    form the lower part of the convex hull, which is returned as sorted arrays of x and y values.
    """
    set_size = rmse_res.shape[0]
    two_d_arrays = gnp.zeros((set_size, 2))
    two_d_arrays[:, 0] = rmse_res
    two_d_arrays[:, 1] = iae_alpha_res
    
    hull = ConvexHull(two_d_arrays)

    lower_curve = []
    x_curve = []
    for vertex in hull.vertices:
        x, y = two_d_arrays[vertex, 0], two_d_arrays[vertex, 1]
        # only keep the part below the set of points
        if y < yliminf and x < xlimsup:
            lower_curve.append(y)
            x_curve.append(x)
        elif x > xlimsup and y < ylimsup:
            lower_curve.append(y)
            x_curve.append(x)

    x_curve = np.array(x_curve)
    lower_curve = np.array(lower_curve)

    ind = np.argsort(x_curve)
    x_curve = x_curve[ind]
    lower_curve = lower_curve[ind]
    return x_curve, lower_curve
