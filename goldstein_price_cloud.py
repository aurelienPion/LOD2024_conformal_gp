import gpmp.num as gnp
import seaborn as sns
import matplotlib.pyplot as plt

import numpy as np

from src.gpmodelmetrics import GPExperiment
from src.functions import goldstein_price
from src.utils import compute_convex_lower_hull

if gnp._gpmp_backend_ == "torch":
    import torch

# Set seed for reproductability
if gnp._gpmp_backend_ == "torch":
    torch.manual_seed(0)
np.random.seed(0)

# Goldstein Price function
d = 2
x_min = gnp.array([-2, -2])
x_max = gnp.array([ 2,  2])

# GP model
p = 2

gpexperiment = GPExperiment(d, p, x_min, x_max, goldstein_price)

# Bounds for GP parameters when computing the cloud
s = 10
logs = np.log(s)
lb = gpexperiment.model.covparam - logs
ub = gpexperiment.model.covparam + logs

# Explanation: We choose ± logs around the covariance parameters to
# allow substantial but controlled variation in the parameters.  This
# range ensures that the parameters can vary significantly (by a
# factor of approximately s in both directions on the original
# scale), which is often sufficient for sensitivity analysis while
# preventing extreme values that might lead to numerical instability
# or non-meaningful results.


# Compute the metrics for a random set of parameters
set_size = 4000
gpexperiment.evaluate_model_variation(lb, ub, set_size=set_size)

# CP
gpexperiment.j_plus_gp_point()

plt.scatter(gpexperiment.rmse_res, gpexperiment.iae_alpha_res)
plt.show()

# Compute the convex hull of the cloud to find the inaccessible area for prediction by GP.
x_curve, lower_curve = compute_convex_lower_hull(
    gnp.asarray(gpexperiment.rmse_res),
    gnp.asarray(gpexperiment.iae_alpha_res),
    yliminf=0.19,
    xlim=5.5e10,
    ylimsup=0.22
)

x_curve_loo, lower_curve_loo = compute_convex_lower_hull(
    gnp.asarray(gpexperiment.rmse_resloo),
    gnp.asarray(gpexperiment.iae_alpha_resloo),
    yliminf=0.2
)

# display the cloud
sns.set_theme(style="ticks", font_scale=1.75)
fig, axs = plt.subplots(1, 2, figsize=(17, 7), sharey=True)

axs[0].plot(
    gpexperiment.rmse_resloo,
    gpexperiment.iae_alpha_resloo,
    "r*",
    alpha=0.5,
    zorder=-1,
)
axs[0].scatter(
    gpexperiment.rmse_remlloo,
    gpexperiment.iae_alpha_remlloo,
    s=150,
    c="b",
    marker="s",
    zorder=1,
    label="REML",
)

axs[0].set_xlabel(r"RMSE$(\theta)$")
axs[0].set_ylabel(r"$J_{\rm IAE}(\theta)$")

axs[0].set_title("Metrics computed by LOO on the train set")

axs[0].fill_between(
    x_curve_loo,
    lower_curve_loo,
    0,
    hatch="/",
    alpha=0.5,
    color="white",
    edgecolor="black",
    zorder=-1,
    label="inaccessible for GP",
)
axs[0].legend()

axs[1].set_ylim(-0.01, np.max(gpexperiment.iae_alpha_resloo) + 0.01)
axs[1].fill_between(
    x_curve,
    lower_curve,
    0,
    hatch="/",
    alpha=0.5,
    color="white",
    edgecolor="black",
    zorder=-1,
    label="inaccessible for GP",
)

axs[1].scatter(
    gpexperiment.rmse_reml,
    gpexperiment.iae_alpha_reml,
    s=150,
    c="b",
    marker="s",
    zorder=1,
    label="REML",
)
axs[1].scatter(
    gpexperiment.rmse_reml,
    gpexperiment.iae_j_plus_gp,
    s=500,
    c="g",
    marker="*",
    label="J+GP method",
    zorder=1,
)

axs[1].plot(
    gpexperiment.rmse_res, gpexperiment.iae_alpha_res, "r*", alpha=0.5, zorder=-1
)
axs[1].set_xlabel(r"RMSE$(\theta)$")

axs[1].set_title("Metrics computed on the test set")


dy = gpexperiment.iae_j_plus_gp - gpexperiment.iae_alpha_reml
axs[1].arrow(
    gpexperiment.rmse_reml,
    gpexperiment.iae_alpha_reml - 0.006,
    0,
    dy + 0.025,
    head_width=50,
    head_length=0.01,
    fc="k",
    ec="k",
)

axs[1].legend()
plt.tight_layout()
plt.show()
