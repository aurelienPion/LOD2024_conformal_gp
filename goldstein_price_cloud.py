import gpmp.num as gnp
import seaborn as sns
import matplotlib.pyplot as plt

import numpy as np

from src.gpmodelmetrics import GPModelMetrics
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

# GP regularity parameter
p = 2

# Bounds for GP parameters when computing the cloud
logsigma_l, logsigma_u = 0.5, -0.4
logrho1_l, logrho1_u = 0.3, -0.2
logrho2_l, logrho2_u = 0.3, -0.2
lb = [logsigma_l, logrho1_l, logrho2_l]
ub = [logsigma_u, logrho1_u, logrho2_u]


model_metrics = GPModelMetrics(d, p, x_min, x_max, goldstein_price)

model_metrics.j_plus_gp_point()

model_metrics.compute_metrics_set(lb, ub)

# Compute the convex hull of the cloud to find the inaccessible area for prediction by GP.
x_curve, lower_curve = compute_convex_lower_hull(
    gnp.asarray(model_metrics.rmse_res),
    gnp.asarray(model_metrics.iae_alpha_res),
    yliminf=0.19,
    xlim=5.5e10,
    ylimsup=0.22,
    nb_p=2 * 10,
)

x_curve_loo, lower_curve_loo = compute_convex_lower_hull(
    gnp.asarray(model_metrics.rmse_resloo),
    gnp.asarray(model_metrics.iae_alpha_resloo),
    yliminf=0.2,
    nb_p=2 * 10,
)

# display the cloud
sns.set_theme(style="ticks", font_scale=1.75)
fig, axs = plt.subplots(1, 2, figsize=(17, 7), sharey=True)

axs[0].plot(
    model_metrics.rmse_resloo,
    model_metrics.iae_alpha_resloo,
    "r*",
    alpha=0.5,
    zorder=-1,
)
axs[0].scatter(
    model_metrics.rmse_remlloo,
    model_metrics.iae_alpha_remlloo,
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

axs[1].set_ylim(-0.01, np.max(model_metrics.iae_alpha_resloo) + 0.01)
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
    model_metrics.rmse_reml,
    model_metrics.iae_alpha_reml,
    s=150,
    c="b",
    marker="s",
    zorder=1,
    label="REML",
)
axs[1].scatter(
    model_metrics.rmse_reml,
    model_metrics.iae_j_plus_gp,
    s=500,
    c="g",
    marker="*",
    label="J+GP method",
    zorder=1,
)

axs[1].plot(
    model_metrics.rmse_res, model_metrics.iae_alpha_res, "r*", alpha=0.5, zorder=-1
)
axs[1].set_xlabel(r"RMSE$(\theta)$")

axs[1].set_title("Metrics computed on the test set")


dy = model_metrics.iae_j_plus_gp - model_metrics.iae_alpha_reml
axs[1].arrow(
    model_metrics.rmse_reml,
    model_metrics.iae_alpha_reml - 0.006,
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
