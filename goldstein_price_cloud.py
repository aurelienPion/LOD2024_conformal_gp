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
x_max = gnp.array([2, 2])

n_train = 150
n_test = 1500

# GP model
p = 2
gpexperiment = GPExperiment(
    d, p, x_min, x_max, goldstein_price, n_train=n_train, n_test=n_test
)

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
set_size = 20000
covparam_set = gpexperiment.evaluate_model_variation(lb, ub, set_size=set_size)


# Conformal prediction
gpexperiment.j_plus_gp_point()

# set the style
sns.set_theme(style="ticks", font_scale=1.75)
iae_max = 0.3
x_loo_max = 5e4

# save the figure (if path is None the Figure is not savec)
path = None

nb_points_plotted = gpexperiment.plot(iae_max=iae_max, x_loo_max=x_loo_max, path=path)

print(f"number of points plotted = {nb_points_plotted}")