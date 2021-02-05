#From https://gist.github.com/adidenkov/98f286c5c6b4789f140fa79041f282bb

from reload import reload_stable

import numpy as np
from skopt import forest_minimize
from skopt.space import Real
from skopt.utils import use_named_args
from skopt.plots import plot_objective
import matplotlib.pyplot as plt


space = [Real(-2, 2, "uniform", name='x'),
         Real(-1, 3, "uniform", name='y')]

@use_named_args(space)
def rosenbrock(x, y):
    # true minimum is 0 at (a, a^2)
    a = 1
    b = 100
    return (a-x)**2 + b*(y-x*x)**2

result1 = forest_minimize(rosenbrock, space, n_calls=25, random_state=0)
result1 = reload_stable(result1, addtl_calls=25, init_seed=0)

result2 = forest_minimize(rosenbrock, space, n_calls=50, random_state=0)

if(np.allclose(result1.func_vals, result2.func_vals)):
    print("TEST PASSED")
    print("OPTIMUM:", result1.fun, "at", result1.x)
else:
    print("TEST FAILED")


# predicted function
plot_objective(result1)

# true function
class dummy_model:
    def predict(self, points):
        return [rosenbrock(point) for point in points]
result1.models[-1] = dummy_model()
plot_objective(result1)

plt.show()