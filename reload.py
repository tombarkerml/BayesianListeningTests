# From https://gist.github.com/adidenkov/98f286c5c6b4789f140fa79041f282bb

import numpy as np
import warnings
from copy import deepcopy
from skopt.learning import GaussianProcessRegressor
from skopt.optimizer import base_minimize
from sklearn.utils import check_random_state


def reload_simple(result, addtl_calls):
    """ Continue an skopt optimization from its returned OptimizeResult object.
        Will not run through previously queried points, but its behavior won't
        be identical to longer, uninterrupted optimizations.
        PARAMETERS
        ----------
        result [OptimizeResult, scipy object]:
            Result of an skopt optimization, as returned by the optimization method.
            Tested methods:
                dummy_minimize
                forest_minimize
                gbrt_minimize
                gp_minimize
        addtl_calls [int]:
            Number of additional iterations to perform.
        RETURNS
        -------
        result_new [OptimizeResult, scipy object]:
            Updated optimization result returned as an OptimizeResult object.
    """
    args = deepcopy(result.specs['args'])
    args['n_calls'] = addtl_calls
    args['n_random_starts'] = 0
    args['x0'] = deepcopy(result.x_iters)
    args['y0'] = deepcopy(result.func_vals)
    args['random_state'] = deepcopy(result.random_state)
    return base_minimize(**args)


def func_new(params):
    global func_, xs_, ys_
    if (len(xs_) > 0):
        y = ys_.pop(0)
        if (params != xs_.pop(0)):
            warnings.warn("Deviated from expected value, re-evaluating", RuntimeWarning)
        else:
            return y
    return func_(params)


def reload_stable(result, addtl_calls, init_seed=None):
    """ Continue an skopt optimization from its returned OptimizeResult object.
        Consistent with uninterrupted optimizations, but will take more time
        feeding in previously queried points.
        PARAMETERS
        ----------
        result [OptimizeResult, scipy object]:
            Result of an skopt optimization, as returned by the optimization method.
            Tested methods:
                dummy_minimize
                forest_minimize
                gbrt_minimize
                gp_minimize
        addtl_calls [int]:
            Number of additional iterations to perform.

        init_seed [int, RandomState instance, or None (default)]:
            The value passed as 'random_state' to the original optimization.
        RETURNS
        -------
        result_new [OptimizeResult, scipy object]:
            Updated optimization result returned as an OptimizeResult object.
    """
    args = deepcopy(result.specs['args'])
    args['n_calls'] += addtl_calls

    # global b/c I couldn't find a better way to pass
    global func_, xs_, ys_
    func_ = args['func']
    xs_ = list(result.x_iters)
    ys_ = list(result.func_vals)
    args['func'] = func_new

    # recover initial random_state
    if (isinstance(args['random_state'], np.random.RandomState)):
        args['random_state'] = check_random_state(init_seed)
        # if gp_minimize
        if (isinstance(result.specs['args']['base_estimator'], GaussianProcessRegressor)):
            args['random_state'].randint(0, np.iinfo(np.int32).max)

    # run the optimization
    result_new = base_minimize(**args)

    # change the function back, to reload multiple times
    result_new.specs['args']['func'] = func_

    return result_new