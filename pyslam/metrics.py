import numpy as np


def rmse(errs):
    """Root mean squared error (RMSE)"""
    errs = np.atleast_2d(errs)
    se = errs**2
    mse = np.mean(se, axis=1)
    return np.squeeze(np.sqrt(mse))


def armse(errs):
    """Average oot mean squared error (ARMSE)"""
    return np.squeeze(np.mean(np.atleast_2d(rmse(errs)), axis=1))
