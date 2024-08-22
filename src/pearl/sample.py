from typing import Any

import numpy as np
from numpy.typing import NDArray
import scipy.stats as stats


def draw_from_trunc_norm(
    lower_bound: float,
    upper_bound: float,
    mu: float,
    sigma: float,
    n: int,
    random_state: np.random.RandomState,
) -> NDArray[Any]:
    """
    Return a numpy array filled with n values drawn from a truncated normal with the given
    parameters. If n=0 return an empty numpy array.

    Parameters
    ----------
    lower_bound : float
        Lower bound of truncation.
    upper_bound : float
        Upper bound of truncation.
    mu : float
        Mean value of normal distribution for sampling.
    sigma : float
        Standard deviation of normal distribution for sampling.
    n : int
        Number of values to sample from distribution.
    random_state : np.random.RandomState
        Random State object for random number sampling.

    Returns
    -------
    NDArray[Any]
        numpy array of sampled values.
    """
    y = np.array([])
    if n != 0:
        # normalize the bounds
        lower_bound = (lower_bound - mu) / sigma
        upper_bound = (upper_bound - mu) / sigma
        y = stats.truncnorm.rvs(
            lower_bound, upper_bound, loc=mu, scale=sigma, size=n, random_state=random_state
        )
    return y
