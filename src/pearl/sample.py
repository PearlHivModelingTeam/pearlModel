import numpy as np
import scipy.stats as stats


def draw_from_trunc_norm(
    a: float,
    b: float,
    mu: float,
    sigma: float,
    n: int,
    random_state: np.random.RandomState,
) -> np.array:
    """Return a numpy array filled with n values drawn from a truncated normal with the given parameters. If n=0 return
    an empty numpy array.
    """
    y = np.array([])
    if n != 0:
        a_mod = (a - mu) / sigma
        b_mod = (b - mu) / sigma
        y = stats.truncnorm.rvs(
            a_mod, b_mod, loc=mu, scale=sigma, size=n, random_state=random_state
        )
    return y
