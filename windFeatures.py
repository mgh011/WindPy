import numpy as np
from scipy import signal

def frictionVelocity(u, v, w):
    """
    Computes the friction velocity.

    Args:
        u: array-like, 1-D (along-wind component)
        v: array-like, 1-D (across-wind component)
        w: array-like, 1-D (vertical wind component)

    Returns:
        u_star: float, friction velocity
        R: 3x3 array, Reynolds's stress tensor
        
    Function Description:
        This function takes in the along-wind component (u), the across-wind 
        component (v), and the vertical wind component (w) of the wind velocity 
        as input parameters. It then detrends the input data and computes 
        the variance and covariance of the three components. The friction 
        velocity (u_star) is then computed as the square root of the sum of 
        the squares of the covariances (uw and vw). The Reynolds stress tensor 
        (R) is also computed using the variance and covariance values. 
        The output of the function is the friction velocity (u_star) 
        and the Reynolds stress tensor (R).

    Author: M. Ghirardelli (on original contribution by E. Cheynet) - Last modified: 13-08-2024
    """

    # Detrend the wind components to remove linear trends
    u = signal.detrend(u)
    v = signal.detrend(v)
    w = signal.detrend(w)

    # Compute variances
    uu = np.var(u)
    vv = np.var(v)
    ww = np.var(w)

    # Compute covariances
    uv = np.mean(u * v)
    uw = np.mean(u * w)
    vw = np.mean(v * w)

    # Compute the friction velocity (u_star)
    u_star = np.sqrt(uw**2 + vw**2)

    # Compute the Reynolds stress tensor (R)
    R = np.array([[uu, uv, uw], [uv, vv, vw], [uw, vw, ww]])

    return u_star, R
