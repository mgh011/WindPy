import numpy as np
from scipy import signal

def friction_velocity(u, v, w):
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

    Author: M. Ghirardelli (built on the original effort of E.Cheynet https://github.com/ECheynet) - Last modified: 13-08-2024
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
    u_star = (uw**2 + vw**2)**0.25

    # Compute the Reynolds stress tensor (R)
    R = np.array([[uu, uv, uw], [uv, vv, vw], [uw, vw, ww]])

    return u_star, R


def obukhovLength(u_star, w, t):
    """
    Computes the Monin-Obukhov length.

    Args:
        u_star: float, friction velocity (pre-calculated)
        w: array-like, 1-D (vertical wind component)
        t: array-like, 1-D (virtual potential temperature)

    Returns:
        L: float, Monin-Obukhov length
        
    Function Description:
        This function computes the Monin-Obukhov length using the friction velocity 
        (u_star), vertical wind component (w), and virtual potential temperature (t).
        It detrends the input data, computes the covariance between w and t, 
        and uses the mean virtual potential temperature to calculate the Monin-Obukhov length (L).
        
    Author: M. Ghirardelli
    Last modified: 24-09-2024
    """
    # Constants
    k = 0.4  # von Kármán constant
    g = 9.81  # gravitational acceleration (m/s^2)
    
    t_mean = t.mean()

    # Detrend the wind components and virtual potential temperature to remove linear trends
    w = signal.detrend(w)
    t = signal.detrend(t)
    
    # Compute the covariance between w and theta_v
    cov_wt = np.mean(w * t)

    # Compute the average virtual potential temperature
    
    # Compute the Monin-Obukhov length (L)
    if cov_wt != 0:  # Avoid division by zero
        L = -(u_star**3 * t_mean) / (k * g * cov_wt)
    else:
        L = np.inf  # In case the heat flux is zero, L is infinite (neutral conditions)

    return L
