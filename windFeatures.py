import numpy as np
from scipy import signal
from scipy.stats import kurtosis

def momentumAndHeatFlux(f, u, v, w, t):
    """
    Computes the momentum fluxes (uw, vw) and the heat flux (wt) based on the input wind 
    velocity components and temperature data. The data is detrended before calculating the fluxes.
    
    Args:
        f: float, sampling frequency (Hz)
        u: array-like, 1-D, along-wind velocity component (m/s)
        v: array-like, 1-D, across-wind velocity component (m/s)
        w: array-like, 1-D, vertical wind velocity component (m/s)
        t: array-like, 1-D, sonic temperature (K)
           
    Returns:
        uw: float, momentum flux (covariance between along-wind and vertical wind components)
        vw: float, momentum flux (covariance between across-wind and vertical wind components)
        wt: float, heat flux (covariance between vertical wind and temperature)
    
    Function Description:
        This function computes the momentum fluxes (uw, vw) and the heat flux (wt) from the 
        along-wind (u), across-wind (v), vertical wind (w), and temperature (t) data. 
        The input data is detrended to remove any linear trends, and the fluxes are computed as the 
        mean of the products of the detrended variables.
    
    Author: M. Ghirardelli 
    Last modified: 24-09-2024
    """

    # Detrend the wind components and temperature to remove linear trends
    u_detrended = signal.detrend(u)
    v_detrended = signal.detrend(v)
    w_detrended = signal.detrend(w)
    t_detrended = signal.detrend(t)
    
    # Compute momentum fluxes (covariances)
    uw = np.nanmean(u_detrended * w_detrended)  # Covariance between along-wind (u) and vertical wind (w)
    vw = np.nanmean(v_detrended * w_detrended)  # Covariance between across-wind (v) and vertical wind (w)
    
    # Compute heat flux (covariance between vertical wind (w) and temperature (t))
    wt = np.nanmean(w_detrended * t_detrended)
    
    return uw, vw, wt


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


def fluxUncertainty(z, f, u, v, w, t):
    """
    Computes uncertainties in momentum and heat fluxes (uw, vw, wt) based on Stiperski et al. (2016)
    (DOI 10.1007/s10546-015-0103-z).

    Args:
        z: float, measurement height (m)
        f: float, sampling frequency (Hz)
        u: array-like, 1-D, along-wind velocity component (m/s)
        v: array-like, 1-D, across-wind velocity component (m/s)
        w: array-like, 1-D, vertical wind velocity component (m/s)
        t: array-like, 1-D, sonic temperature (K)
        
    Returns:
        a_UW: float, uncertainty in momentum flux for uw (along-wind and vertical)
        a_VW: float, uncertainty in momentum flux for vw (across-wind and vertical)
        a_WT: float, uncertainty in heat flux for wt (vertical wind and sonic temperature)
        
    Requires:
        frictionVelocity function
        numpy
        scipy
        
    Function Description:
        This function computes the uncertainties in momentum and heat fluxes fluxes using the 
        along-wind (u), across-wind (v), and vertical (w) components of wind velocity and sonic temperature (t). 
        The wind components and temperature are detrended, and then the squared covariances (uw, vw and WT) 
        are computed. The uncertainties in these fluxes are then calculated based 
        on the measurement height, sampling frequency, and wind speed.

        The output of the function includes the uncertainties in the fluxes (uw, vw and wt).

    Author: M. Ghirardelli 
    Last modified: 24-09-2024
    """
    
    # Compute horizontal wind speed
    U = np.sqrt(np.nanmean(u**2 + v**2))

    
    # Detrend the wind components and temperature
    u = signal.detrend(u)
    v = signal.detrend(v)
    w = signal.detrend(w)
    t = signal.detrend(t)
    
    # Compute friction velocity (u_star)
    u_star, _ = frictionVelocity(u, v, w)
    
    # Covariance between vertical wind and temperature (w * t)
    cov_wt = np.nanmean(w * t)
    
    # Compute mean squared products of u*w, v*w, and w*t (covariances)
    x_UW = np.nanmean((u * w) ** 2)
    x_VW = np.nanmean((v * w) ** 2)
    x_WT = np.nanmean((w * t) ** 2)
    
    # Calculate measurement duration
    num_data_points = len(u)
    duration = num_data_points / f  # Duration in seconds
    
    # Coefficient for uncertainty calculation
    COEFF = z / (duration * U)
    
    # Initialize uncertainties
    a_UW = np.nan
    a_VW = np.nan
    a_WT = np.nan
    
    # Check if u_star and cov_wt are not zero before calculating uncertainties
    if u_star != 0 and cov_wt != 0:
        # Uncertainty in uw (along-wind and vertical)
        A = (x_UW / u_star ** 4) - 1
        a_UW = np.sqrt(np.maximum(0, COEFF * A))
        
        # Uncertainty in vw (across-wind and vertical)
        A = (x_VW / u_star ** 4) - 1
        a_VW = np.sqrt(np.maximum(0, COEFF * A))   
        
        # Uncertainty in wt (vertical wind and sonic temperature)
        A = (x_WT / cov_wt ** 2) - 1
        a_WT = np.sqrt(np.maximum(0, COEFF * A))
        
    return a_UW, a_VW, a_WT


def variableUncertainty(z, f, u, v, Var):
    """
    Computes uncertainties related to any turbulent variable (e.g. u, v, w, t) based on Stiperski et al. (2016)
    (DOI 10.1007/s10546-015-0103-z).

    Args:
        z: float, measurement height (m)
        f: float, sampling frequency (Hz)
        u: array-like, 1-D, along-wind velocity component (m/s)
        v: array-like, 1-D, across-wind velocity component (m/s)
        Var: array-like, 1-D, turbulent variable (e.g. w, temperature, etc.)
           
    Returns:
        a_VarVar: float, uncertainty in the turbulent variable
    
    Function Description:
        This function computes the uncertainty in a turbulent variable (e.g., wind component, temperature, etc.) 
        based on the measurement height, sampling frequency, and the mean wind speed. It uses the 
        kurtosis of the variable (Var) as part of the calculation and detrends the variable 
        before applying the uncertainty formula.
    
    Author: M. Ghirardelli 
    Last modified: 24-09-2024
    """

    # Compute mean horizontal wind speed
    U = np.sqrt(np.nanmean(u**2 + v**2))
    
    # Detrend the input variable
    var = signal.detrend(Var)
    
    # Calculate measurement duration
    num_data_points = len(var)
    duration = num_data_points / f  # Duration in seconds
    
    # Coefficient for uncertainty calculation
    COEFF = z / (duration * U)
    
    # Calculate the kurtosis of the detrended variable (kurtosis represents the peakedness of the distribution)
    A = kurtosis(var, fisher=False, nan_policy='omit') - 3
    
    # Compute the uncertainty using the formula from Stiperski et al. (2016)
    a_VarVar = np.sqrt(np.maximum(0, 4 * COEFF * A))
    
    return a_VarVar
