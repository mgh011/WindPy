import numpy as np

def nandetrend(x): 
    """
    Detrend a signal while handling NaNs by removing a linear trend from the data.

    Args:
        x: Array-like 1-D or list of values. The input time series or data array
           that may contain NaN values.

    Returns:
        x_detrended: A 1-D array with the linear trend removed. NaN values in the 
                     input remain NaN in the output.

    Requires:
        numpy

    Function Description:
        This nandetrend function is designed to remove a linear trend from a 1D 
        array (x) that may contain NaN values (missing data). The linear trend 
        is essentially a straight line that best fits the valid data points. 
        The function identifies valid (non-NaN) data points, fits a straight line 
        to those points using least-squares regression, and then subtracts the 
        trend, leaving you with the detrended values.

    Author: M. Ghirardelli (based on the original effort of E. Cheynet) 
    Last modified: 24-09-2024
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    kp = np.where(~np.isnan(x))[0]
    if len(kp) >= 2:
        denom = kp[-1] - kp[0]
        denom = denom if denom != 0 else 1
        a = np.vstack(((kp - kp[0]) / denom, np.ones(len(kp)))).T #The design matrix a is constructed for performing a least-squares regression.
        xi_kp = x[kp]
        coeffs, _, _, _ = np.linalg.lstsq(a, xi_kp, rcond=None)
        trend = a @ coeffs
        x_detrended = np.full_like(x, np.nan)
        x_detrended[kp] = xi_kp - trend
    elif len(kp) == 1:
        x_detrended = np.full_like(x, np.nan)
        x_detrended[kp] = 0
    else:
        x_detrended = x.copy()
        
    return x_detrended
