def nandetrend(x): 
    """
    This nandetrend function is designed to remove a linear trend from a 1D array (x) that may contain NaN values (missing data). 
    The linear trend is essentially a straight line that best fits the data. 
    The function works by identifying valid (non-NaN) data points, fitting a straight line to those points using least-squares regression, 
    and then subtracting this trend from the original data, leaving you with the "detrended" values.
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    kp = np.where(~np.isnan(x))[0]
    if len(kp) >= 2:
        denom = kp[-1] - kp[0]
        denom = denom if denom != 0 else 1
        a = np.vstack(((kp - kp[0]) / denom, np.ones(len(kp)))).T
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
