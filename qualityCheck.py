def stationaryTest(u, t, Nwin, thres1, thres2):
    """
    Assess the stationarity of a time series using a moving window function.

    Args:
        u: Array-like 1-D or list of values
        t: The corresponding time indices for the time series u.
        Nwin: The size of the moving window used to calculate rolling statistics.
        thres1: The threshold for the rolling mean test to consider the series 
                non-stationary.
        thres2: The threshold for the rolling standard deviation test to consider 
                the series non-stationary.

    Returns:
        err1: The maximum relative deviation of the rolling mean from the global 
              mean.
        err2: The maximum relative deviation of the rolling standard deviation 
              from the global standard deviation.
        flag: An indicator (1 or 0) signifying whether the time series is 
              stationary (0) or non-stationary (1).

    Requires:
        pandas,
        numpy
        
    Function Description:
            This function is useful in time series analysis when you want to 
            determine whether the statistical properties of a series 
            (like mean and variance) remain constant over time, which is a key 
            assumption in many statistical models. Non-stationary time series 
            often require transformation (e.g., differencing, detrending) 
            before further analysis.

    Author: M. Ghirardelli (on E.Cheynet) - Last modified: 13-08-2024
    """

    y = pd.Series(u, index=t)
    err1 = np.max(np.abs(y.rolling(Nwin).mean() / u.mean() - 1))
    err2 = np.max(np.abs(y.rolling(Nwin).std() / y.std() - 1))
    if err1 > thres1 or err2 > thres2:
        flag = 1  # time series is non-stationary
    else:
        flag = 0  # time series is stationary
    return err1, err2, flag


 
