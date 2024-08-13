import numpy as np
import pandas as pd


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

    Author: M. Ghirardelli (built on the original effort of E.Cheynet https://github.com/ECheynet) - Last modified: 13-08-2024
    """

    y = pd.Series(u, index=t)
    err1 = np.max(np.abs(y.rolling(Nwin).mean() / u.mean() - 1))
    err2 = np.max(np.abs(y.rolling(Nwin).std() / y.std() - 1))
    if err1 > thres1 or err2 > thres2:
        flag = 1  # time series is non-stationary
    else:
        flag = 0  # time series is stationary
    return err1, err2, flag


def saturationTest(u, v, w, t, sonic_type):
    """
    Check the saturation levels of total wind speed and temperature measurements from sonic anemometers.

    Args:
        u: Array-like 1-D or list of values representing the wind component in the x-direction.
        v: Array-like 1-D or list of values representing the wind component in the y-direction.
        w: Array-like 1-D or list of values representing the wind component in the z-direction.
        t: Array-like 1-D or list of values representing the temperature measurements.
        sonic_type: A string representing the type or model of the sonic anemometer (case insensitive).

    Returns:
        flag_array: A 1-D array of integers where:
                    - 0: Both the total wind speed and temperature are within the saturation limits.
                    - 1: The total wind speed exceeds the saturation limit.
                    - 2: The temperature exceeds the saturation limit.
                    - 3: Both the wind speed and temperature exceed the saturation limits.

    Requires:
        numpy

    Function Description:
        This function checks whether the total wind speed (calculated as sqrt(u^2 + v^2 + w^2)) 
        and temperature (t) measurements from a sonic anemometer are within predefined saturation limits. 
        The function returns a flag array that indicates the status of each measurement:
        - 0 indicates that both the total wind speed and temperature are within limits.
        - 1 indicates that the total wind speed is outside the limits.
        - 2 indicates that the temperature is outside the limits.
        - 3 indicates that both the wind speed and temperature are outside the limits.

        The `sonic_type` argument is case insensitive, allowing flexibility in specifying the model name.

    Author: M. Ghirardelli - Last modified: 13-08-2024
    """
    # Normalize the sonic type to lowercase
    sonic_type = sonic_type.lower()
    
    # Map variations of sonic names to a canonical name
    name_mapping = {
        'young81000': ['rmyoung810000', 'young810000', 'young810000rm'],
        'csat3': ['csat3', 'campbellcsat3', 'csat3campbell'],
        'csat3b': ['csat3b', 'campbellcsat3b', 'csat3bcampbell'],
        # Add other mappings here
    }
    
    # Invert the mapping to get a lookup table
    lookup = {variation: canonical for canonical, variations in name_mapping.items() for variation in variations}
    
    # Check if the provided sonic_type is in the lookup table
    if sonic_type not in lookup:
        raise ValueError(f"The sonic type '{sonic_type}' is unknown. Please check for misspelling or update the sonic list.")
    
    # Get the canonical name from the lookup
    canonical_name = lookup[sonic_type]
    
    # Define saturation limits for different types of sonic anemometers
    limits = {
        'young81000': {'wind_speed': 40, 't': (-50, 50)},
        'csat3': {'wind_speed': 65, 't': (-50, 60)},
        'csat3b': {'wind_speed': 65, 't': (-40, 50)},
        # Add other sonic types with their respective limits here
    }
    
    # Extract the limits for the specified sonic type
    sonic_limits = limits[canonical_name]
    
    # Calculate the total wind speed
    wind_speed = np.sqrt(u**2 + v**2 + w**2)
    
    # Initialize the flag array with 0s (indicating all values are within limits)
    flag_array = np.zeros_like(u, dtype=int)
    
    # Check if the total wind speed exceeds the limit
    wind_out_of_limits = wind_speed > sonic_limits['wind_speed']
    
    # Check if the temperature exceeds the limits
    temp_out_of_limits = (t < sonic_limits['t'][0]) | (t > sonic_limits['t'][1])
    
    # Set flags based on conditions
    flag_array[wind_out_of_limits] = 1
    flag_array[temp_out_of_limits] = 2
    flag_array[wind_out_of_limits & temp_out_of_limits] = 3
    
    return flag_array
