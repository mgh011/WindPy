import numpy as np

def despike_mauder(data, q=7):
    """
    Identify outliers using the Median Absolute Deviation (MAD) method, based on the approach
    proposed by Mauder et al., 2012, that relies on a default threshold of 7 MAD deviations. 
    Moreover, a warning is added if more than 5 consecutive outliers are detected.
    Also prints the total number of outliers and their relative proportion.

    Args:
        data: Array-like 1-D of numerical values.
        q: Threshold factor for outlier detection. The default is 7.

    Returns:
        flag_array: A boolean array where 0 indicates that the value is within acceptable limits
                    and 1 indicates that the value is an outlier.

    Requires:
        numpy

    Function Description:
        This function detects outliers in a dataset using the MAD method. A warning is printed if 
        more than 5 consecutive outliers are detected in the data. The total number of outliers and their 
        relative proportion to the total data are also printed. The default threshold factor (q = 7) 
        is based on the methodology proposed by Mauder et al., 2007, which is commonly used in atmospheric 
        science and related fields to filter out spikes in data.

    Author: M. Ghirardelli - Last modified: 13-08-2024
    """
    # Compute the median of the data
    median = np.median(data)
    
    # Compute the MAD
    mad = np.median(np.abs(data - median))
    
    # Compute the lower and upper bounds
    lower_bound = median - q * mad / 0.6745
    upper_bound = median + q * mad / 0.6745
    
    # Identify outliers and flag them
    flag_array = np.where((data < lower_bound) | (data > upper_bound), 1, 0)
    
    # Calculate the total number of outliers
    total_outliers = np.sum(flag_array)
    relative_outliers = total_outliers / len(data)
    
    # Print the total and relative number of outliers
    print(f"Total outliers detected: {total_outliers}")
    print(f"Relative number of outliers: {relative_outliers:.2%}")
    
    # Check for sequences of more than 5 consecutive outliers
    consecutive_count = 0
    for i in range(len(flag_array)):
        if flag_array[i] == 1:
            consecutive_count += 1
            if consecutive_count > 5:
                print("Warning: More than 5 consecutive outliers detected!")
                break
        else:
            consecutive_count = 0
    
    return flag_array
