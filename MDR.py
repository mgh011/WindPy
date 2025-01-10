import numpy as np

def mdr_vickers(time_series, use_median=False):
    """
    Compute the Multiresolution Decomposition Spectrum (MDR) of a time series using progressive rolling averages (or medians). 
    Based on Vickers&Mahrt 2003 (https://doi.org/10.1175/1520-0426(2003)20<660:TCGATF>2.0.CO;2).

    Args:
        time_series (array-like): 1-D sequence of values representing the time series data.
                                  Its length must be a power of two (2^M).
        use_median (bool, optional): 
            If True, use rolling medians instead of rolling means for the decomposition.
            Defaults to False.

    Returns:
        scales (ndarray): Array of window sizes (2^m) from the largest scale down to the smallest.
        D (ndarray): Array of MDR values (second moment of segment means/medians) at each scale.
                     D[0] corresponds to the largest scale (entire record), 
                     and D[-1] corresponds to the smallest scale (1 point).

    Requires:
        numpy

    Function Description:
        This function performs a multiresolution decomposition of the input time series by iteratively:
          1. Splitting the current residual into segments of width 2^m.
          2. Computing the mean or median of each segment (depending on use_median).
          3. Subtracting these segment-wise averages from the residual at the corresponding indices.
          4. Recording the "energy" (the second moment) of those segment averages as a contribution
             to the total variance at that specific scale.
        
        It starts from the largest scale (m = M, where the segment is the entire series) down to the 
        smallest scale (m = 0, where each segment is a single point). The result is an array of 
        variance contributions D, aka the Multiresolution Decomposition Spectrum (MDR).

        - If use_median = False (default), standard means are used.
        - If use_median = True, the function uses medians, making the decomposition more robust
          to outliers.

        The length of the input time_series must be 2^M. If your data length does not meet this 
        requirement, you can pad it to the nearest power of two or truncate it appropriately.

    Author: M. Ghirardelli - Last modified: 14-08-2024
    """
    
    # Convert input to a NumPy array of floats
    w = np.asarray(time_series, dtype=float)
    N = len(w)
    
    # Verify that N is a power of two
    M = int(np.log2(N))
    if 2**M != N:
        raise ValueError("Length of time_series must be 2^M (a power of two).")

    # Choose the aggregation function (mean or median)
    agg_func = np.median if use_median else np.mean
    
    # Initialize the residual as a copy of the original signal
    residual = w.copy()
    
    # Prepare arrays to store results
    D = []
    scales = []
    
    # Iterate from the largest scale (m = M) down to the smallest (m = 0)
    for m in range(M, -1, -1):
        segment_size = 2**m
        n_segments = N // segment_size
        
        # Store the scale (segment size) for reference
        scales.append(segment_size)
        
        segment_averages = np.zeros(n_segments)
        
        # Compute the mean/median for each segment and subtract it from the residual
        for seg_idx in range(n_segments):
            start = seg_idx * segment_size
            end = start + segment_size
            
            seg_data = residual[start:end]
            mu = agg_func(seg_data)    # mean or median
            segment_averages[seg_idx] = mu
            
            # Subtract the segment's average from the residual
            residual[start:end] = seg_data - mu
        
        # The MDR contribution at scale m is the mean of the squared segment averages
        D_scale = np.mean(segment_averages**2)
        D.append(D_scale)
    
    # Convert to arrays for convenience
    D = np.array(D)
    scales = np.array(scales)
    
    return scales, D

