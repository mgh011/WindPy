import numpy as np
import matplotlib.pyplot as plt

def fft(time_series, sample_rate):
    """
    Compute the Fast Fourier Transform (FFT) of a time series.

    Args:
        time_series: Array-like 1-D or list of values representing the time series data.
        sample_rate: Sampling rate of the time series in Hz.

    Returns:
        freqs: Array of frequency bins.
        fft_values: Array of complex values representing the amplitude and phase of each frequency component.

    Requires:
        numpy

    Function Description:
        This function computes the Fast Fourier Transform (FFT) of a time series, which converts 
        the time-domain signal into its frequency components. The function returns the frequency bins 
        and the corresponding FFT values.

    Author: M. Ghirardelli - Last modified: 13-08-2024
    """
    # Compute the FFT
    fft_values = np.fft.fft(time_series)
    
    # Compute the corresponding frequencies
    freqs = np.fft.fftfreq(len(time_series), d=1/sample_rate)
    
    return freqs, fft_values

def plot_fft(freqs, fft_values, **kwargs):
    """
    Plot the magnitude of the FFT of a time series on a log-log scale.

    Args:
        freqs: Array of frequency bins.
        fft_values: Array of complex values representing the amplitude and phase of each frequency component.
        **kwargs: Additional keyword arguments for custom plot labels (e.g., xlabel, ylabel).

    Requires:
        matplotlib

    Function Description:
        This function plots the magnitude of the FFT result on a log-log scale. It typically shows only the positive 
        frequencies since the FFT is symmetric for real-valued time series. Users can customize labels 
        and other plot attributes via additional keyword arguments.

    Author: M. Ghirardelli - Last modified: 13-08-2024
    """
    # Compute the magnitude of the FFT (absolute value)
    magnitude = np.abs(fft_values)
    
    # Only plot the positive frequencies
    positive_freqs = freqs[:len(freqs)//2]
    positive_magnitude = magnitude[:len(magnitude)//2]
    
    # Plot the FFT result on a log-log scale
    plt.figure(figsize=(10, 6))
    plt.loglog(positive_freqs, positive_magnitude)
    
    # Set default labels
    xlabel = kwargs.get('xlabel', 'Frequency (Hz)')
    ylabel = kwargs.get('ylabel', 'Magnitude')
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(kwargs.get('title', 'Magnitude of FFT'))
    plt.grid(True)
    
    # Apply any additional keyword arguments to plt functions
    for key, value in kwargs.items():
        if key not in ['xlabel', 'ylabel', 'title']:
            plt.gca().set(**{key: value})
    
    plt.show()
