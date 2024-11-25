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


def psd(freqs, fft_values, sample_rate):
    """
    Compute the Power Spectral Density (PSD) from the FFT values.

    Args:
        freqs: Array of frequency bins in Hz.
        fft_values: Array of complex FFT values.
        sample_rate: Sampling rate of the time series in Hz.
        
    Requires:
        numpy
        
    Returns:
        psd: Array of PSD values corresponding to the positive frequency bins.

    Function Description:
        This function computes the Power Spectral Density (PSD) of the signal by taking the squared 
        magnitude of the FFT values, normalizing them by the sample rate, and adjusting for the length 
        of the time series. The PSD is only calculated for the positive frequencies.

    Author: M. Ghirardelli - Last modified: 14-08-2024
    """
    power = np.abs(fft_values)**2
    psd = power / (len(freqs) * sample_rate)
    psd = psd[:len(freqs)//2] * 2
    psd[0] = psd[0] / 2
    return psd

def plot_psd(freqs, psd, **kwargs):
    """
    Plot the Power Spectral Density (PSD) on a log-log scale.

    Args:
        freqs: Array of frequency bins in Hz.
        psd: Array of PSD values in SI units (W/Hz or dB/Hz).
        **kwargs: Additional keyword arguments for custom plot labels (e.g., xlabel, ylabel).

    Requires:
        matoplotlib

    Function Description:
        This function plots the PSD on a log-log scale. It shows only the positive frequencies.
        Users can customize labels and other plot attributes via additional keyword arguments.

    Author: M. Ghirardelli - Last modified: 14-08-2024
    """
    positive_freqs = freqs[:len(freqs)//2]

    plt.figure(figsize=(10, 6))
    plt.loglog(positive_freqs, psd)
    
    plt.xlabel(kwargs.get('xlabel', 'Frequency (Hz)'))
    plt.ylabel(kwargs.get('ylabel', 'Power/Frequency (W/Hz)' if 'ylabel' not in kwargs else kwargs['ylabel']))
    plt.title(kwargs.get('title', 'Power Spectral Density'))
    plt.grid(True)
    
    for key, value in kwargs.items():
        if key not in ['xlabel', 'ylabel', 'title']:
            plt.gca().set(**{key: value})
    
    plt.show()

def binSpectra(f, Su, Nb):
    """
        binSpectra(f,Su,Nb) smoothens the estimated power spectral density (PSD) 
    estimates by binning over logarithmic-spaced bins defined by the array newF

        Input:
                f: [Nx1] array:  original frequency vector

                Su: [Nx1] array:  PSD estimate

                Nb: [1 x M] or [Mx1] array:  target bins (as a frequency vector)


        Output
                newF: [1 x B] array: new frequency vector

                newS: [1 x B] array: new PSD estimate


        Author: E. Cheynet - UiB - Last modified: 10-03-2023
        """

    newF0 = np.logspace(np.log10(f[1]*0.8), np.log10(f[-1]*1.1), Nb)
    newSu, newF, ind = binned_statistic(f, Su,
                                        statistic='median',
                                        bins=newF0)
    newF = newF[0:-1]
    newF = newF[~np.isnan(newSu)]
    newSu = newSu[~np.isnan(newSu)]
    return newSu, newF

frequency = 32
