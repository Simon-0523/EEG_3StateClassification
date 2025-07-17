import numpy as np
from scipy.signal import stft

def compute_stft(data, fs=256, nperseg=256, noverlap=147):
    """
    Compute Short-Time Fourier Transform (STFT) of the input data.

    Parameters:
    data (numpy.ndarray): Input signal data.
    fs (int): Sampling frequency of the signal.
    nperseg (int): Length of each segment for STFT.
    noverlap (int): Number of points to overlap between segments.

    Returns:
    f (numpy.ndarray): Array of sample frequencies.
    t (numpy.ndarray): Array of segment times.
    Zxx (numpy.ndarray): STFT of data.
    """
    f, t, Zxx = stft(data, fs=fs, nperseg=nperseg, noverlap=noverlap, boundary=None, padded=False)
    return f, t, Zxx

# # Example usage
# if __name__ == "__main__":
#     # Generate a sample signal
#     fs = 256  # Sampling frequency
#     t = np.linspace(0, 1, fs, endpoint=False)
#     signal = np.sin(2 * np.pi * 50 * t) + np.sin(2 * np.pi * 120 * t)

#     # Compute STFT
#     f, t, Zxx = compute_stft(signal, fs=fs)

#     # Print the results
#     print("STFT result:", Zxx.shape)