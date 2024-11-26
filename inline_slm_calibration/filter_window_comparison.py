"""
For different filtering windows, compare the consistency of phase and amplitude detection of a sine wave signal.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal.windows import flattop, gaussian

# Parameters
fs = 500        # Sampling frequency (Hz)
N = 1024        # Number of samples
t = np.arange(N) / fs  # Time vector

# Frequency range to analyze
f_min = 3       # Minimum frequency (Hz)
f_max = 25      # Maximum frequency (Hz)
f_steps = 500   # Number of frequency steps
frequencies = np.linspace(f_min, f_max, f_steps)

# Phase shifts to consider (from 0 to 2*pi)
phase_shifts = np.linspace(0, 2*np.pi, 100)

# Window functions to compare
std_in_samples = N / 2  # Standard deviation in samples for Gaussian window
windows = {
    'Rectangular (No Window)': np.ones(N),
    'Gaussian Window': gaussian(N, std_in_samples),
    'Hann Window': np.hanning(N),
    'Hamming Window': np.hamming(N),
    'Blackman Window': np.blackman(N),
    'Flat-top Window': flattop(N)
}

# Arrays to store standard deviation for magnitude and phase for each window and frequency
magnitude_std = {key: [] for key in windows.keys()}
phase_std = {key: [] for key in windows.keys()}

# Main frequency loop
for f_signal in frequencies:
    # Ground truth magnitude and phase
    ground_truth_magnitude = 1.0  # Since amplitude of sine wave is 1
    ground_truth_phases = phase_shifts

    # Frequency vector for FFT
    freqs = np.fft.fftfreq(N, d=1/fs)

    # Find the index of the closest frequency in the FFT output
    idx = np.argmin(np.abs(freqs - f_signal))

    # Arrays to store magnitudes and phases for each window function
    magnitudes = {key: [] for key in windows.keys()}
    phases = {key: [] for key in windows.keys()}

    # Loop over phase shifts
    for phi in phase_shifts:
        # Generate the sine wave with the current phase shift
        x = np.sin(2 * np.pi * f_signal * t + phi)

        # For each window, compute FFT magnitudes and phases
        for window_name, window in windows.items():
            # Apply the window to the signal
            x_windowed = x * window

            # Perform FFT
            X = np.fft.fft(x_windowed)

            # Normalize the FFT magnitude
            # Correct for the amplitude change due to windowing
            amplitude_correction = np.sum(window) / N
            magnitude = np.abs(X[idx]) / (amplitude_correction * N / 2)

            # Compute the detected phase
            detected_phase = np.angle(X[idx])

            # Store magnitude and phase
            magnitudes[window_name].append(magnitude)
            phases[window_name].append(detected_phase)

    # After looping over phases, compute standard deviation for each window function
    for window_name in windows.keys():
        # Convert phases to numpy array and unwrap
        detected_phases = np.unwrap(phases[window_name])

        # Adjust phases to be relative to the actual phase shifts
        phase_errors = detected_phases - ground_truth_phases
        # Wrap the phase errors between -pi and pi
        phase_errors = (phase_errors + np.pi) % (2 * np.pi) - np.pi

        # Compute magnitude errors
        magnitude_errors = np.array(magnitudes[window_name]) - ground_truth_magnitude

        # Compute standard deviation for magnitude and phase errors
        std_magnitude = np.std(magnitude_errors)
        std_phase = np.std(phase_errors)

        # Store standard deviation values
        magnitude_std[window_name].append(std_magnitude)
        phase_std[window_name].append(std_phase)

# Convert standard deviation dictionaries to arrays
for key in magnitude_std.keys():
    magnitude_std[key] = np.array(magnitude_std[key])
    phase_std[key] = np.array(phase_std[key])

# Plotting standard deviation vs Frequency

# Plot standard deviation of Magnitude
plt.figure(figsize=(12, 6))
for window_name in windows.keys():
    plt.plot(frequencies, magnitude_std[window_name], label=window_name)
plt.title('Standard Deviation of Magnitude Error vs Frequency')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Standard Deviation of Magnitude Error')
plt.yscale('log')
plt.ylim(bottom=1e-7, top=1e-1)
plt.legend()
plt.grid(True)

# Plot standard deviation of Phase
plt.figure(figsize=(12, 6))
for window_name in windows.keys():
    plt.plot(frequencies, phase_std[window_name], label=window_name)
plt.title('Standard Deviation of Phase Error vs Frequency')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Standard Deviation of Phase Error (radians)')
plt.yscale('log')
plt.ylim(bottom=1e-7)
plt.legend()
plt.grid(True)
plt.show()
