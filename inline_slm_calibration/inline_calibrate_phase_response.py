"""
Process and plot inline calibration measurements. Compare with reference method that uses a Twymann-Green interferometer
with Fourier fringe analysis. Before running this script, please ensure that data_folder (defined in directories.py)
points to a valid folder and contains the measurement data files specified in the settings of this script.
"""
# External (3rd party)
import numpy as np
import matplotlib.pyplot as plt

# Internal
from calibration_functions import learn_field, detrend
from directories import data_folder
from plot_utilities import plot_results_ground_truth
from import_utilities import import_reference_calibrations, import_inline_calibration


# === Settings === #
# Paths/globs to measurement data files
inline_glob = data_folder.glob("inline/inline-slm-calibration_t*.npz")                  # Our inline method
ref_glob = data_folder.glob("tg_fringe/tg-fringe-slm-calibration-r*_noraw.npz")         # Reference TG fringe

plt.rcParams.update({'font.size': 14})
settings = {
    "do_plot": False,
    "do_end_plot": True,
    "do_noise_plot": True,
    "plot_per_its": 300,
    "nonlinearity": 2.0,
    "learning_rate": 0.3,
    "iterations": 3000,
}

# === Import and process reference === #
ref_gray, ref_phase, ref_phase_std, ref_amplitude_norm, ref_amplitude_norm_std = \
    import_reference_calibrations(ref_glob, do_plot=settings['do_plot'])

# Initialize
inline_files = list(inline_glob)
inline_gray_all = [None] * len(inline_files)
inline_phase_all = [None] * len(inline_files)
inline_amp_all = [None] * len(inline_files)
inline_amp_norm_all = [None] * len(inline_files)
nonlin_all = [None] * len(inline_files)
measurements_all = [None] * len(inline_files)
measurement_vars_all = [None] * len(inline_files)

# Process files
for n_f, filepath in enumerate(inline_files):
    # === Import and process inline measurement === #
    gv0, gv1, measurements, measurement_variances = import_inline_calibration(filepath, settings['do_plot'])
    measurements_detrended = detrend(gv0, gv1, measurements, do_plot=settings['do_plot'])  # Compensate for bleaching

    # Learn phase response
    nonlin, a, b, P_bg, phase, amplitude, amplitude_norm = learn_field(
        gray_values0=gv0, gray_values1=gv1, measurements=measurements_detrended, **settings)

    print(f"a={a:.4f} (1.0), b={b:.4f}, P_bg={P_bg:.4f}, nonlin = {nonlin:.4f} ({settings['nonlinearity']})")

    # Store results in array
    inline_gray_all[n_f] = gv0
    inline_amp_all[n_f] = amplitude
    inline_amp_norm_all[n_f] = amplitude_norm
    inline_phase_all[n_f] = phase
    nonlin_all[n_f] = nonlin
    measurements_all[n_f] = measurements
    measurement_vars_all[n_f] = measurement_variances

# Summarize results with median and std
inline_gray = inline_gray_all[0]
inline_amplitude_norm = np.median(inline_amp_norm_all, axis=0)
inline_amplitude_norm_std = np.std(inline_amp_norm_all, axis=0)
inline_amplitude_norm_std_per_measurement = np.std(inline_amp_norm_all, axis=1)
inline_phase = np.median(inline_phase_all, axis=0)
inline_phase -= inline_phase.mean()
inline_phase_std = np.std(inline_phase_all, axis=0)

print([f'{amp_std:.2g}' for amp_std in inline_amplitude_norm_std_per_measurement])
n_max = np.argmax(inline_amplitude_norm_std_per_measurement)
print(f'σ_A={inline_amplitude_norm_std_per_measurement[n_max]:.2g} for {inline_files[n_max]}')

if settings['do_noise_plot']:
    plt.figure()
    for n_f, filepath in enumerate(inline_files):
        plt.plot(measurements_all[n_f].flatten(), np.sqrt(measurement_vars_all[n_f].flatten()), '.', label=f'Meas.{n_f + 1}')

    plt.title('Noise analysis')
    plt.xlabel('$\\langle I\\rangle - I_{bg}$')
    plt.ylabel('$\\sigma_{I}$')
    plt.legend()

if settings['do_end_plot']:
    # Note: during the last TG fringe measurement, gray values [0, 254] were measured (instead of [0, 255])
    # -> leave out index 255 from plot
    plot_results_ground_truth(
        inline_gray[:-1], inline_phase[:-1], inline_phase_std[:-1], inline_amplitude_norm[:-1], inline_amplitude_norm_std[:-1],
        ref_gray, ref_phase, ref_phase_std, ref_amplitude_norm, ref_amplitude_norm_std)
