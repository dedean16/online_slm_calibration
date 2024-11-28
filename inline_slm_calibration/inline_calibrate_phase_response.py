"""
Process and plot inline calibration measurements. Compare with reference method that uses a Twymann-Green interferometer
with Fourier fringe analysis. Before running this script, please ensure that data_folder (defined in directories.py)
points to a valid folder and contains the measurement data files specified in the settings of this script.
"""
# External
import matplotlib.pyplot as plt

# Internal
from calibration_functions import learn_field, detrend
from directories import data_folder
from plot_utilities import plot_results_ground_truth
from import_utilities import import_reference_calibrations, import_inline_calibration


# === Settings === #
# Paths/globs to measurement data files
inline_file = data_folder.joinpath("inline/inline-slm-calibration_t1731676417.npz")     # Our inline method
ref_glob = data_folder.glob("tg_fringe/tg-fringe-slm-calibration-r*_noraw.npz")         # Reference TG fringe

plt.rcParams.update({'font.size': 14})
settings = {
    "do_plot": False,
    "do_end_plot": True,
    "plot_per_its": 500,
    "nonlinearity": 2.0,
    "learning_rate": 0.3,
    "iterations": 50000,
}

# === Import and process inline measurement === #
gv0, gv1, measurements = import_inline_calibration(inline_file, settings['do_plot'])

# === Import and process reference === #
ref_gray, ref_phase, ref_phase_std, ref_amplitude, ref_amplitude_std = \
    import_reference_calibrations(ref_glob, do_plot=settings['do_plot'])

# Learn phase response
nonlin, a, b, P_bg, phase, amplitude = learn_field(
    gray_values0=gv0, gray_values1=gv1, measurements=measurements, **settings)

print(f"a={a:.4f} (1.0), b={b:.4f}, P_bg={P_bg:.4f}, nonlin = {nonlin:.4f} ({settings['nonlinearity']})")

if settings['do_end_plot']:
    # Note: during the last TG fringe measurement, gray values [0, 254] were measured (instead of [0, 255])
    # -> leave out index 255 from plot
    plot_results_ground_truth(gv0[:-1], phase[:-1], amplitude[:-1],
                              ref_gray, ref_phase, ref_phase_std, ref_amplitude, ref_amplitude_std)
