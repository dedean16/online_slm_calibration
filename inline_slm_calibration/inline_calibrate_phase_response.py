import numpy as np
import matplotlib.pyplot as plt

from calibration_functions import learn_field, detrend
from directories import data_folder
from plot_utilities import plot_results_ground_truth
from import_utilities import import_reference_calibrations, import_inline_calibration


plt.rcParams.update({'font.size': 14})


# === Settings === #
# Paths/globs to measurement files
inline_file = data_folder.joinpath("inline/inline-slm-calibration_t1731676417.npz")
ref_glob = data_folder.glob("tg_fringe/tg-fringe-slm-calibration-r*_noraw.npz")  # Reference

settings = {
    "do_plot": False,
    "do_end_plot": False,
    "plot_per_its": 300,
    "nonlinearity": 2.0,
    "learning_rate": 0.3,
    "iterations": 3000,
}

# === Import and process inline measurement === #
gv0, gv1, measurements = import_inline_calibration(inline_file, settings['do_plot'])
measurements = detrend(gv0, gv1, measurements, do_plot=settings['do_plot'])         # Compensate for photo-bleaching


# === Import and process reference === #
ref_gray, ref_phase, ref_phase_std, ref_amplitude, ref_amplitude_std = \
    import_reference_calibrations(ref_glob, do_plot=settings['do_plot'])

# Learn phase response
nonlin, a, b, P_bg, phase, amplitude = learn_field(
    gray_values0=gv0, gray_values1=gv1, measurements=measurements, **settings
)

print(f"a={a:.4f} (1.0), b={b:.4f}, P_bg={P_bg:.4f}, nonlin = {nonlin:.4f} ({settings['nonlinearity']})")

phase -= phase.mean()

if settings['do_end_plot']:
    # Note: during the last TG fringe measurement, gray values [0, 254] were measured (instead of [0, 255])
    # -> leave out index 255 from plot
    plot_results_ground_truth(gv0[:-1], phase[:-1], amplitude[:-1],
                              ref_gray, ref_phase, ref_phase_std, ref_amplitude, ref_amplitude_std)

    plt.figure()
    amplitude_norm = amplitude / amplitude.mean()
    Er = amplitude_norm * np.exp(1.0j * phase)
    ref_amplitude_norm = ref_amplitude / ref_amplitude.mean()
    E_ref = ref_amplitude_norm * np.exp(1.0j * ref_phase)
    plt.plot(E_ref.real, E_ref.imag, label="Reference")
    plt.plot(Er.real, Er.imag, label="Predicted")
    plt.legend()

    plt.figure()
    plt.plot(np.angle(E_ref.conj() * Er[:-1].numpy()))
    plt.title('Phase difference Ref-Pred')
    plt.xlabel('Gray value')
    plt.ylabel('$\\Delta\\phi$ (rad)')
    plt.show()
