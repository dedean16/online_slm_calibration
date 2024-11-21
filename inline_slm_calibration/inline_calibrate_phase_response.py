import numpy as np
import matplotlib.pyplot as plt
import h5py

from helper_functions import get_dict_from_hdf5
from calibration_functions import learn_field, detrend
from directories import data_folder
from plot_utilities import plot_results_ground_truth, plot_result_feedback_fit


plt.rcParams.update({'font.size': 14})


# === Settings === #
# Import feedback measurements and reference phase response
inline_file = data_folder.joinpath("inline/inline-slm-calibration_t1731670344.npz")
ref_glob = data_folder.glob("tg_fringe/tg-fringe-slm-calibration-r*_noraw.npz")  # Reference

settings = {
    "do_plot": True,
    "do_end_plot": True,
    "plot_per_its": 50,
    "nonlinearity": 2.0,
    "learning_rate": 0.3,
    "iterations": 4000,
    "smooth_loss_factor": 0.0,
    "balance_factor": 0.0,
}

# === Import and process inline measurement === #
npz_data = np.load(inline_file)
measurements = npz_data["frames"].mean(axis=(0, 1, 2)) - npz_data['dark_frame'].mean()
gv0 = npz_data['gray_values1'][0]
gv1 = npz_data['gray_values2'][0]

# Compensate for photo-bleaching
measurements = detrend(gv0, gv1, measurements)


# === Import and process reference === #
ref_files = list(ref_glob)
ref_gray_all = [None] * len(ref_files)
ref_phase_all = [None] * len(ref_files)
ref_amp_all = [None] * len(ref_files)

for n_f, filepath in enumerate(ref_files):
    npz_data = np.load(filepath)
    ref_gray_all[n_f] = npz_data["gray_values"]
    ref_phase = np.unwrap(np.angle(npz_data["field"]))
    ref_phase -= ref_phase.mean()
    ref_phase_all[n_f] = ref_phase * np.sign(ref_phase[-1] - ref_phase[0])
    ref_amp_all[n_f] = np.abs(npz_data["field"])

ref_gray = ref_gray_all[0]
ref_amplitude = np.median(ref_amp_all, axis=0)
ref_phase = np.median(ref_phase_all, axis=0)
ref_phase -= ref_phase.mean()
ref_phase_std = np.std(ref_phase_all, axis=0)

# plt.plot(np.abs(ref_field_all).T)
plt.figure()
plt.plot(np.asarray(ref_phase_all).T)
plt.xlabel('Gray value')
plt.ylabel('Phase')
plt.title('TG fringe calibration, unwrapped\ncompensated for sign and offset')
plt.pause(0.01)


# Learn phase response
nl, a, b, s_bg, phase, amplitude = learn_field(
    gray_values0=gv0, gray_values1=gv1, measurements=measurements, **settings
)

print(f"a={a:.4f} (1.0), b={b:.4f}, s_bg={s_bg:.4f}, nl = {nl:.4f} ({settings['nonlinearity']})")

phase -= phase.mean()

# Note: during the last TG fringe measurement, gray values [0, 254] were measured (instead of [0, 255])
# -> leave out index 255 from plot
plot_results_ground_truth(gv0[:-1], phase[:-1], amplitude[:-1], ref_gray, ref_phase, ref_phase_std, ref_amplitude)

plt.figure()
amplitude_norm = amplitude / amplitude.mean()
E = amplitude_norm * np.exp(1.0j * phase)
ref_amplitude_norm = ref_amplitude / ref_amplitude.mean()
E_ref = ref_amplitude_norm * np.exp(1.0j * ref_phase)
plt.plot(E_ref.real, E_ref.imag, label="Reference")
plt.plot(E.real, E.imag, label="Predicted")
plt.legend()

plt.figure()
plt.plot(np.angle(E_ref.conj() * E[:-1].numpy()))
plt.title('Phase difference Ref-Pred')
plt.xlabel('Gray value')
plt.ylabel('$\\Delta\\phi$ (rad)')
plt.show()
