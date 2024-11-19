import glob

import numpy as np
import matplotlib.pyplot as plt
import h5py

from helper_functions import get_dict_from_hdf5
from calibration_functions import learn_field, detrend
from directories import data_folder, data_folder
from plot_utilities import plot_results_ground_truth, plot_result_feedback_fit


plt.rcParams.update({'font.size': 14})


# === Settings === #
# Import feedback measurements and reference phase response
ref_glob = str(data_folder.joinpath("tg_fringe")) + '/tg-fringe-slm-calibration-r*.npz'  # Reference phase response
filepath_measurements = data_folder.joinpath("slm_calibration_signal_feedback.mat")

settings = {
    "do_plot": True,
    "do_end_plot": True,
    "plot_per_its": 300,
    "nonlinearity": 2.0,
    "learning_rate": 0.3,
    "iterations": 2000,
    "smooth_loss_factor": 1.0,
    "balance_factor": 1.0,
}

with h5py.File(filepath_measurements, "r") as f:
    feedback_dict = get_dict_from_hdf5(f)
    measurements = feedback_dict["feedback"]
    # # Uncomment to hide outliers
    # measurements[165, :] = (measurements[166, :] + measurements[164, :]) / 2
    # measurements[173, :] = (measurements[174, :] + measurements[172, :]) / 2
    gv0 = feedback_dict["gv_row"].astype(int).ravel()
    gv1 = feedback_dict["gv_col"].astype(int).ravel()
    gv0 = np.asarray((
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
        23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43,
        44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
        64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83,
        84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 100, 101, 102, 103,
        104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119,
        120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135,
        136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151,
        152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 165, 166, 167, 168,
        169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184,
        185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200,
        201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216,
        217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 231, 232, 233,
        234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249,
        250, 251, 252, 253, 254, 255, 255, 1, 2, 3
    ))
    gv1 = np.asarray((0, 32, 65, 97, 130, 162, 195, 227, 0))


ref_files = glob.glob(ref_glob)
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
ref_phase_std = np.std(ref_phase_all, axis=0)

# plt.plot(np.abs(ref_field_all).T)
plt.plot(np.asarray(ref_phase_all).T)
plt.xlabel('Gray value')
plt.ylabel('Phase')
plt.title('TG fringe calibration, unwrapped\ncompensated for sign and offset')
plt.pause(0.01)

# Compensate for photo-bleaching
#measurements = measurements[:, 3:]
#gv1 = gv1[3:]
measurements = detrend(gv0, gv1, measurements)


# Learn phase response
nl, lr, phase, amplitude = learn_field(
    gray_values0=gv0, gray_values1=gv1, measurements=measurements, **settings
)

print(f"lr = {lr:.4f} (1.0), nl = {nl:.4f} ({settings['nonlinearity']})")

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
plt.plot(np.angle(E_ref.numpy().conj() * E))
plt.title('Phase difference Ref-Pred')
plt.xlabel('Gray value')
plt.ylabel('$\\Delta\\phi$ (rad)')
plt.show()
