import numpy as np
import matplotlib.pyplot as plt
import h5py

from helper_functions import get_dict_from_hdf5
from calibration_functions import learn_field, detrend, arcsin_phase_retrieve
from directories import data_folder
from online_slm_calibration.plot_utilities import plot_results_ground_truth

# === Settings === #
settings = {
    "do_plot": True,
    "plot_per_its": 1500,
    "nonlinearity": 2.0,
    "learning_rate": 0.1,
    "iterations": 3000,
    "smooth_loss_factor": 1.0,
}

# Import feedback measurements and reference phase response
filepath_ref = data_folder.joinpath("slm_reference_phase_response.mat")  # Reference phase response
filepath_measurements = data_folder.joinpath("slm_calibration_signal_feedback.mat")
with h5py.File(filepath_measurements, "r") as f:
    feedback_dict = get_dict_from_hdf5(f)
    measurements = feedback_dict["feedback"]
    gv0 = feedback_dict["gv_row"].astype(int).ravel() % 256
    gv1 = feedback_dict["gv_col"].astype(int).ravel() % 256

with h5py.File(filepath_ref) as f:
    ref_dict = get_dict_from_hdf5(f)
    ref_gray = ref_dict["gray_values"][0]
    ref_phase = ref_dict["phase_mean"][0]
    ref_phase_err = ref_dict["phase_std"][0]
    ref_amplitude = ref_dict["modulation_depth"][0]

# Compensate for photo-bleaching
#measurements = measurements[:, 3:]
#gv1 = gv1[3:]
measurements = detrend(gv0, gv1, measurements)

plt.figure()
plt.plot(measurements[:, 0])
plt.title('Measurements 0')
plt.show()


# Learn phase response
nl, lr, phase, amplitude = learn_field(
    gray_values0=gv0, gray_values1=gv1, measurements=measurements, **settings
)

print(f"lr = {lr} (1.0), nl = {nl} ({settings['nonlinearity']})")

plot_results_ground_truth(phase, amplitude, ref_phase)

phase_arcsin = arcsin_phase_retrieve(measurements[:, 0], nl, [-1.348, -1.185], [0, 133, 164, 182, 201])

plt.figure()
plt.plot(phase, label='Fit')
plt.plot(ref_phase, label='Ref')
plt.plot(phase_arcsin, label='arcsin')
plt.legend()
plt.show()

plt.figure()
amplitude_norm = amplitude / amplitude.mean()
E = amplitude_norm * np.exp(1.0j * phase)
ref_amplitude_norm = ref_amplitude / ref_amplitude.mean()
E_ref = ref_amplitude_norm * np.exp(1.0j * ref_phase)
plt.plot(E_ref.real, E_ref.imag, label="Reference")
plt.plot(E.real, E.imag, label="Predicted")
plt.legend()

plt.figure()
plt.plot(np.angle(E_ref.conj() * E))
plt.title('Phase difference Ref-Pred')
plt.xlabel('Gray value')
plt.ylabel('$\\Delta\\phi$ (rad)')
plt.show()
