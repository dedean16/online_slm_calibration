# External 3rd party
import torch
import matplotlib.pyplot as plt
import h5py

# External ours
from openwfs.algorithms.troubleshoot import field_correlation

# Internal
from helper_functions import get_dict_from_hdf5
from calibration_functions import grow_learn_field
from directories import data_folder


# === Settings === #
do_plot = True
do_end_plot = True
N = 2                           # Non-linearity factor. 1 = linear, 2 = 2PEF, 3 = 3PEF, etc., 0 = PMT is broken :)

filepath_ref = data_folder.joinpath('slm_reference_phase_response.mat')     # Reference phase response
filepath_measurements = data_folder.joinpath('slm_calibration_signal_feedback.mat')

# Import feedback measurements
with h5py.File(filepath_measurements, "r") as f:
    feedback_dict = get_dict_from_hdf5(f)

feedback_meas = torch.tensor(feedback_dict['feedback'])
gv0 = torch.tensor(feedback_dict['gv_row'] % 256, dtype=torch.int32)
gv1 = torch.tensor(feedback_dict['gv_col'] % 256, dtype=torch.int32)

# Import reference phase response measurements
with h5py.File(filepath_ref) as f:
    ref_dict = get_dict_from_hdf5(f)

# Learn phase response
B, phase, amplitude = grow_learn_field(
    gray_values0=gv0, gray_values1=gv1, measurements=feedback_meas, nonlinearity=N, learning_rate=0.05, iterations=100,
    do_plot=do_plot, do_end_plot=do_end_plot, plot_per_its=30)

print(f'b = {amplitude.mean()}, B = {B} (1.0)')

plt.figure()
plt.subplot(2, 1, 1)
plt.errorbar(ref_dict['gray_values'][0], ref_dict['phase_mean'][0], yerr=ref_dict['phase_std'][0],
             linestyle='--', color='#333333', label='Reference')
plt.plot(phase, color='C0', label='Predicted')
plt.xlabel('Gray value')
plt.ylabel('Phase response')
plt.legend()
plt.subplot(2, 1, 2)
plt.plot(amplitude, color='C0', label='Amplitude')
plt.show()
