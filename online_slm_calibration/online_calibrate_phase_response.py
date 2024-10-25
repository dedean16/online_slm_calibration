# Built-in
import os

# External 3rd party
import torch
import matplotlib.pyplot as plt
import h5py
import numpy as np
from scipy.ndimage import median_filter

# External ours
from openwfs.algorithms.troubleshoot import field_correlation

# Internal
from helper_functions import get_dict_from_hdf5
from calibration_functions import import_lut, grow_learn_lut
from directories import data_folder


# === Settings === #
do_plot = False
do_end_plot = True
plot_per_its = 1500
N = 2                           # Non-linearity factor. 1 = linear, 2 = 2PEF, 3 = 3PEF, etc., 0 = PMT is broken :)


filepath_ref = data_folder.joinpath('slm_reference_phase_response.mat')     # Reference phase response
filepath_measurements = data_folder.joinpath('slm_calibration_signal_feedback.mat')

# Import feedback measurements
with h5py.File(filepath_measurements, "r") as f:
    file_dict = get_dict_from_hdf5(f)

feedback_meas_raw = torch.tensor(file_dict['feedback'])
feedback_meas_med = torch.tensor(median_filter(feedback_meas_raw, size=(1, 1)))
gv0 = torch.tensor(file_dict['gv_row'] % 256, dtype=torch.int32)
gv1 = torch.tensor(file_dict['gv_col'][0:8, :] % 256, dtype=torch.int32)

# Import reference phase response measurements
with h5py.File(filepath_ref) as f:
    ref_dict = get_dict_from_hdf5(f)

phase_response_per_gv = grow_learn_lut(
    gray_values0=gv0, gray_values1=gv1, feedback_measurements=feedback_meas_med, nonlinearity=N, learning_rate=0.02,
    iterations=1001, do_plot=do_plot, do_end_plot=do_end_plot, plot_per_its=plot_per_its, smooth_factor=10.0,
    gray_value_slice_size=8)

# Plot
plt.subplot(1, 3, 1)
plt.cla()
linear_phase = np.linspace(0.0, 2*np.pi, 256)
plt.errorbar(ref_dict['gray_values'][0], ref_dict['phase_mean'][0], yerr=ref_dict['phase_std'][0],
             linestyle='--', color='#333333', label='Reference')
plt.plot(phase_response_per_gv - phase_response_per_gv[0], color='C0', label='Predicted')
plt.xlabel('Gray value')
plt.ylabel('Phase response (rad)')
plt.legend()
plt.show()
