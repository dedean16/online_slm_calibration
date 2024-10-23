# Built-in
import os

# External 3rd party
import torch
import matplotlib.pyplot as plt
import h5py

# External ours
from openwfs.algorithms.troubleshoot import field_correlation

# Internal
from helper_functions import get_dict_from_hdf5
from calibration_functions import import_lut, learn_field
from directories import localdata


# === Settings === #
do_plot = True
do_end_plot = True
plot_per_its = 20
N = 2                           # Non-linearity factor. 1 = linear, 2 = 2PEF, 3 = 3PEF, etc., 0 = PMT is broken :)
iterations = 1000


filepath_lut = os.path.join(localdata, '2023_08_inline_slm_calibration/LUT Files/corrected_2022_08_26 10-47-28.blt')
filepath_measurements = os.path.join(localdata, 'harish_signal_feedback.mat')

with h5py.File(filepath_measurements, "r") as f:
    file_dict = get_dict_from_hdf5(f)

feedback_meas = torch.tensor(file_dict['feedback'])
gv0 = torch.tensor(file_dict['gv_row'] % 256, dtype=torch.int32)
gv1 = torch.tensor(file_dict['gv_col'] % 256, dtype=torch.int32)

lut_correct = import_lut(filepath_lut=filepath_lut)

lr, phase_response_per_gv_fit, amplitude = learn_field(
    gray_values0=gv0, gray_values1=gv1,measurements=feedback_meas, nonlinearity=N, learning_rate=0.05, iterations=1000,
    do_plot=do_plot, do_end_plot=do_end_plot, plot_per_its=30)

print(f'b = {amplitude.mean()}, lr = {lr} (1.0)')

plt.figure()
plt.subplot(2, 1, 1)
# plt.plot(phase_response_per_gv_gt, color='C0', label='Ground truth')
plt.plot(phase_response_per_gv_fit, '--', color='C1', label='Predicted')
plt.xlabel('Gray value')
plt.ylabel('Phase response')
plt.legend()
plt.subplot(2, 1, 2)
plt.plot(amplitude, color='C0', label='Amplitude')
plt.show()
