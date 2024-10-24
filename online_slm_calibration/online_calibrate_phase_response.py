# External 3rd party
import torch
import matplotlib.pyplot as plt
import h5py

# External ours
from openwfs.algorithms.troubleshoot import field_correlation

# Internal
from helper_functions import get_dict_from_hdf5
from calibration_functions import import_lut, learn_field
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

bleach = torch.mean(feedback_meas, 0)
bleach = torch.linspace(bleach[0], bleach[-1], len(bleach))
feedback_meas = feedback_meas - bleach.reshape(1, -1)
#feedback_meas[90,:] = 0.5 * (feedback_meas[89,:]+feedback_meas[91,:])
#feedback_meas[82,:] = 0.5 * (feedback_meas[81,:]+feedback_meas[83,:])
# extent = (gv1.min()-0.5, gv1.max()+0.5, gv0.min()-0.5, gv0.max()+0.5)
# plt.imshow(feedback_meas, extent=extent)
# plt.show()
#
# ff = feedback_meas[torch.flatten(gv1), :]
# plt.figure()
# plt.imshow(ff)
# plt.show()


with h5py.File(filepath_ref) as f:
    ref_dict = get_dict_from_hdf5(f)

# Learn phase response
lr, phase_response_per_gv_fit, amplitude = learn_field(
    gray_values0=gv0, gray_values1=gv1, measurements=feedback_meas, nonlinearity=N, learning_rate=0.3, iterations=1800,
    do_plot=do_plot, do_end_plot=do_end_plot, plot_per_its=30, smooth_loss_factor=0)

print(f'b = {amplitude.mean()}, lr = {lr} (1.0)')

plt.figure()
plt.subplot(2, 1, 1)
plt.errorbar(ref_dict['gray_values'][0], ref_dict['phase_mean'][0], yerr=ref_dict['phase_std'][0],
             linestyle='--', color='#333333', label='Reference')
plt.plot(phase_response_per_gv_fit, color='C0', label='Predicted')
plt.xlabel('Gray value')
plt.ylabel('Phase response')
plt.legend()
plt.subplot(2, 1, 2)
plt.plot(amplitude, color='C0', label='Amplitude')
plt.show()

