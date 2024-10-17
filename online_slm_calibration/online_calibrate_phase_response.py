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
from calibration_functions import import_lut, learn_lut
from directories import localdata


# === Settings === #
do_plot = True
do_end_plot = True
plot_per_its = 10
N = 2                           # Non-linearity factor. 1 = linear, 2 = 2PEF, 3 = 3PEF, etc., 0 = PMT is broken :)
iterations = 1000


filepath_lut = os.path.join(localdata, '2023_08_inline_slm_calibration/LUT Files/corrected_2022_08_26 10-47-28.blt')
filepath_measurements = os.path.join(localdata, 'harish_signal_feedback.mat')

with h5py.File(filepath_measurements, "r") as f:
    file_dict = get_dict_from_hdf5(f)

feedback_meas = torch.tensor(file_dict['feedback'])
gv0 = torch.tensor(file_dict['gv_row'])
gv1 = torch.tensor(file_dict['gv_col'])

lut_correct = import_lut(filepath_lut=filepath_lut)

learn_lut(gray_values0=gv0, gray_values1=gv1, feedback_measurements=feedback_meas, nonlinearity=N, iterations=1000,
          do_plot=do_plot, do_end_plot=do_end_plot, plot_per_its=plot_per_its)

