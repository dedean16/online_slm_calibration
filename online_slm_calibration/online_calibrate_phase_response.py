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
from calibration_functions import import_lut
from directories import localdata


# === Settings === #
do_plot = False
plot_per_its = 100
N = 2                           # Non-linearity factor. 1 = linear, 2 = 2PEF, 3 = 3PEF, etc., 0 = PMT is broken :)
num_poly_terms = 6              # Number of polynomial terms to fit


filepath_lut = os.path.join(localdata, '2023_08_inline_slm_calibration/LUT Files/corrected_2022_08_26 10-47-28.blt')
filepath_measurements = os.path.join(localdata, 'harish_signal_feedback.mat')

with h5py.File(filepath_measurements, "r") as f:
    file_dict = get_dict_from_hdf5(f)

feedback_meas = torch.tensor(file_dict['feedback'])
gv0 = torch.tensor(file_dict['gv_row'])
gv1 = torch.tensor(file_dict['gv_col'])

lut_correct = import_lut(filepath_lut=filepath_lut)

