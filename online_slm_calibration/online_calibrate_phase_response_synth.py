# External 3rd party
import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# External ours
from openwfs.algorithms.troubleshoot import field_correlation

# Internal
from helper_functions import get_dict_from_hdf5
from calibration_functions import learn_lut, predict_feedback
from directories import localdata


# === Settings === #
do_plot = True
do_end_plot = True
plot_per_its = 40
N = 2                           # Non-linearity factor. 1 = linear, 2 = 2PEF, 3 = 3PEF, etc., 0 = PMT is broken :)
iterations = 2001

noise_level = 1.0


phase_response_per_gv_gt = 2.5 * np.pi * torch.linspace(0.0, 1.0, 256) ** 2
a_gt = torch.tensor(5.0)
b_gt = torch.tensor(20.0)

gv0 = torch.arange(0, 256, dtype=torch.int32)
gv1 = torch.arange(0, 256, 32, dtype=torch.int32)


feedback_meas = predict_feedback(gv0, gv1, a_gt, b_gt, phase_response_per_gv_gt, nonlinearity=N, noise_level=noise_level)

learn_lut(gray_values0=gv0, gray_values1=gv1, feedback_measurements=feedback_meas, nonlinearity=N, learning_rate=0.04,
          iterations=iterations, do_plot=do_plot, do_end_plot=do_end_plot, plot_per_its=plot_per_its)
