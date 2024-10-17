# Built-in
import os

# External 3rd party
import numpy as np
import torch
from tqdm import tqdm
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

feedback_meas = torch.tensor(file_dict['feedback']).unsqueeze(0)
gv1 = torch.tensor(file_dict['gv_row']).view(1, -1, 1)
gv2 = torch.tensor(file_dict['gv_col']).view(1, 1, -1)

lut_correct = import_lut(filepath_lut=filepath_lut)

# Create init prediction


# Initialize parameters and optimizer
learning_rate = 3e-2
params = [{'lr': learning_rate, 'params': [a, b, c]}]
optimizer = torch.optim.Adam(params, lr=learning_rate, amsgrad=True)


iterations = 30000
progress_bar = tqdm(total=iterations)

if do_plot:
    plt.figure(figsize=(13, 4))

for it in range(iterations):
    feedback = predict_feedback(phase1, phase2, a, b, c, N, noise_level)

    error = (feedback_meas - feedback).abs().pow(2).sum()

    # Gradient descent step
    error.backward()
    optimizer.step()
    optimizer.zero_grad()

    if do_plot and it % plot_per_its == 0:
        plt.clf()
        plt.subplot(1, 3, 1)
        plot_phase_curve(c, phase1, phase_lut_correct)
        plot_feedback_fit(feedback_meas, feedback, phase1, phase2)
        plt.pause(0.01)

    progress_bar.update()

if do_plot:
    plt.clf()
else:
    plt.figure(figsize=(13, 4))
plt.subplot(1, 3, 1)
plot_phase_curve(c, phase1, phase_lut_correct)
plot_feedback_fit(feedback_meas, feedback, phase1, phase2)
plt.show()

print(f'a: {a}')
print(f'b: {b}')
print(f'c: {c}')
