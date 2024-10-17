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


# === Settings === #
do_plot = False
plot_per_its = 100
N = 2                           # Non-linearity factor. 1 = linear, 2 = 2PEF, 3 = 3PEF, etc., 0 = PMT is broken :)
num_poly_terms = 6              # Number of polynomial terms to fit

gv1_slice = slice(0, 255)
gv2_slice = slice(0, 9)


# Read inline signal feedback data
filepath = '/home/dani/LocalData/harish_signal_feedback.mat'

with h5py.File(filepath, "r") as f:
    file_dict = get_dict_from_hdf5(f)

gauss_sigma = (0, 3, 0)
feedback_meas_raw = torch.tensor(file_dict['feedback']).unsqueeze(0)

gv1_raw = torch.tensor(file_dict['gv_row']).view(1, -1, 1)
gv2_raw = torch.tensor(file_dict['gv_col']).view(1, 1, -1)
gv1 = gv1_raw[:, gv1_slice, :]
gv2 = gv2_raw[:, :, gv2_slice]
phase1 = 2*np.pi/256 * gv1
phase2 = 2*np.pi/256 * gv2
feedback_meas = feedback_meas_raw[:, gv1_slice, gv2_slice]

# Create init prediction
a = torch.tensor(feedback_meas.mean(), requires_grad=True)
b = torch.tensor(feedback_meas.std(), requires_grad=True)
c = torch.zeros(num_poly_terms, 1, 1)
c[1, 0, 0] = 2.3
c.requires_grad = True
noise_level = 0.0


# Import correct LUT
def read_file_to_tensor(file_path):
    with open(file_path, 'r') as file:
        numbers = [float(line.strip()) for line in file.readlines()]
    return torch.tensor(numbers)


filepath_lut = '/home/dani/LocalData/2023_08_inline_slm_calibration/LUT Files/corrected_2022_08_26 10-47-28.blt'
lut_correct = read_file_to_tensor(filepath_lut) / 8
n = len(lut_correct)
phase_lut_correct = 2*np.pi / n * lut_correct


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
