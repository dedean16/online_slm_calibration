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


def phase_correlation(phase1, phase2):
    return field_correlation(torch.exp(1j * phase1), torch.exp(1j * phase2))


def phase_response(input_phase, c, dim=-3):
    """

    Args:

    """
    pows = torch.arange(c.numel()).view(c.shape)
    actual_phase = 2*np.pi * (c * (input_phase/(2*np.pi)) ** pows).sum(dim=dim)
    return actual_phase


def predict_feedback(phase_in1, phase_in2, a, b, c, N, noise_level):
    """

    Args:

    """
    phase_diff_actual = phase_response(phase_in2, c) - phase_response(phase_in1, c)
    feedback_clean = a + b * (torch.cos(phase_diff_actual / 2)**(2*N))
    feedback = feedback_clean + noise_level * torch.randn(feedback_clean.shape)
    return feedback


def plot_phase_curve(c_pred, phase, phase_lut_correct):
    phase_in = phase.squeeze()
    phase_curve_pred = phase_response(phase_in, c_pred.detach()).squeeze()

    n = len(phase_lut_correct)
    phase_lut_out = np.arange(0, 2*np.pi, 2*np.pi/n)

    plt.plot(phase_lut_out, phase_lut_out, '--', color=(0.8, 0.8, 0.8), label='Linear')
    plt.plot(phase_lut_correct, phase_lut_out, label='Ground truth')
    plt.plot(phase_in, phase_curve_pred - 8, label='Prediction')
    plt.xlabel('phase in')
    plt.ylabel('phase actual')
    plt.title(f'iter {it}')
    plt.legend()
    plt.ylim((-2*np.pi, 4*np.pi))


def plot_feedback_fit(feedback_meas, feedback, phase1, phase2):
    plt.subplot(1, 3, 2)
    extent = (phase2.min(), phase2.max(), phase1.min(), phase1.max())
    vmin = feedback_meas.min()
    vmax = feedback_meas.max()
    plt.imshow(feedback_meas.squeeze().detach(), extent=extent, interpolation='nearest', vmin=vmin, vmax=vmax)
    plt.title('Measured feedback')
    plt.colorbar()

    plt.subplot(1, 3, 3)
    plt.imshow(feedback.squeeze().detach(), extent=extent, interpolation='nearest', vmin=vmin, vmax=vmax)
    plt.title('Predicted feedback')
    plt.colorbar()


# === Settings === #
do_plot = False
plot_per_its = 100
N = 2                           # Non-linearity factor. 1 = linear, 2 = 2PEF, 3 = 3PEF, etc., 0 = PMT is broken :)
num_poly_terms = 6              # Number of polynomial terms to fit

gv1_slice = slice(100, 200)
gv2_slice = slice(2, 6)


# Read inline signal feedback data
filepath = '/home/dani/LocalData/harish_signal_feedback.mat'

with h5py.File(filepath, "r") as f:
    file_dict = get_dict_from_hdf5(f)

gauss_sigma = (0, 3, 0)
feedback_meas_raw = torch.tensor(file_dict['feedback']).unsqueeze(0)


# from scipy.ndimage import gaussian_filter
# feedback_meas_med = gaussian_filter(feedback_meas_raw, sigma=gauss_sigma)
# phase_speed = torch.tensor(feedback_meas_med).diff(dim=1).abs().pow(2).mean(dim=2).sqrt()
#
# plt.plot(feedback_meas_raw.squeeze(), '--')
# plt.plot(feedback_meas_med.squeeze())
# plt.show()
# # feedback_meas_med = feedback_meas_raw.view(1, -1, median_samples, 9).median(dim=2).values
# # plt.figure()
# # plt.plot(feedback_meas_raw.squeeze())
# # plt.figure()
# # plt.plot(feedback_meas_med.squeeze())
# # plt.show()
# plt.plot(phase_speed.squeeze())
# plt.show()

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
