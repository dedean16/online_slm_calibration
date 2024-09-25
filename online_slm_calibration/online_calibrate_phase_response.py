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


def plot_phase_curve(c_gt, c_pred):
    phase_in = torch.linspace(0, 2 * np.pi, 100)
    phase_curve_gt = phase_response(phase_in, c_gt).squeeze()
    phase_curve_pred = phase_response(phase_in, c_pred.detach()).squeeze()

    phase_corr = phase_correlation(phase_curve_gt, phase_curve_pred)

    plt.plot(phase_in, phase_in, '--', color=(0.8, 0.8, 0.8), label='Linear')
    plt.plot(phase_in, phase_curve_gt, label='Ground truth')
    plt.plot(phase_in, phase_curve_pred, label='Prediction')
    plt.xlabel('phase in')
    plt.ylabel('phase actual')
    plt.title(f'iter {it}, |phase correlation|={np.abs(phase_corr):.4f}')
    plt.legend()


def plot_feedback_fit(feedback_meas, feedback, phase1, phase2):
    plt.subplot(1, 3, 2)
    extent = (phase2.min(), phase2.max(), phase1.min(), phase1.max())
    plt.imshow(feedback_meas.squeeze().detach(), extent=extent, interpolation='nearest')
    plt.title('Synth feedback')

    plt.subplot(1, 3, 3)
    plt.imshow(feedback.squeeze().detach(), extent=extent, interpolation='nearest')
    plt.title('Predicted feedback')


do_plot = True
plot_per_its = 10

gv_slice = slice(120, 220)


def read_file_to_tensor(file_path):
    with open(file_path, 'r') as file:
        numbers = [float(line.strip()) for line in file.readlines()]
    return torch.tensor(numbers)

filepath_lut = '/home/dani/LocalData/2023_08_inline_slm_calibration/LUT Files/corrected_2022_08_26 10-47-28.blt'
lut_correct = read_file_to_tensor(filepath_lut) / 8
plt.plot(lut_correct)
plt.show()


# Read inline data
filepath = '/home/dani/LocalData/harish_signal_feedback.mat'

with h5py.File(filepath, "r") as f:
    file_dict = get_dict_from_hdf5(f)

feedback_meas = torch.tensor(file_dict['feedback']).unsqueeze(0)
P1 = feedback_meas.shape[1]
P2 = feedback_meas.shape[2]
phase1 = 2*np.pi/256 * torch.tensor(file_dict['gv_row']).view(1, -1, 1)
phase2 = 2*np.pi/256 * torch.tensor(file_dict['gv_col']).view(1, 1, -1)
N = 2

# Create init prediction
num_terms = 6
a = torch.tensor(0.0, requires_grad=True)
b = torch.tensor(1.0, requires_grad=True)
c = torch.zeros(num_terms, 1, 1)
c[1, 0, 0] = 2.0
c.requires_grad = True
noise_level = 0.0


# Initialize parameters and optimizer
learning_rate = 1e-3
params = [{'lr': learning_rate, 'params': [a, b, c]}]
optimizer = torch.optim.Adam(params, lr=learning_rate, amsgrad=True)


iterations = 2000
progress_bar = tqdm(total=iterations)

if do_plot:
    plt.figure(figsize=(12, 4))

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
        plot_phase_curve(c.detach(), c)
        plot_feedback_fit(feedback_meas, feedback, phase1, phase2)
        plt.pause(0.01)

    progress_bar.update()


plt.clf()
plt.subplot(1, 3, 1)
plot_phase_curve(c.detach(), c)
plot_feedback_fit(feedback_meas, feedback, phase1, phase2)
plt.show()

print(f'a: {a}')
print(f'b: {b}')
print(f'c: {c}')
