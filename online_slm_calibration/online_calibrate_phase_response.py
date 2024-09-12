import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

from openwfs.algorithms.troubleshoot import field_correlation


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


def plot_phase_curve(c_gt, c):
    phase_in = torch.linspace(0, 2 * np.pi, 100)
    phase_curve_gt = phase_response(phase_in, c_gt).squeeze()
    phase_curve_pred = phase_response(phase_in, c.detach()).squeeze()

    phase_corr = phase_correlation(phase_curve_gt, phase_curve_pred)

    plt.plot(phase_in, phase_in, '--', color=(0.8, 0.8, 0.8), label='Linear')
    plt.plot(phase_in, phase_curve_gt, label='Ground truth')
    plt.plot(phase_in, phase_curve_pred, label='Prediction')
    plt.xlabel('phase in')
    plt.ylabel('phase actual')
    plt.title(f'iter {it}, |phase correlation|={np.abs(phase_corr):.4f}')
    plt.legend()


do_plot = False

# Define 'measured' phases
P1 = 24
P2 = 12
phase1 = 2*np.pi/P1 * torch.arange(P1).view(1, 1, -1)
phase2 = 2*np.pi/P2 * torch.arange(P2).view(1, -1, 1)
N = 2

# Create synthetic measurement
num_terms = 5
a_gt = 0.2
b_gt = 0.7
c_gt = torch.randn(num_terms, 1, 1) * 0.2
c_gt[1, 0, 0] += 1
noise_level_gt = 0
feedback_gt = predict_feedback(phase1, phase2, a_gt, b_gt, c_gt, N, noise_level_gt)

# Create init prediction
num_terms = 5
a = torch.tensor(0.0, requires_grad=True)
b = torch.tensor(1.0, requires_grad=True)
c = torch.zeros(num_terms, 1, 1)
c[1, 0, 0] += 1
c.requires_grad = True
noise_level = 0.15


# Initialize parameters and optimizer
learning_rate = 2e-3
params = [{'lr': learning_rate, 'params': [a, b, c]}]
optimizer = torch.optim.Adam(params, lr=learning_rate, amsgrad=True)


iterations = 1000
progress_bar = tqdm(total=iterations)

for it in range(iterations):
    feedback = predict_feedback(phase1, phase2, a, b, c, N, noise_level)

    error = (feedback_gt - feedback).abs().pow(2).sum()

    # Gradient descent step
    error.backward()
    optimizer.step()
    optimizer.zero_grad()

    if do_plot:
        plt.cla()
        plot_phase_curve(c_gt, c)
        plt.pause(0.01)

    progress_bar.update()


plt.cla()
plot_phase_curve(c_gt, c)
plt.show()

print(f'a_gt: {a_gt}')
print(f'a: {a}')
print(f'b_gt: {b_gt}')
print(f'b: {b}')
print(f'c_gt: {c_gt}')
print(f'c: {c}')
