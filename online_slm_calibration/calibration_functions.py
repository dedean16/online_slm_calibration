# External 3rd party
from torch import Tensor as tt
import numpy as np
import torch
import matplotlib.pyplot as plt

# External ours
from openwfs.algorithms.troubleshoot import field_correlation


def phase_correlation(phase1, phase2):
    """
    Compute the field correlation between two phase curves.
    """
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


def import_lut(filepath_lut, scaling=8.0) -> tt:
    """
    Import blt lookup table from .blt file; a text file containing 256 gray values, corresponding to the range [0, 2π).

    Args:
        filepath_lut: Filepath to the .blt file.
        scaling: Scaling factor w.r.t. the range [0, 255] i.e. a bit depth of 8-bit, used for the .blt file. The range
        of the gray values is by default [0, 2047], which corresponds to a scaling of 8.0.

    Returns: the lookup table as 256-element tensor.
    """
    with open(filepath_lut, 'r') as file:
        numbers = [float(line.strip()) for line in file.readlines()]
    return torch.tensor(numbers) / scaling


