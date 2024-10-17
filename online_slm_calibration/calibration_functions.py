# External 3rd party
import torch
from torch import Tensor as tt
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

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


def predict_feedback(gray_value0, gray_value1, a: tt, b: tt, phase_response_per_gv: tt, nonlinearity,
                     noise_level=0.0) -> tt:
    """

    Args:
        gray_value0: Tensor containing gray values of group 0, corresponding to dim 0 of feedback.
        gray_value1: Tensor containing gray values of group 1, corresponding to dim 1 of feedback.
        a: Predicted feedback offset.
        b: Predicted feedback interference signal factor.
        phase_response_per_gv: Phase response per gray value. The corresponding gray values coincide with
            the array index.
        nonlinearity: Nonlinearity number. 1 = linear, 2 = 2PEF, 3 = 3PEF, etc., 0 = detector is broken :)
        noise_level: Optional noise level.

    Returns:
        Predicted
    """
    # Set array dimensions
    gv0 = gray_value0.view(-1, 1)
    gv1 = gray_value1.view(1, -1)

    # Compute phase difference and predicted clean feedback
    phase_diff_actual = phase_response(gv1, phase_response_per_gv) - phase_response(gv0, phase_response_per_gv)
    feedback_clean = a + b * (torch.cos(phase_diff_actual / 2) ** (2 * nonlinearity))

    # Add optional noise if requested
    if noise_level == 0.0:
        return feedback_clean
    else:
        return feedback_clean + noise_level * torch.randn(feedback_clean.shape)


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


def plot_feedback_fit(feedback_meas, feedback, gray_values0, gray_values1):
    plt.subplot(1, 3, 2)
    extent = (gray_values1.min(), gray_values1.max(), gray_values0.min(), gray_values0.max())
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


def learn_lut(gray_values0: tt, gray_values1: tt, feedback_measurements: tt, nonlinearity=2, iterations: int = 1000,
              init_noise_level=0.1, do_plot: bool = False, plot_per_its: int = 10, do_end_plot: bool = True) -> tt:
    """
    Learn the phase lookup table from dual phase stepping measurements.

    Args:
        gray_values0:
        gray_values1: Same as gray_values1, for the second group.
        feedback_measurements:
        nonlinearity: Nonlinearity number. 1 = linear, 2 = 2PEF, 3 = 3PEF, etc., 0 = detector is broken :)
        iterations: Number of learning iterations.
        do_plot: If True, plot during learning.
        plot_per_its: Plot per this many learning iterations.
        init_noise_level: Standard deviation of the Gaussian noise added to the initial phase_response guess.
    Returns:
    """
    # Create init prediction
    phase_response_per_gv = torch.linspace(0, 2*np.pi, 256) + torch.randn(256) * init_noise_level
    phase_response_per_gv.requires_grad = True
    a = torch.tensor(feedback_measurements.mean(), requires_grad=True)
    b = torch.tensor(3*feedback_measurements.std(), requires_grad=True)


    # Initialize parameters and optimizer
    learning_rate = 5e-2
    params = [{'lr': learning_rate, 'params': [phase_response_per_gv]}]
    optimizer = torch.optim.Adam(params, lr=learning_rate, amsgrad=True)

    progress_bar = tqdm(total=iterations)

    if do_plot:
        plt.figure(figsize=(13, 4))

    for it in range(iterations):
        feedback_predicted = predict_feedback(gray_values0, gray_values1, phase_response_per_gv, nonlinearity)
        error = (feedback_measurements - feedback_predicted).abs().pow(2).sum()

        # Gradient descent step
        error.backward()
        optimizer.step()
        optimizer.zero_grad()

        if do_plot and it % plot_per_its == 0:
            plt.clf()
            plt.subplot(1, 3, 1)
            plot_feedback_fit(feedback_measurements, feedback_predicted, gray_values0, gray_values1)
            plt.pause(0.01)

        progress_bar.update()

    if do_end_plot:
        if do_plot:
            plt.clf()
        else:
            plt.figure(figsize=(13, 4))
        plt.subplot(1, 3, 1)

        # TODO: plot LUT vs correct LUT
        plot_feedback_fit(feedback_measurements, feedback_predicted, gray_values0, gray_values1)
        plt.show()

    return phase_response_per_gv
