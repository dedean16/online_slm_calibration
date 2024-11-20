# External 3rd party
from typing import Tuple

import torch
from torch import Tensor as tt, Tensor
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

# External ours
from openwfs.algorithms.troubleshoot import field_correlation

# Internal
from plot_utilities import plot_field_response, plot_feedback_fit, plot_result_feedback_fit


def phase_correlation(phase1, phase2):
    """
    Compute the field correlation between two phase curves.
    """
    return field_correlation(torch.exp(1j * phase1), torch.exp(1j * phase2))

def detrend(gray_value0, gray_value1, measurements: np.ndarray, do_plot=False):
    m = torch.tensor(measurements.flatten(order="F"))
    m = m / m.abs().mean()

    # locate elements for which gv0 == gv1. These are measured twice and should be equal except for noise and photobleaching.
    gv0 = np.asarray(gray_value0)
    sym_selection = [np.nonzero(gv0 == gv1)[0][0].item() for gv1 in gray_value1]

    learning_rate = 0.05

    # Initial values
    offset = torch.tensor(0.1 * (m.max() - m.min()), dtype=torch.float32, requires_grad=True)
    decay = torch.tensor(0.1 / len(m), dtype=torch.float32, requires_grad=True)
    t = np.cumsum(m)

    def photobleaching_fit():
        return offset * torch.exp(-decay * t)

    def take_diag(M):
        return M.reshape((measurements.shape[1], measurements.shape[0]))[:, sym_selection].diagonal()

    params = [
        {"params": [offset], "lr": learning_rate},
        {"params": [decay], "lr": 10*learning_rate / len(m)}
    ]
    optimizer = torch.optim.Adam(params, lr=learning_rate, amsgrad=True)

    plt.figure(figsize=(15, 5))

    for it in range(300):
        m_fit = photobleaching_fit()
        m_compensated = m / m_fit
        loss = (take_diag(m) - take_diag(m_fit)).pow(2).mean()

        measurements_compensated = m_compensated.detach().numpy().reshape(measurements.shape, order='F')

        if it % 10 == 0:
            plt.clf()
            plt.subplot(1, 3, 1)
            plt.imshow(measurements_compensated, aspect='auto', interpolation='nearest')
            plt.title(f'Offset={offset.detach():.3f}, decay={decay.detach():.3g}')

            plt.subplot(1, 3, 2)
            plt.plot(take_diag(m).detach())
            plt.plot(take_diag(m_fit).detach())
            plt.ylim((0, 10))
            plt.title(f'Fit diagonal entries (gv0==gv1)')

            plt.subplot(1, 3, 3)
            plt.plot(take_diag(m_compensated).detach())
            plt.ylim((0, 10))
            plt.title(f'Compensated diagonal entries (gv0==gv1), {it}')
            plt.pause(0.01)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    ff = measurements_compensated[sym_selection, :]

    if do_plot:
        plt.figure()
        plt.imshow(ff)
        plt.title('Selected measurements\nwith gray values swapped')

        plt.figure()
        plt.plot(m)
        plt.show()

    return measurements_compensated


def predict_feedback(
    gray_value0, gray_value1, a: tt, b: tt, phase_response_per_gv: tt, nonlinearity, noise_level=0.0
) -> tt:
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
    # Get phases
    phase0 = phase_response_per_gv[gray_value0].view(-1, 1)
    phase1 = phase_response_per_gv[gray_value1].view(1, -1)

    # Compute phase difference and predicted clean feedback
    phase_diff = phase1 - phase0
    feedback_clean = a + b * (torch.cos(phase_diff / 2) ** (2 * nonlinearity))

    # Add optional noise if requested
    if noise_level == 0.0:
        return feedback_clean
    else:
        return feedback_clean + noise_level * torch.randn(feedback_clean.shape)


def import_lut(filepath_lut, scaling=8.0) -> tt:
    """
    Import blt lookup table from .blt file; a text file containing 256 gray values, corresponding to the range [0, 2π).

    Args:
        filepath_lut: Filepath to the .blt file.
        scaling: Scaling factor w.r.t. the range [0, 255] i.e. a bit depth of 8-bit, used for the .blt file. The range
        of the gray values is by default [0, 2047], which corresponds to a scaling of 8.0.

    Returns: the lookup table as 256-element tensor.
    """
    with open(filepath_lut, "r") as file:
        numbers = [float(line.strip()) for line in file.readlines()]
    return torch.tensor(numbers) / scaling


# Create a Gaussian kernel function
def gaussian_kernel1d(kernel_size: int, sigma: float):
    # Create a tensor of equally spaced values (kernel_size elements centered around 0)
    x = torch.arange(-(kernel_size // 2), kernel_size // 2 + 1, dtype=torch.float32)
    gaussian = torch.exp(-0.5 * (x / sigma) ** 2)
    return gaussian / gaussian.sum()


def window_cosine_edge(size, edge_width):
    """
    Create a window with cosine edges.

    Args:
        size (int): Total size of the window.
        edge_width (int): Width of the cosine-tapered edges.

    Returns:
        np.ndarray: The window array.
    """
    if edge_width * 2 > size:
        raise ValueError("edge_width must be less than or equal to half of the size")

    window = np.ones(size)
    edge = np.linspace(0, np.pi / 2, edge_width)
    taper = np.sin(edge) ** 2

    window[:edge_width] = taper
    window[-edge_width:] = taper[::-1]

    return window


def learn_field(
    *,
    gray_values0,
    gray_values1,
    measurements,
    nonlinearity=1,
    iterations: int = 50,
    do_plot: bool = False,
    do_end_plot: bool = False,
    plot_per_its: int = 10,
    learning_rate=0.1,
    phase_stroke_init=2.5 * torch.pi,
    balance_factor=1.0,
    smooth_loss_factor=1.0,
) -> tuple[float, float, np.array, np.ndarray]:
    """
    Learn the phase lookup table from dual phase stepping measurements.
    This function uses the model:

        I[i,j] = |a * E[i] + b * E[j]|^(non_linearity)

    It learns lr_ratio, a and all complex numbers E[i] for each gray value i.

    Args:
        gray_values0:
        gray_values1: Same as gray_values1, for the second group.
        measurements:
        nonlinearity: Nonlinearity number. 1 = linear, 2 = 2PEF, 3 = 3PEF, etc., 0 = detector is broken :)
        iterations: Number of learning iterations.
        do_plot: If True, plot during learning.
        plot_per_its: Plot per this many learning iterations.
        init_noise_level: Standard deviation of the Gaussian noise added to the initial phase_response guess.
        sigma: Standard deviation of gaussian kernel for computing smoothness score.
        smooth_loss_factor: Factor for multiplying smoothness loss.

    Returns:
        nonlinearity, lr, phase[i], amplitude[i]
    """

    # Initial guess:
    # normalize measurements to have std=1, then subtract the mean
    # then initialize a = 1.0 and E=(peak-peak(measurements)/2)^(1/non_linearity)
    measurements = torch.tensor(measurements, dtype=torch.float32)
    measurements = measurements / measurements.std()
    measurements.detach()
    E_abs_init = (0.5 * (measurements.max() - measurements.min())).pow(1 / nonlinearity)
    E = E_abs_init * torch.exp(1j * torch.linspace(0, phase_stroke_init, 256))
    E.detach()
    E.requires_grad_(True)
    a = torch.tensor(1.0, requires_grad=True, dtype=torch.complex64)
    b = torch.tensor(1.0, requires_grad=True, dtype=torch.complex64)
    s_bg = torch.tensor(0.0, requires_grad=True)
    nonlinearity = torch.tensor(nonlinearity, dtype=torch.float32, requires_grad=True)

    # Initialize parameters and optimizer
    params = [{"lr": learning_rate, "params": [E, a, b, s_bg]}, {"lr": learning_rate * 0.1, "params": [nonlinearity]}]
    optimizer = torch.optim.Adam(params, lr=learning_rate, amsgrad=True, betas=(0.95, 0.9995))
    progress_bar = tqdm(total=iterations)

    def model(E, a, b, s_bg):
        E0 = E[gray_values0].view(-1, 1)
        E1 = E[gray_values1].view(1, -1)
        I_excite = (a * E0 + b * E1).abs().pow(2)
        signal_intenstity = I_excite.pow(nonlinearity) + s_bg
        return signal_intenstity

    for it in range(iterations):
        feedback_predicted = model(E, a, b, s_bg)
        loss_meas = (measurements - feedback_predicted).pow(2).mean()
        loss_reg = balance_factor * (a - b).abs().pow(2)
        loss_smooth = smooth_loss_factor * torch.std(abs(E))
        loss = loss_meas + loss_reg + loss_smooth

        # Gradient descent step
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if do_plot and (it % plot_per_its == 0 or it == 0 or it == iterations - 1):
            if it == 0:
                plt.figure(figsize=(13, 4))
            else:
                plt.clf()
            plt.subplot(1, 3, 1)
            plot_field_response(E)
            plot_feedback_fit(measurements, feedback_predicted, gray_values0, gray_values1)
            plt.title(f"feedback: {loss_meas:.3g}, smoothness: {loss_smooth:.3g}, lr_reg: {loss_reg:.3g}" +
                      f"\na: {a:.3g}, b: {b:.3g}, s_bg: {s_bg:.3g}")
            plt.pause(0.01)

        progress_bar.update()

    # split phase and amplitude, and unwrap phase
    Ed = (E - 0.5 * (E.real.max() + E.real.min()) - 0.5j * (E.imag.max() + E.imag.min())).detach()
    amplitude = Ed.abs()
    phase = np.unwrap(np.angle(Ed))
    phase *= np.sign(phase[-1] - phase[0])
    phase -= phase.mean()

    if do_plot and do_end_plot:
        plt.figure(figsize=(14, 4.3))
        plt.subplots_adjust(left=0.05, right=0.98, bottom=0.15)
        plot_result_feedback_fit(measurements, feedback_predicted, gray_values0, gray_values1)

    return nonlinearity.item(), a.item(), b.item(), s_bg.item(), phase, amplitude.detach()

