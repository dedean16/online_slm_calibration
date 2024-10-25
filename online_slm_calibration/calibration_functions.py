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


def phase_correlation(phase1, phase2):
    """
    Compute the field correlation between two phase curves.
    """
    return field_correlation(torch.exp(1j * phase1), torch.exp(1j * phase2))

def detrend(gray_value0, gray_value1, measurements: np.ndarray):
    m = torch.tensor(measurements.flatten(order="F"))
    m = m / m.abs().mean()

    # locate elements for which gv0 == gv1. These are measured twice and should be equal except for noise and photobleaching.
    gv0 = np.asarray(gray_value0)
    sym_selection = [np.nonzero(gv0 == gv1)[0][0].item() for gv1 in gray_value1]

    learning_rate = 0.01
    offset = torch.tensor(0.1 * (m.max() - m.min()), dtype=torch.float32, requires_grad=True)
    decay = torch.tensor(0.1 / len(m), dtype=torch.float32, requires_grad=True)
    t = torch.tensor(range(len(m)))
    def compensate_bleaching():
        return (m - offset) * torch.exp(decay * t)

    params = [{"lr": learning_rate, "params": [offset]}, {"lr": learning_rate / len(m), "params": [decay]}]
    optimizer = torch.optim.Adam(params, lr=learning_rate, amsgrad=True)

    for it in range(100):
        c = compensate_bleaching()
        c = c.reshape((measurements.shape[1], measurements.shape[0]))[:, sym_selection]
        loss = (c - c.t()).pow(2).mean() / c.pow(2).mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    m = compensate_bleaching().detach().numpy()
    measurements = m.reshape(measurements.shape, order='F')

    # feedback_meas[90,:] = 0.5 * (feedback_meas[89,:]+feedback_meas[91,:])
    # feedback_meas[82,:] = 0.5 * (feedback_meas[81,:]+feedback_meas[83,:])
    # extent = (gv1.min()-0.5, gv1.max()+0.5, gv0.min()-0.5, gv0.max()+0.5)
    # plt.imshow(feedback_meas, extent=extent)
    # plt.show()
    #
    ff = measurements[sym_selection, :]
    plt.figure()
    plt.imshow(ff)
    plt.show()

    plt.plot(m)
    plt.show()
    return measurements


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


def plot_phase_response(phase_response_per_gv):
    plt.plot(phase_response_per_gv.detach())
    plt.xlabel("Gray value")
    plt.ylabel("Phase (rad)")
    plt.title("Predicted phase response")


def plot_feedback_fit(feedback_measurements, feedback, gray_values0, gray_values1):
    plt.subplot(1, 3, 2)
    extent = (gray_values1.min(), gray_values1.max(), gray_values0.min(), gray_values0.max())
    vmin = torch.minimum(feedback_measurements.min(), feedback.min())
    vmax = torch.maximum(feedback_measurements.max(), feedback.max())
    plt.imshow(feedback_measurements.detach(), extent=extent, interpolation="nearest", vmin=vmin, vmax=vmax)
    plt.title("Measured feedback")
    plt.colorbar()

    plt.subplot(1, 3, 3)
    plt.imshow(feedback.detach(), extent=extent, interpolation="nearest", vmin=vmin, vmax=vmax)
    plt.title("Predicted feedback")
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
    with open(filepath_lut, "r") as file:
        numbers = [float(line.strip()) for line in file.readlines()]
    return torch.tensor(numbers) / scaling


def learn_lut(
    gray_values0: tt,
    gray_values1: tt,
    feedback_measurements: tt,
    nonlinearity=2,
    iterations: int = 500,
    init_noise_level=0.01,
    do_plot: bool = False,
    plot_per_its: int = 10,
    do_end_plot: bool = True,
    smooth_factor=2.0,
    learning_rate=0.001,
    phase_response_per_gv_init=None,
) -> tt:
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
    # Initial guess phase response
    if phase_response_per_gv_init is None:
        phase_response_per_gv = torch.linspace(0, 2 * np.pi, 256) + torch.randn(256) * init_noise_level
    else:
        phase_response_per_gv = phase_response_per_gv_init.detach()
    phase_response_per_gv.requires_grad = True

    # Initial guess a and b
    ### TODO: rescale feedback measurements instead of guessing order of magnitude of a and b
    a = torch.tensor(feedback_measurements.mean(), requires_grad=True)
    b = torch.tensor(3 * feedback_measurements.std(), requires_grad=True)

    # Initialize parameters and optimizer
    params = [{"lr": learning_rate, "params": [phase_response_per_gv, a, b]}]
    optimizer = torch.optim.Adam(params, lr=learning_rate, amsgrad=True)

    progress_bar = tqdm(total=iterations)

    if do_plot:
        plt.figure(figsize=(13, 4))

    for it in range(iterations):
        feedback_predicted = predict_feedback(gray_values0, gray_values1, a, b, phase_response_per_gv, nonlinearity)
        normalized_feedback_errors = (feedback_measurements - feedback_predicted) / feedback_measurements.mean()
        feedback_mse = normalized_feedback_errors.abs().pow(2).mean()
        smoothness_mse = smooth_factor * phase_response_per_gv.diff(n=2).pow(2).mean()
        error = feedback_mse + smoothness_mse

        # Gradient descent step
        error.backward()
        optimizer.step()
        optimizer.zero_grad()

        if do_plot and (it % plot_per_its == 0 or it == 0):
            plt.clf()
            plt.subplot(1, 3, 1)
            plot_phase_response(phase_response_per_gv)
            plot_feedback_fit(feedback_measurements, feedback_predicted, gray_values0, gray_values1)
            plt.title(f"feedback mse: {feedback_mse:.3g}, smoothness mse: {smoothness_mse:.3g}\na: {a:.3g}, b: {b:.3g}")
            plt.pause(0.01)

        progress_bar.update()

    if do_end_plot:
        if do_plot:
            plt.clf()
        else:
            plt.figure(figsize=(13, 4))
        plt.subplot(1, 3, 1)

        plot_phase_response(phase_response_per_gv)
        plot_feedback_fit(feedback_measurements, feedback_predicted, gray_values0, gray_values1)
        plt.pause(0.1)

    return phase_response_per_gv.detach()


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


def unwrap(phase):
    ### TODO: The current approach does not guarantee |phase steps| < π. This will work for most practical cases,
    ### TODO: but should be done recursively to work for any case.
    ### ??
    
    dphase = torch.diff(phase)
    dphase = torch.where(dphase > np.pi, dphase - 2 * np.pi, dphase)
    dphase = torch.where(dphase < -np.pi, dphase + 2 * np.pi, dphase)
    phase = torch.cat((torch.tensor([0]), torch.cumsum(dphase, dim=0)), dim=0)
    if phase[-1] < 0:
        phase = -phase  # ensure phase is increasing
    return phase


def learn_field(
    *,
    gray_values0,
    gray_values1,
    measurements,
    nonlinearity=1,
    iterations: int = 50,
    do_plot: bool = False,
    plot_per_its: int = 10,
    learning_rate=0.1,
    phase_stroke_init=2.5 * torch.pi,
    balance_factor=1.0,
    smooth_loss_factor=1.0,
) -> tuple[float, float, np.array, np.ndarray]:
    """
    Learn the phase lookup table from dual phase stepping measurements.
    This function uses the model:

        I[i,j] = |E[i] + lr_ratio * E[j]|^(non_linearity)

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
        nonlinearity, lr, phase[i], ampltude[i]
    """

    # Initial guess:
    # normalize measurements to have std=1, then subtract the mean
    # then initialize a = 1.0 and E=(peak-peak(measurements)/2)^(1/non_linearity)
    measurements = torch.tensor(measurements, dtype=torch.float32)
    measurements = measurements / measurements.std()
    measurements -= measurements.mean()
    measurements.detach()
    b = (0.5 * (measurements.max() - measurements.min())).pow(1 / nonlinearity)
    E = b * torch.exp(1j * torch.linspace(0, phase_stroke_init, 256))
    E.detach()
    E.requires_grad_(True)
    lr = torch.tensor(1.0, requires_grad=True, dtype=torch.complex64)
    nonlinearity = torch.tensor(nonlinearity, dtype=torch.float32, requires_grad=True)

    # Initialize parameters and optimizer
    params = [{"lr": learning_rate, "params": [lr, E]}, {"lr": learning_rate * 0.1, "params": [nonlinearity]}]
    optimizer = torch.optim.Adam(params, lr=learning_rate, amsgrad=True, betas=(0.95, 0.9995))
    progress_bar = tqdm(total=iterations)

    def model(E, lr):
        E0 = E[gray_values0].view(-1, 1)
        E1 = E[gray_values1].view(1, -1)
        signal_intenstity = (E0 + lr * E1).abs().pow(2 * nonlinearity)
        return signal_intenstity - signal_intenstity.mean()

    for it in range(iterations):
        feedback_predicted = model(E, lr)
        loss_meas = (measurements - feedback_predicted).pow(2).mean()
        loss_reg = balance_factor * (lr - 1.0).abs().pow(2)
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
            plot_phase_response(torch.angle(E))
            plot_feedback_fit(measurements, feedback_predicted, gray_values0, gray_values1)
            plt.title(f"feedback: {loss_meas:.3g}, smoothness: {loss_smooth:.3g}, lr_reg:{loss_reg:.3g}\nlr: {lr:.3g}")
            plt.pause(0.01)

        progress_bar.update()

    # split phase and amplitude, and unwrap phase
    E = E - 0.5 * (E.real.max() + E.real.min()) - 0.5j * (E.imag.max() + E.imag.min())  # experimental
    amplitude = E.abs()
    phase = unwrap(torch.angle(E))

    return nonlinearity.item(), lr.item(), phase.detach().numpy(), amplitude.detach().numpy()


def grow_learn_field(gray_values0: tt, gray_values1: tt, measurements: tt, gray_value_slice_size=16,
                     **kwargs) -> tuple[float, Tensor, Tensor]:
    """
    Learn the phase lookup table from dual phase stepping measurements, piece by piece.

    Args:
        gray_values0:
        gray_values1: Same as gray_values1, for the second group.
        measurements:
        nonlinearity: Nonlinearity number. 1 = linear, 2 = 2PEF, 3 = 3PEF, etc., 0 = detector is broken :)
        iterations: Number of learning iterations.
        do_plot: If True, plot during learning.
        plot_per_its: Plot per this many learning iterations.
        init_noise_level: Standard deviation of the Gaussian noise added to the initial phase_response guess.
    Returns:
    """
    slice_iterations = int(np.ceil(np.maximum(gray_values0.max(), gray_values1.max()) / gray_value_slice_size))
    phase_response_per_gv = torch.linspace(0.0, 2 * np.pi, 256)
    phase = torch.linspace(0.0, 2*np.pi, 256)
    amplitude = torch.ones(256)
    E = amplitude * torch.exp(1j * phase)
    B = 1.0

    # Learn iteratively larger slices of the measurement data
    for slice_it in range(slice_iterations):
        # Crop to the part of the measurements that we will learn this iteration
        c = (slice_it + 1) * gray_value_slice_size
        crop_index0 = (gray_values0 < c).sum()
        crop_index1 = (gray_values1 < c).sum()
        cropped_gray_values0 = gray_values0[0:crop_index0]
        cropped_gray_values1 = gray_values1[0:crop_index1]
        cropped_feedback_measurements = measurements[0:crop_index0, 0:crop_index1]

        cropped_phase_response_per_gv_init = learn_lut(
            gray_values0=cropped_gray_values0,
            gray_values1=cropped_gray_values1,
            feedback_measurements=cropped_feedback_measurements,
            phase_response_per_gv_init=phase_response_per_gv[:gray_value_crop_size],
            **kwargs,
        )
        # Learn with new slice
        B, cropped_phase, cropped_amplitude = \
            learn_field(gray_values0=cropped_gray_values0,
                        gray_values1=cropped_gray_values1,
                        measurements=cropped_feedback_measurements,
                        E_init=E[:c],
                        B_init=B,
                        **kwargs)

        phase[:c] = cropped_phase
        phase[c:] = cropped_phase[-1]
        amplitude[:c] = cropped_amplitude
        amplitude[c:] = cropped_amplitude[-1]
        E = amplitude * np.exp(1j * phase)

    return B, phase, amplitude
