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
    plt.xlabel('Gray value')
    plt.ylabel('Phase (rad)')
    plt.title('Predicted phase response')


def plot_feedback_fit(feedback_measurements, feedback, gray_values0, gray_values1):
    plt.subplot(1, 3, 2)
    extent = (gray_values1.min(), gray_values1.max(), gray_values0.max(), gray_values0.min())
    vmin = torch.minimum(feedback_measurements.min(), feedback.min())
    vmax = torch.maximum(feedback_measurements.max(), feedback.max())
    plt.imshow(feedback_measurements.detach(), extent=extent, interpolation='nearest', vmin=vmin, vmax=vmax)
    plt.title('Measured feedback')
    plt.xlabel('Gray value group 1')
    plt.ylabel('Gray value group 0')
    plt.colorbar()

    plt.subplot(1, 3, 3)
    plt.imshow(feedback.detach(), extent=extent, interpolation='nearest', vmin=vmin, vmax=vmax)
    plt.title('Predicted feedback')
    plt.xlabel('Gray value group 1')
    plt.ylabel('Gray value group 0')
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


def learn_lut(gray_values0: tt, gray_values1: tt, feedback_measurements: tt, nonlinearity=2, iterations: int = 500,
              init_noise_level=0.01, do_plot: bool = False, plot_per_its: int = 10,
              smooth_factor=2.0, learning_rate=0.001, phase_response_per_gv_init=None) -> tt:
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
        phase_response_per_gv = torch.linspace(0, 2*np.pi, 256) + torch.randn(256) * init_noise_level
    else:
        phase_response_per_gv = phase_response_per_gv_init.detach()
    phase_response_per_gv.requires_grad = True

    # Initial guess a and b
    ### TODO: rescale feedback measurements instead of guessing order of magnitude of a and b
    a = torch.tensor(feedback_measurements.mean(), requires_grad=True)
    b = torch.tensor(3*feedback_measurements.std(), requires_grad=True)

    # Initialize parameters and optimizer
    params = [{'lr': learning_rate, 'params': [phase_response_per_gv, a, b]}]
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

        if do_plot and (it % plot_per_its == 0 or it == 0 or it == (iterations-1)):
            plt.clf()
            plt.subplot(1, 3, 1)
            plot_phase_response(phase_response_per_gv)
            plot_feedback_fit(feedback_measurements, feedback_predicted, gray_values0, gray_values1)
            plt.title(f'feedback mse: {feedback_mse:.3g}, smoothness mse: {smoothness_mse:.3g}\na: {a:.3g}, b: {b:.3g}')
            plt.pause(0.01)

        progress_bar.update()

    return phase_response_per_gv.detach(), feedback_predicted


def grow_learn_lut(gray_values0: tt, gray_values1: tt, feedback_measurements: tt, gray_value_slice_size=16,
                   do_end_plot=False, **kwargs) -> tt:
    """
    Learn the phase lookup table from dual phase stepping measurements, piece by piece.

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
    slice_iterations = int(np.ceil(np.maximum(gray_values0.max(), gray_values1.max()) / gray_value_slice_size))
    phase_response_per_gv = torch.linspace(0.0, 2*np.pi, 256)

    # Normalize feedback measurements
    feedback_meas_norm = (feedback_measurements - feedback_measurements.mean()) \
                         / feedback_measurements.std()

    for slice_it in range(slice_iterations):
        # Crop to the part of the measurements that we will learn this iteration
        gray_value_crop_size = (slice_it + 1) * gray_value_slice_size
        crop_index0 = (gray_values0 < gray_value_crop_size).sum()
        crop_index1 = (gray_values1 < gray_value_crop_size).sum()
        cropped_gray_values0 = gray_values0[0:crop_index0]
        cropped_gray_values1 = gray_values1[0:crop_index1]
        cropped_feedback_measurements = feedback_meas_norm[0:crop_index0, 0:crop_index1]

        # Fit newly cropped part
        cropped_phase_response_per_gv_init, feedback_predicted = \
            learn_lut(gray_values0=cropped_gray_values0,
                      gray_values1=cropped_gray_values1,
                      feedback_measurements=cropped_feedback_measurements,
                      phase_response_per_gv_init=phase_response_per_gv[:gray_value_crop_size],
                      **kwargs)

        # Fill in newly learned cropped part into full phase response array
        phase_response_per_gv[:gray_value_crop_size] = cropped_phase_response_per_gv_init
        phase_response_per_gv[gray_value_crop_size:] = cropped_phase_response_per_gv_init[-1]   # Extrapolate constant

    if do_end_plot:
        plt.figure(figsize=(13, 4))
        plt.subplot(1, 3, 1)
        plot_phase_response(phase_response_per_gv)
        plot_feedback_fit(feedback_meas_norm, feedback_predicted, gray_values0, gray_values1)
        plt.pause(0.1)

    return phase_response_per_gv
