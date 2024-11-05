import torch
from matplotlib import pyplot as plt


def plot_results_ground_truth(phase, amplitude, gray_values, phase_gt):
    phase = phase - phase[50] + phase_gt[50]
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(phase_gt, color='C0', label='Ground truth')
    plt.plot(gray_values, phase, '+', color='C1', label='Predicted')
    plt.xlabel('Gray value')
    plt.ylabel('Phase response')
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.plot(gray_values, amplitude, '.', color='C0', label='Amplitude')
    plt.show()


def plot_field_response(field_response_per_gv):
    plt.plot(field_response_per_gv.detach().abs(), label='$A$')
    plt.plot(field_response_per_gv.detach().angle(), label='$\\phi$')
    plt.xlabel("Gray value")
    plt.ylabel("Phase (rad) | Relative amplitude")
    plt.legend()
    plt.title("Predicted field response")


def plot_feedback_fit(feedback_measurements, feedback, gray_values0, gray_values1):
    plt.subplot(1, 3, 2)
    extent = (gray_values1.min()-0.5, gray_values1.max()+0.5, gray_values0.max()+0.5, gray_values0.min()-0.5)
    vmin = torch.minimum(feedback_measurements.min(), feedback.min())
    vmax = torch.maximum(feedback_measurements.max(), feedback.max())
    plt.imshow(feedback_measurements.detach(), extent=extent, interpolation="nearest", vmin=vmin, vmax=vmax)
    plt.title("Measured feedback")
    plt.colorbar()

    plt.subplot(1, 3, 3)
    plt.imshow(feedback.detach(), extent=extent, interpolation="nearest", vmin=vmin, vmax=vmax)
    plt.title("Predicted feedback")
    plt.colorbar()


def plot_result_feedback_fit(feedback_measurements, feedback, gray_values0, gray_values1):
    extent = (gray_values1.min()-0.5, gray_values1.max()+0.5, gray_values0.max()+0.5, gray_values0.min()-0.5)
    vmin = torch.minimum(feedback_measurements.min(), feedback.min())
    vmax = torch.maximum(feedback_measurements.max(), feedback.max())

    plt.subplot(1, 3, 1)
    plt.imshow(feedback_measurements.detach(), extent=extent, interpolation="nearest", vmin=vmin, vmax=vmax)
    plt.title("a. Measured signal power")
    plt.xlabel('$g_B$')
    plt.ylabel('$g_A$')
    plt.colorbar()

    plt.subplot(1, 3, 2)
    plt.imshow(feedback.detach(), extent=extent, interpolation="nearest", vmin=vmin, vmax=vmax)
    plt.title("b. Fit signal power")
    plt.xlabel('$g_B$')
    plt.ylabel('$g_A$')
    plt.colorbar()

    plt.subplot(1, 3, 3)
    plt.imshow(feedback_measurements.detach() - feedback.detach(), extent=extent, interpolation="nearest")
    plt.title("c. Difference")
    plt.xlabel('$g_B$')
    plt.ylabel('$g_A$')
    plt.colorbar()
