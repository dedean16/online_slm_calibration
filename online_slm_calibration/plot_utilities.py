import numpy as np
import torch
from matplotlib import pyplot as plt


def plot_results_ground_truth(gray_values, phase, amplitude, gray_values_ref, phase_ref, phase_ref_err, amplitude_ref):
    phase = phase - phase[50] + phase_ref[50]

    plt.figure(figsize=(6, 8))
    plt.subplots_adjust(left=0.20, hspace=0.35, top=0.95, bottom=0.08)

    plt.subplot(2, 1, 1)
    plt.errorbar(gray_values_ref, phase_ref, yerr=phase_ref_err, color='C0', label='Reference')      # plot phase with std error
    plt.plot(gray_values, phase, '+', color='C1', label='Our fit')
    plt.xlabel('Gray value')
    plt.ylabel('Phase')
    plt.title('a. Phase response')
    plt.legend()

    plt.subplot(2, 1, 2)
    rel_amplitude_ref = amplitude_ref / amplitude_ref.mean()
    rel_amplitude = amplitude / amplitude.mean()
    plt.plot(rel_amplitude_ref, color='C0', label='Reference')
    plt.plot(gray_values, rel_amplitude, '+', color='C1', label='Our fit')
    plt.xlabel('Gray value')
    plt.ylabel('Normalized amplitude')
    plt.ylim((0, 1.1 * np.maximum(rel_amplitude.max(), rel_amplitude_ref.max())))
    plt.title('b. Normalized amplitude response')
    plt.legend()
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
    plt.imshow((feedback_measurements.detach() - feedback.detach()).abs(), extent=extent, interpolation="nearest")
    plt.title("c. Absolute difference")
    plt.xlabel('$g_B$')
    plt.ylabel('$g_A$')
    plt.colorbar()
