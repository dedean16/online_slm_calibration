import numpy as np
import torch
from matplotlib import pyplot as plt


def plot_results_ground_truth(gray_values, phase, phase_std, amplitude, amplitude_std,
                              ref_gray_values, ref_phase, ref_phase_std, ref_amplitude, ref_amplitude_std):
    lightC0 = '#a8d3f0'
    lightC1 = '#ffc899'

    # Plot calibration curves of phase and amplitude
    plt.figure(figsize=(9, 5))
    plt.subplots_adjust(left=0.1, right=0.95, hspace=0.35, wspace=0.35, top=0.92, bottom=0.12)

    plt.subplot(1, 2, 1)
    plt.errorbar(ref_gray_values, ref_phase, yerr=ref_phase_std, color='C0', ecolor=lightC0, label='Reference')
    plt.errorbar(gray_values, phase, yerr=phase_std, color='C1', ecolor=lightC1, label='Inline (ours)')
    plt.xlabel('Gray value')
    plt.ylabel('Phase (rad)')
    plt.title('a. Phase response')
    plt.legend()

    plt.subplot(1, 2, 2)
    rel_ref_amplitude = ref_amplitude / ref_amplitude.mean()
    rel_ref_amplitude_std = ref_amplitude_std / ref_amplitude.mean()
    rel_amplitude = amplitude / amplitude.mean()
    plt.errorbar(gray_values, rel_amplitude, yerr=amplitude_std, color='C1', ecolor=lightC1, label='Inline (ours)')
    plt.errorbar(ref_gray_values, rel_ref_amplitude, yerr=rel_ref_amplitude_std, color='C0', ecolor=lightC0, label='Reference')
    plt.xlabel('Gray value')
    plt.ylabel('Normalized amplitude')
    plt.ylim((0, 1.1 * np.maximum(rel_amplitude.max(), rel_ref_amplitude.max())))
    plt.title('b. Normalized amplitude response')
    plt.legend()

    # Phase difference
    plt.figure()
    plt.subplots_adjust(left=0.2, right=0.95, top=0.9, bottom=0.15)
    plt.plot(np.diff(ref_phase))
    plt.title('Phase response slope')
    plt.xlabel('Gray level')
    plt.ylabel('$d\\phi/dg$')

    # Plot field response in complex plane
    plt.figure()
    amplitude_norm = amplitude / amplitude.mean()
    E_norm = amplitude_norm * np.exp(1.0j * phase)
    ref_amplitude_norm = ref_amplitude / ref_amplitude.mean()
    E_ref_norm = ref_amplitude_norm * np.exp(1.0j * ref_phase)
    plt.plot(E_ref_norm.real, E_ref_norm.imag, label="Reference")
    plt.plot(E_norm.real, E_norm.imag, label="Our method")
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
    plt.title("a. Measured signal")
    plt.xlabel('$g_B$')
    plt.ylabel('$g_A$')
    plt.colorbar()

    plt.subplot(1, 3, 2)
    plt.imshow(feedback.detach(), extent=extent, interpolation="nearest", vmin=vmin, vmax=vmax)
    plt.title("b. Fit signal")
    plt.xlabel('$g_B$')
    plt.ylabel('$g_A$')
    plt.colorbar()

    plt.subplot(1, 3, 3)
    plt.imshow((feedback_measurements.detach() - feedback.detach()).abs(), extent=extent, interpolation="nearest")
    plt.title("c. Residual")
    plt.xlabel('$g_B$')
    plt.ylabel('$g_A$')
    plt.colorbar()
