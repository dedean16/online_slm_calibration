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
