from matplotlib import pyplot as plt


def plot_results_ground_truth(phase, amplitude, phase_gt):
    phase = phase - phase.mean() + phase_gt.mean()
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(phase_gt, color='C0', label='Ground truth')
    plt.plot(phase, '--', color='C1', label='Predicted')
    plt.xlabel('Gray value')
    plt.ylabel('Phase response')
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.plot(amplitude, color='C0', label='Amplitude')
    plt.show()
